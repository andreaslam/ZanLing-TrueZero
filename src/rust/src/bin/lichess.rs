use cozy_chess::Board;
use crossbeam::thread;
use dotenv::dotenv;
use flume::Sender;
use futures_util::StreamExt;
use rusqlite::{params, Connection, OptionalExtension};
use tokio::{
    sync::Mutex,
    time::{sleep, timeout, Duration},
};

use litchee::{
    api::gameplay::{
        board::{LichessBoardEvent, LichessGameState, LichessIncomingEvent},
        challenges::{LichessChallengeColor, LichessChallengeStatus},
        games::LichessGameStatusName,
    },
    model::{LichessColor, LichessPerfs, LichessTitle},
    LichessClient,
};

use lru::LruCache;
use shakmaty::{fen::Fen, uci::UciMove, CastlingMode, Chess, Color, EnPassantMode, Position};

use tzrust::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    dataformat::ZeroEvaluationAbs,
    executor::{executor_static, Message, Packet},
    mcts::get_move,
    mcts_trainer::{EvalMode, TypeRequest::TrainerSearch},
    settings::{CPUCTSettings, FPUSettings, MovesLeftSettings, PSTSettings, SearchSettings},
    uci::time_to_nodes,
};

use std::{
    collections::{HashMap, HashSet},
    env,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

const MOVES_TO_GO: u64 = 30;
const LRU_CACHE_SIZE: usize = 100_000;

const MAX_ENGINE_TIME_MS: Option<u128> = Some(100_000);
const MAX_ENGINE_NODES: u64 = 5_000_000_000;
const FALLBACK_ENGINE_NODES: u64 = 1600;

const NO_TIME_LIMIT_ENGINE_NODES: u64 = 2_000;
const NO_TIME_LIMIT_WALL_TIMEOUT_MS: u64 = 60_000;
const ENGINE_WALL_TIMEOUT_MS: u64 = 12_000_000;

const DRAW_ACCEPT_TIME_MS: u64 = 10_000;
const DRAW_MAX_EVAL: f64 = 0.0;

const GAME_STREAM_RECONNECT_DELAY_MS: u64 = 1_000;
const GAME_STREAM_MAX_RECONNECTS: u32 = 10;

const EVENT_STREAM_RECONNECT_DELAY_MS: u64 = 2_000;

/*
 * Matchmaking.
 *
 * The event stream and matchmaking run in separate Tokio tasks.
 */
const ONLINE_BOT_SCAN_INTERVAL_SECS: u64 = 60;
const ONLINE_BOT_CHALLENGE_COOLDOWN_SECS: u64 = 1_800;

const MAX_BOT_CHALLENGES_PER_SCAN: usize = 5;
const BOT_CHALLENGE_SEND_DELAY_MS: u64 = 2_000;

/*
 * Human matchmaking.
 */
const HUMAN_CHALLENGE_COOLDOWN_DAYS: i64 = 7;

const MAX_HUMAN_CHALLENGES_PER_SCAN: usize = 5;
const HUMAN_CHALLENGE_SEND_DELAY_MS: u64 = 2_000;

const ONLINE_HUMAN_PAGE_URL: &str = "https://lichess.org/player";

const MAX_ONLINE_HUMAN_CANDIDATES: usize = 300;

/*
 * Persistent graph discovery.
 *
 * The database is deliberately local to the bot.
 *
 * The graph is:
 *
 *     seed bot
 *         |
 *         v
 *     game history
 *         |
 *         v
 *     opponent
 *         |
 *         v
 *     database
 *         |
 *         v
 *     expand opponent's games
 *         |
 *         +----> more opponents
 *
 * This means the discovered player set survives process restarts.
 */
const PLAYER_GRAPH_DB_PATH: &str = "lichess_player_graph.sqlite3";

const GRAPH_GAMES_PER_PLAYER: usize = 25;

/*
 * How many already-known players to expand during each matchmaking scan.
 *
 * Keeping this bounded is important because otherwise a large graph could
 * make one scan last indefinitely.
 */
const MAX_GRAPH_PLAYERS_PER_SCAN: usize = 10;

/*
 * Do not re-fetch the same player's recent games more frequently than this.
 *
 * Newly discovered users are still expanded immediately because they have no
 * row in `player_expansions`.
 */
const BOT_CHALLENGE_RATED: bool = true;
const HUMAN_CHALLENGE_RATED: bool = BOT_CHALLENGE_RATED;

/*
 * Outgoing matchmaking time controls.
 *
 * A control is selected independently for every challenge. Keeping the
 * presets here makes the supported range explicit and easy to adjust.
 */
struct ActiveGameGuard(Arc<AtomicUsize>);

impl ActiveGameGuard {
    fn new(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Self(counter)
    }
}

impl Drop for ActiveGameGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let token = env::var("LICHESS_API_KEY").expect("LICHESS_API_KEY must be set");

    let net_path = String::from(r"C:\Users\andre\RemoteFolder\ZanLing-TrueZero\tz_163.pt");

    let (tensor_exe_send, tensor_exe_recv) = flume::unbounded::<Packet>();

    let (ctrl_send, ctrl_recv) = flume::unbounded::<Message>();

    let game_tensor_send = tensor_exe_send.clone();

    thread::scope(|scope| {
        let executor_handle = scope
            .builder()
            .name("executor-lichess".to_string())
            .spawn(move |_| {
                println!("Starting executor thread.");

                executor_static(net_path, tensor_exe_recv, ctrl_recv, 1);

                println!("Executor thread stopped.");
            })
            .expect("Failed to spawn executor thread");

        let game_handle = scope
            .builder()
            .name("game-loop-lichess".to_string())
            .spawn(move |_| {
                println!("Starting game loop thread.");

                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create game-loop Tokio runtime");

                runtime.block_on(async move {
                    if let Err(error) = game_loop(token, game_tensor_send).await {
                        eprintln!("Game loop error: {error}");
                    }
                });

                println!("Game loop thread stopped.");
            })
            .expect("Failed to spawn game loop thread");

        executor_handle.join().expect("Executor thread panicked");

        game_handle.join().expect("Game thread panicked");

        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    })
    .expect("Thread scope failed")?;

    let _ = ctrl_send;

    Ok(())
}

#[path = "lichess/game.rs"]
mod game;
#[path = "lichess/graph.rs"]
mod graph;
#[path = "lichess/runtime.rs"]
mod runtime;

use game::*;
use graph::*;
use runtime::*;
use tzrust::lichess_time_control::*;
