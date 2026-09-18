use cozy_chess::Board;
use crossbeam::thread;
use dotenv::dotenv;
use flume::{Receiver, Sender};
use futures_util::StreamExt;
use rusqlite::{params, Connection, OptionalExtension};
use tokio::{
    sync::{watch, Mutex, OwnedSemaphorePermit, Semaphore},
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
    data_path, data_path_str,
    dataformat::{Position as TrainingPosition, Simulation, ZeroEvaluationAbs, ZeroEvaluationPov},
    executor::{executor_main, Packet},
    fileformat::BinaryOutput,
    mcts::get_move,
    mcts_trainer::{EvalMode, TypeRequest::TrainerSearch},
    message_types::{DataFileType, Entity, MessageServer, MessageType, Statistics},
    selfplay::{CollectorMessage, DataGen},
    settings::{CPUCTSettings, FPUSettings, MovesLeftSettings, PSTSettings, SearchSettings},
    uci::{time_to_nodes, UCIMsg},
};

use std::io::Write;
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

const MAX_ENGINE_TIME_MS: Option<u128> = Some(10_000);
const MAX_ENGINE_NODES: u64 = 5_000_000_000;
const FALLBACK_ENGINE_NODES: u64 = 1600;
const DEFAULT_MAX_CONCURRENT_GAMES: usize = 1;

const NO_TIME_LIMIT_ENGINE_NODES: u64 = 2_000;
const NO_TIME_LIMIT_WALL_TIMEOUT_MS: u64 = 60_000;
const ENGINE_WALL_TIMEOUT_MS: u64 = 10_000;
// Inference can be delayed while self-play batches are being flushed. Give a
// search enough time to finish instead of converting a healthy search into a
// first-legal-move fallback.
const ENGINE_TIMEOUT_GRACE_MS: u64 = 5_000;
const MIN_SEARCH_TIME_MS: u64 = 25;
const SHORT_TIME_RESERVE_MS: u64 = 35;
const LONG_TIME_RESERVE_FRACTION: f64 = 0.15;

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
    let num_executors = env_usize("TZ_NUM_EXECUTORS", 2).max(1);
    // Keep the Lichess self-play workload comparable to main.rs unless the
    // operator explicitly overrides it.
    let batch_size = env_usize("TZ_BATCH_SIZE", 1024).max(1);
    let num_generators = env_usize(
        "TZ_NUM_GENERATORS",
        num_executors
            .saturating_mul(batch_size)
            .saturating_mul(2),
    )
    .max(1);

    let mut stream = loop {
        match std::net::TcpStream::connect("127.0.0.1:38475") {
            Ok(stream) => break stream,
            Err(_) => std::thread::sleep(std::time::Duration::from_secs(1)),
        }
    };
    let init = MessageServer {
        purpose: MessageType::Initialise(Entity::RustDataGen),
    };
    writeln!(stream, "{}", serde_json::to_string(&init)?)?;

    let mut net_senders = Vec::with_capacity(num_executors);
    let mut net_receivers = Vec::with_capacity(num_executors);
    for _ in 0..num_executors {
        let (sender, receiver) = flume::bounded::<String>(2);
        net_senders.push(sender);
        net_receivers.push(receiver);
    }
    let (executor_ready_send, executor_ready_recv) = flume::unbounded::<bool>();
    // A queue smaller than the executor batch size forces handle_requests()
    // to flush partial batches after its timeout. That produces many small
    // GPU launches: utilization can look high while aggregate NPS collapses.
    // Match main.rs by keeping enough pending requests to fill the configured
    // batches. Pause self-play before starting a live game, so this backlog is
    // bounded and does not grow while the game is being played.
    let default_queue_capacity = num_generators
        .max(batch_size.saturating_mul(num_executors))
        .max(1);
    let queue_capacity =
        env_usize("TZ_TENSOR_QUEUE_CAPACITY", default_queue_capacity).max(1);
    let (tensor_send, tensor_recv) = flume::bounded::<Packet>(queue_capacity);
    let (collector_send, collector_recv) = flume::bounded::<CollectorMessage>(num_generators);
    let (id_send, id_recv) = flume::bounded::<usize>(1);
    let command_stream = stream.try_clone()?;
    let collect_stream = stream.try_clone()?;

    thread::scope(|scope| {
        let _command_handle = scope
            .builder()
            .name("server-commander-lichess".to_string())
            .spawn(move |_| {
                commander_main(command_stream, net_senders, id_send);
            })
            .expect("Failed to spawn executor thread");

        for (executor_id, net_receiver) in net_receivers.into_iter().enumerate() {
            let executor_collector = collector_send.clone();
            let tensor_receiver = tensor_recv.clone();
            let executor_ready = executor_ready_send.clone();
            scope
                .builder()
                .name(format!("executor-{executor_id}"))
                .spawn(move |_| {
                    executor_main(
                        net_receiver,
                        tensor_receiver,
                        batch_size,
                        Some(executor_collector),
                        Some(executor_ready),
                        executor_id,
                    );
                })
                .expect("Failed to spawn executor thread");
        }

        let _collector_handle = scope
            .builder()
            .name("collector-lichess".to_string())
            .spawn(move |_| collector_main(collector_recv, collect_stream, id_recv))
            .expect("Failed to spawn collector thread");

        let _game_handle = scope
            .builder()
            .name("game-loop-lichess".to_string())
            .spawn(move |_| {
                println!("Starting game loop thread.");

                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create game-loop Tokio runtime");

                runtime.block_on(async move {
                    if let Err(error) =
                        game_loop(
                            token,
                            tensor_send,
                            collector_send.clone(),
                            executor_ready_recv,
                            num_executors,
                            num_generators,
                        )
                        .await
                    {
                        eprintln!("Game loop error: {error}");
                    }
                });

                println!("Game loop thread stopped.");
            })
            .expect("Failed to spawn game loop thread");

        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    })
    .expect("Thread scope failed")?;

    Ok(())
}

fn commander_main(
    mut stream: std::net::TcpStream,
    net_senders: Vec<Sender<String>>,
    id_send: Sender<usize>,
) {
    use std::{
        fs::File,
        io::{BufRead, BufReader, Write},
        time::{SystemTime, UNIX_EPOCH},
    };
    let mut reader = BufReader::new(stream.try_clone().expect("clone server stream"));
    let mut has_network = false;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            return;
        }
        let Ok(message) = serde_json::from_str::<MessageServer>(&line) else {
            continue;
        };
        match message.purpose {
            MessageType::IdentityConfirmation((Entity::RustDataGen, id)) => {
                let _ = id_send.send(id);
            }
            MessageType::NewNetworkData(data) => {
                let path = data_path_str(&format!(
                    "nets/lichess_net_{}_{}.pt",
                    std::process::id(),
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ));
                let mut file = File::create(&path).expect("create received network");
                file.write_all(&data).expect("write received network");
                for net_sender in &net_senders {
                    net_sender
                        .send(path.clone())
                        .expect("executor network channel disconnected");
                }
                has_network = true;
            }

            _ => {}
        }
        if !has_network {
            let request = MessageServer {
                purpose: MessageType::RequestingNet(),
            };
            let _ = writeln!(stream, "{}", serde_json::to_string(&request).unwrap());
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn collector_main(
    receiver: Receiver<CollectorMessage>,
    mut stream: std::net::TcpStream,
    id_recv: Receiver<usize>,
) {
    use std::{
        fs,
        io::{Read, Write},
        time::{SystemTime, UNIX_EPOCH},
    };
    let id = id_recv.recv().unwrap_or(0);
    let root = data_path("games");
    let _ = fs::create_dir_all(&root);
    let mut path = data_path_str(&format!(
        "games/lichess_gen_{}_{}",
        id,
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut output = BinaryOutput::new(&path, "chess").expect("create lichess output");
    let mut nps_start_time = Instant::now();
    let mut nps_total = 0usize;
    let mut evals_start_time = Instant::now();
    let mut evals_total = 0usize;

    let send_statistics = |stream: &mut std::net::TcpStream, statistics: Statistics| {
        let message = MessageServer {
            purpose: MessageType::StatisticsSend(statistics),
        };
        let wire = format!("{}\n", serde_json::to_string(&message).unwrap());
        stream.write_all(wire.as_bytes())?;
        stream.flush()?;
        Ok::<(), std::io::Error>(())
    };

    loop {
        match receiver.recv() {
            Ok(CollectorMessage::FinishedGame(sim)) => {
                output.append(&sim).expect("append lichess game");
                if output.game_count() >= 100 {
                    output.finish().expect("finish lichess output");
                    let files = [".bin", ".off", ".json"];
                    let mut data = Vec::new();
                    for (index, suffix) in files.into_iter().enumerate() {
                        let mut bytes = Vec::new();
                        fs::File::open(format!("{}{}", path, suffix))
                            .unwrap()
                            .read_to_end(&mut bytes)
                            .unwrap();
                        data.push(match index {
                            0 => DataFileType::BinFile(bytes),
                            1 => DataFileType::OffFile(bytes),
                            _ => DataFileType::MetaDataFile(bytes),
                        });
                    }
                    let message = MessageServer {
                        purpose: MessageType::JobSendData(data),
                    };
                    writeln!(stream, "{}", serde_json::to_string(&message).unwrap())
                        .expect("failed to send generated lichess games");
                    stream
                        .flush()
                        .expect("failed to flush generated lichess games");
                    for suffix in files {
                        let file_path = format!("{}{}", path, suffix);
                        fs::remove_file(&file_path)
                            .expect("failed to delete sent lichess game data file");
                    }
                    path = data_path_str(&format!(
                        "games/lichess_gen_{}_{}",
                        id,
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_nanos()
                    ));
                    output = BinaryOutput::new(&path, "chess").expect("reopen lichess output");
                }
            }
            Ok(CollectorMessage::GeneratorStatistics(nps)) => {
                nps_total = nps_total.saturating_add(nps);
                let elapsed = nps_start_time.elapsed();
                if elapsed >= Duration::from_secs(1) {
                    let nps = (nps_total as f64 / elapsed.as_secs_f64()) as usize;
                    send_statistics(&mut stream, Statistics::NodesPerSecond(nps))
                        .expect("send self-play statistics");
                    nps_start_time = Instant::now();
                    nps_total = 0;
                }
            }
            Ok(CollectorMessage::ExecutorStatistics(evals)) => {
                evals_total = evals_total.saturating_add(evals);
                let elapsed = evals_start_time.elapsed();
                if elapsed >= Duration::from_secs(1) {
                    let evals_per_second =
                        (evals_total as f64 / elapsed.as_secs_f64()) as usize;
                    send_statistics(&mut stream, Statistics::EvalsPerSecond(evals_per_second))
                        .expect("send executor statistics");
                    evals_start_time = Instant::now();
                    evals_total = 0;
                }
            }
            Ok(_) => {}
            Err(_) => return,
        }
    }
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
