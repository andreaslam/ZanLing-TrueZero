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
    model::{LichessColor, LichessTitle},
    LichessClient,
};

use lru::LruCache;
use shakmaty::{
    fen::Fen,
    uci::UciMove,
    CastlingMode,
    Chess,
    Color,
    EnPassantMode,
    Position,
};

use tzrust::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    dataformat::ZeroEvaluationAbs,
    executor::{executor_static, Message, Packet},
    mcts::get_move,
    mcts_trainer::{EvalMode, TypeRequest::TrainerSearch},
    settings::{
        CPUCTSettings,
        FPUSettings,
        MovesLeftSettings,
        PSTSettings,
        SearchSettings,
    },
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
    time::{SystemTime, UNIX_EPOCH, Instant},
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

const ONLINE_HUMAN_PAGE_URL: &str =
    "https://lichess.org/player";

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
const PLAYER_GRAPH_DB_PATH: &str =
    "lichess_player_graph.sqlite3";

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
const GRAPH_REEXPAND_INTERVAL_SECS: u64 = 1_800;

/*
 * Outgoing challenge time control.
 *
 * 3+2 rated.
 */
const BOT_CHALLENGE_INITIAL_SECONDS: u32 = 180;
const BOT_CHALLENGE_INCREMENT_SECONDS: u32 = 2;
const BOT_CHALLENGE_RATED: bool = true;

/*
 * Humans use the same challenge settings as bots.
 */
const HUMAN_CHALLENGE_INITIAL_SECONDS: u32 =
    BOT_CHALLENGE_INITIAL_SECONDS;

const HUMAN_CHALLENGE_INCREMENT_SECONDS: u32 =
    BOT_CHALLENGE_INCREMENT_SECONDS;

const HUMAN_CHALLENGE_RATED: bool =
    BOT_CHALLENGE_RATED;

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

    let token =
        env::var("LICHESS_API_KEY")
            .expect("LICHESS_API_KEY must be set");

    let net_path =
        String::from(
            r"C:\Users\andre\RemoteFolder\ZanLing-TrueZero\tz_163.pt",
        );

    let (tensor_exe_send, tensor_exe_recv) =
        flume::unbounded::<Packet>();

    let (ctrl_send, ctrl_recv) =
        flume::unbounded::<Message>();

    let game_tensor_send =
        tensor_exe_send.clone();

    thread::scope(|scope| {
        let executor_handle = scope
            .builder()
            .name("executor-lichess".to_string())
            .spawn(move |_| {
                println!("Starting executor thread.");

                executor_static(
                    net_path,
                    tensor_exe_recv,
                    ctrl_recv,
                    1,
                );

                println!("Executor thread stopped.");
            })
            .expect("Failed to spawn executor thread");

        let game_handle = scope
            .builder()
            .name("game-loop-lichess".to_string())
            .spawn(move |_| {
                println!("Starting game loop thread.");

                let runtime =
                    tokio::runtime::Builder::new_multi_thread()
                        .enable_all()
                        .build()
                        .expect(
                            "Failed to create game-loop Tokio runtime",
                        );

                runtime.block_on(async move {
                    if let Err(error) =
                        game_loop(
                            token,
                            game_tensor_send,
                        )
                        .await
                    {
                        eprintln!(
                            "Game loop error: {error}"
                        );
                    }
                });

                println!("Game loop thread stopped.");
            })
            .expect("Failed to spawn game loop thread");

        executor_handle
            .join()
            .expect("Executor thread panicked");

        game_handle
            .join()
            .expect("Game thread panicked");

        Ok::<
            (),
            Box<dyn std::error::Error + Send + Sync>,
        >(())
    })
    .expect("Thread scope failed")?;

    let _ = ctrl_send;

    Ok(())
}

async fn game_loop(
    token: String,
    tensor_exe_send: Sender<Packet>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client =
        Arc::new(
            LichessClient::builder()
                .token(&token)
                .build()
                .expect(
                    "Failed to build Lichess client",
                ),
        );

    let me =
        client
            .account()
            .profile()
            .await
            .expect(
                "Failed to fetch Lichess profile",
            );

    println!(
        "Logged in as {}",
        me.user.username
    );

    if let Some(perfs) =
        me.user.perfs.as_ref()
    {
        if let Some(blitz) =
            perfs.blitz.as_ref()
        {
            println!(
                "Current Blitz rating: {}",
                blitz.rating
            );
        } else {
            println!(
                "Current Blitz rating unavailable: \
                 account has no Blitz rating."
            );
        }
    } else {
        println!(
            "Current Blitz rating unavailable: \
             account has no performance data."
        );
    }

    let api_request_lock =
        Arc::new(Mutex::new(()));

    let active_games =
        Arc::new(AtomicUsize::new(0));

    let event_client =
        Arc::clone(&client);

    let event_tensor_send =
        tensor_exe_send.clone();

    let event_api_lock =
        Arc::clone(&api_request_lock);

    let event_active_games =
        Arc::clone(&active_games);

    let event_task =
        tokio::spawn(async move {
            event_loop(
                event_client,
                event_tensor_send,
                event_api_lock,
                event_active_games,
            )
            .await
        });

    /*
     * Matchmaking has its own task and therefore can never block
     * consumption of the Lichess event stream.
     */
    let matchmaking_client =
        Arc::clone(&client);

    let matchmaking_api_lock =
        Arc::clone(&api_request_lock);

    let username =
        me.user.username.clone();

    let matchmaking_task =
        tokio::spawn(async move {
            matchmaking_loop(
                matchmaking_client,
                username,
                matchmaking_api_lock,
            )
            .await
        });

    tokio::select! {
        result = event_task => {
            match result {
                Ok(Ok(())) => {
                    Err(
                        "Lichess event task terminated unexpectedly"
                            .into()
                    )
                }

                Ok(Err(error)) => {
                    Err(error)
                }

                Err(error) => {
                    Err(
                        format!(
                            "Lichess event task panicked: {}",
                            error
                        )
                        .into()
                    )
                }
            }
        }

        result = matchmaking_task => {
            match result {
                Ok(Ok(())) => {
                    Err(
                        "Matchmaking task terminated unexpectedly"
                            .into()
                    )
                }

                Ok(Err(error)) => {
                    Err(error)
                }

                Err(error) => {
                    Err(
                        format!(
                            "Matchmaking task panicked: {}",
                            error
                        )
                        .into()
                    )
                }
            }
        }
    }
}

async fn event_loop(
    client: Arc<LichessClient>,
    tensor_exe_send: Sender<Packet>,
    api_request_lock: Arc<Mutex<()>>,
    active_games: Arc<AtomicUsize>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    loop {
        println!(
            "Opening Lichess bot event stream..."
        );

        let bot_api =
            client.bot();

        let mut events =
            match bot_api.stream_events().await {
                Ok(stream) => {
                    println!(
                        "Lichess bot event stream connected."
                    );

                    stream
                }

                Err(error) => {
                    eprintln!(
                        "Failed to open Lichess bot \
                         event stream: {}",
                        error
                    );

                    sleep(
                        Duration::from_millis(
                            EVENT_STREAM_RECONNECT_DELAY_MS,
                        ),
                    )
                    .await;

                    continue;
                }
            };

        while let Some(event_result) =
            events.next().await
        {
            match event_result {
                Ok(
                    LichessIncomingEvent::Challenge {
                        challenge,
                    },
                ) => {
                    println!(
                        "Incoming challenge event: {:?}",
                        challenge
                    );

                    if !matches!(
                        challenge.status,
                        LichessChallengeStatus::Created
                    ) {
                        println!(
                            "Ignoring challenge {} with status {:?}",
                            challenge.id,
                            challenge.status
                        );

                        continue;
                    }

                    println!(
                        "Accepting challenge {}: {} -> {} (direction={:?})",
                        challenge.id,
                        challenge
                            .challenger
                            .as_ref()
                            .map(|user| user.name.as_str())
                            .unwrap_or("<unknown>"),
                        challenge
                            .dest_user
                            .as_ref()
                            .map(|user| user.name.as_str())
                            .unwrap_or("<unknown>"),
                        challenge.direction
                    );

                    let _api_guard =
                        api_request_lock.lock().await;

                    println!(
                        "Accepting incoming challenge {}.",
                        challenge.id
                    );

                    match client
                        .challenges()
                        .accept(&challenge.id, None)
                        .await
                    {
                        Ok(_) => {
                            println!(
                                "Accepted challenge {}",
                                challenge.id
                            );
                        }

                        Err(error) => {
                            eprintln!(
                                "Failed to accept challenge {}: {}",
                                challenge.id,
                                error
                            );
                        }
                    }
                }

                Ok(
                    LichessIncomingEvent::GameStart {
                        game,
                    },
                ) => {
                    let Some(game_id) = game.id else {
                        println!(
                            "GameStart without game ID"
                        );

                        continue;
                    };

                    let Some(our_color) =
                        game.color
                    else {
                        println!(
                            "GameStart {} without \
                             our color",
                            game_id
                        );

                        continue;
                    };

                    println!(
                        "Game started: {} ({:?})",
                        game_id,
                        our_color
                    );

                    let game_client =
                        Arc::clone(&client);

                    let game_tensor_send =
                        tensor_exe_send.clone();

                    let game_active_games =
                        Arc::clone(&active_games);

                    let game_api_lock =
                        Arc::clone(&api_request_lock);

                    tokio::spawn(async move {
                        if let Err(error) =
                            run_game_owned(
                                game_client,
                                game_id.clone(),
                                our_color,
                                game_tensor_send,
                                game_active_games,
                                game_api_lock,
                            )
                            .await
                        {
                            eprintln!(
                                "Game {} error: {}",
                                game_id,
                                error
                            );
                        }

                        println!(
                            "Game {} finished.",
                            game_id
                        );
                    });
                }

                Ok(_other) => {
                    println!(
                        "Ignoring unrelated bot event."
                    );
                }

                Err(error) => {
                    eprintln!(
                        "Lichess bot event stream error: {}",
                        error
                    );

                    break;
                }
            }
        }

        println!(
            "Global Lichess event stream ended; \
             reconnecting in {} ms...",
            EVENT_STREAM_RECONNECT_DELAY_MS
        );

        sleep(
            Duration::from_millis(
                EVENT_STREAM_RECONNECT_DELAY_MS,
            ),
        )
        .await;
    }
}

async fn matchmaking_loop(
    client: Arc<LichessClient>,
    our_username: String,
    api_request_lock: Arc<Mutex<()>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    /*
     * Open the graph database once for the lifetime of the matchmaking
     * task.
     */
    let graph_db =
        Arc::new(
            Mutex::new(
                open_player_graph_database(
                    PLAYER_GRAPH_DB_PATH,
                )?,
            ),
        );

    let mut bot_challenge_cooldowns:
        HashMap<String, Instant> =
        HashMap::new();

    let mut bot_pending_challenges:
        HashSet<String> =
        HashSet::new();

    let mut human_pending_challenges:
        HashSet<String> =
        HashSet::new();

    let mut scan_interval =
        tokio::time::interval(
            Duration::from_secs(
                ONLINE_BOT_SCAN_INTERVAL_SECS,
            ),
        );

    loop {
        scan_interval.tick().await;

        /*
         * EXISTING BOT MATCHMAKING.
         */
        if let Err(error) =
            challenge_online_bots(
                &client,
                &our_username,
                &mut bot_challenge_cooldowns,
                &mut bot_pending_challenges,
                &api_request_lock,
            )
            .await
        {
            eprintln!(
                "Online bot matchmaking error: {}",
                error
            );
        }

        /*
         * NEW GRAPH DISCOVERY SOURCE.
         *
         * This happens independently of the actual human challenge scan.
         */
        if let Err(error) =
            expand_player_graph(
                &client,
                &our_username,
                &graph_db,
            )
            .await
        {
            eprintln!(
                "Player graph expansion error: {}",
                error
            );
        }

        /*
         * EXISTING DIRECT ONLINE-HUMAN SOURCE.
         *
         * This remains unchanged conceptually and is still useful because
         * it discovers currently-online users independently of the graph.
         */
        if let Err(error) =
            challenge_online_humans(
                &client,
                &our_username,
                &mut human_pending_challenges,
                &api_request_lock,
                &graph_db,
            )
            .await
        {
            eprintln!(
                "Online human matchmaking error: {}",
                error
            );
        }
    }
}

/*
 * EXISTING BOT MATCHMAKING.
 */
async fn challenge_online_bots(
    client: &LichessClient,
    our_username: &str,
    cooldowns: &mut HashMap<String, Instant>,
    pending_challenges: &mut HashSet<String>,
    api_request_lock: &Arc<Mutex<()>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!(
        "Scanning currently-online Lichess bots..."
    );

    let bot_api =
        client.bot();

    let mut online_bots =
        bot_api.online(None).await?;

    let now =
        Instant::now();

    cooldowns.retain(
        |_, last_attempt| {
            now.duration_since(*last_attempt)
                .as_secs()
                < ONLINE_BOT_CHALLENGE_COOLDOWN_SECS
        },
    );

    pending_challenges.retain(
        |username| {
            cooldowns.contains_key(username)
        },
    );

    let mut online_count =
        0usize;

    let mut challenge_count =
        0usize;

    while let Some(bot_result) =
        online_bots.next().await
    {
        let bot =
            match bot_result {
                Ok(bot) => bot,

                Err(error) => {
                    eprintln!(
                        "Error while reading online \
                         bot list: {}",
                        error
                    );

                    continue;
                }
            };

        online_count += 1;

        if bot.username
            .eq_ignore_ascii_case(our_username)
        {
            println!(
                "Skipping ourselves: {}",
                bot.username
            );

            continue;
        }

        if challenge_count >=
            MAX_BOT_CHALLENGES_PER_SCAN
        {
            println!(
                "Reached bot matchmaking limit of {} \
                 challenge attempts this scan.",
                MAX_BOT_CHALLENGES_PER_SCAN
            );

            break;
        }

        if pending_challenges
            .contains(&bot.username)
        {
            println!(
                "Skipping {}: outgoing challenge \
                 already pending.",
                bot.username
            );

            continue;
        }

        if let Some(last_attempt) =
            cooldowns.get(&bot.username)
        {
            let age =
                now.duration_since(*last_attempt)
                    .as_secs();

            if age <
                ONLINE_BOT_CHALLENGE_COOLDOWN_SECS
            {
                println!(
                    "Skipping {}: challenge cooldown \
                     active ({}s old).",
                    bot.username,
                    age
                );

                continue;
            }
        }

        cooldowns.insert(
            bot.username.clone(),
            now,
        );

        pending_challenges.insert(
            bot.username.clone(),
        );

        challenge_count += 1;

        let challenge_color =
            challenge_color_for_bot(
                &bot.username,
            );

        println!(
            "Challenging online bot {} as {:?} \
             with {}+{} rated ({}/{})",
            bot.username,
            challenge_color,
            BOT_CHALLENGE_INITIAL_SECONDS / 60,
            BOT_CHALLENGE_INCREMENT_SECONDS,
            challenge_count,
            MAX_BOT_CHALLENGES_PER_SCAN
        );

        let result = {
            let _api_guard =
                api_request_lock.lock().await;

            client
                .challenges()
                .challenge(&bot.username)
                .color(challenge_color)
                .rated(BOT_CHALLENGE_RATED)
                .clock(
                    BOT_CHALLENGE_INITIAL_SECONDS,
                    BOT_CHALLENGE_INCREMENT_SECONDS,
                )
                .send()
                .await
        };

        match result {
            Ok(challenge) => {
                println!(
                    "Challenge sent to {}: \
                     id={}, status={:?}",
                    bot.username,
                    challenge.id,
                    challenge.status
                );

                if !matches!(
                    challenge.status,
                    LichessChallengeStatus::Created
                ) {
                    pending_challenges
                        .remove(&bot.username);
                }
            }

            Err(error) => {
                eprintln!(
                    "Failed to challenge {}: {}",
                    bot.username,
                    error
                );

                pending_challenges
                    .remove(&bot.username);
            }
        }

        if challenge_count <
            MAX_BOT_CHALLENGES_PER_SCAN
        {
            sleep(
                Duration::from_millis(
                    BOT_CHALLENGE_SEND_DELAY_MS,
                ),
            )
            .await;
        }
    }

    println!(
        "Online bot scan complete: {} bots seen, \
         {} challenge attempts made, \
         {} outgoing challenges currently pending.",
        online_count,
        challenge_count,
        pending_challenges.len()
    );

    Ok(())
}

/*
 * ============================================================================
 * PLAYER GRAPH
 * ============================================================================
 *
 * This is the additional human-discovery source.
 *
 * It deliberately does NOT replace the existing /player implementation.
 */

/*
 * Open/create the persistent graph database.
 */
fn open_player_graph_database(
    path: &str,
) -> Result<
    Connection,
    Box<dyn std::error::Error + Send + Sync>,
> {
    let connection =
        Connection::open(path)?;

    connection.execute_batch("PRAGMA foreign_keys = ON;")?;

    let has_legacy_players =
        connection
            .query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM sqlite_master
                    WHERE type = 'table' AND name = 'players'
                )",
                [],
                |row| row.get::<_, bool>(0),
            )?
            && !connection
                .prepare("PRAGMA table_info(players)")?
                .query_map([], |row| row.get::<_, String>(1))?
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .any(|column| column == "id");

    if has_legacy_players {
        connection.execute_batch("BEGIN IMMEDIATE;")?;
        connection.execute_batch(
            r#"
            ALTER TABLE players RENAME TO players_legacy;
            ALTER TABLE player_edges RENAME TO player_edges_legacy;
            ALTER TABLE human_challenges RENAME TO human_challenges_legacy;
            "#,
        )?;
    }

    connection.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL COLLATE NOCASE UNIQUE,
            is_bot INTEGER CHECK (is_bot IS NULL OR is_bot IN (0, 1)),
            first_seen_at INTEGER NOT NULL,
            last_seen_online_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS discovery_sources (
            id INTEGER PRIMARY KEY,
            source_key TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS player_discoveries (
            player_id INTEGER NOT NULL REFERENCES players(id),
            source_id INTEGER NOT NULL REFERENCES discovery_sources(id),
            first_seen_at INTEGER NOT NULL,
            last_seen_at INTEGER NOT NULL,
            PRIMARY KEY (player_id, source_id)
        );

        CREATE TABLE IF NOT EXISTS player_expansions (
            player_id INTEGER PRIMARY KEY REFERENCES players(id),
            last_expanded_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS player_edges (
            player_id INTEGER NOT NULL REFERENCES players(id),
            opponent_id INTEGER NOT NULL REFERENCES players(id),
            first_seen_at INTEGER NOT NULL,
            last_seen_at INTEGER NOT NULL,
            PRIMARY KEY (player_id, opponent_id),
            CHECK (player_id <> opponent_id)
        );

        CREATE TABLE IF NOT EXISTS human_challenges (
            player_id INTEGER PRIMARY KEY REFERENCES players(id),
            last_challenged_at INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_player_expansions_due
            ON player_expansions(last_expanded_at);

        CREATE INDEX IF NOT EXISTS idx_players_bot
            ON players(is_bot);

        CREATE INDEX IF NOT EXISTS idx_player_edges_opponent
            ON player_edges(opponent_id);

        CREATE INDEX IF NOT EXISTS idx_player_discoveries_source
            ON player_discoveries(source_id);
        "#,
    )?;

    if has_legacy_players {
        connection.execute_batch(
            r#"
            INSERT INTO players (
                username,
                is_bot,
                first_seen_at,
                last_seen_online_at
            )
            SELECT username, is_bot, first_seen_at, last_seen_online_at
            FROM players_legacy;

            INSERT INTO discovery_sources (source_key)
            SELECT DISTINCT discovered_from
            FROM players_legacy
            WHERE discovered_from IS NOT NULL;

            INSERT INTO player_discoveries (
                player_id,
                source_id,
                first_seen_at,
                last_seen_at
            )
            SELECT p.id, s.id, p.first_seen_at, p.first_seen_at
            FROM players_legacy old
            JOIN players p ON p.username = old.username
            JOIN discovery_sources s
                ON s.source_key = old.discovered_from
            WHERE old.discovered_from IS NOT NULL;

            INSERT INTO player_expansions (player_id, last_expanded_at)
            SELECT p.id, old.last_expanded_at
            FROM players_legacy old
            JOIN players p ON p.username = old.username
            WHERE old.last_expanded_at IS NOT NULL;

            INSERT INTO player_edges (
                player_id,
                opponent_id,
                first_seen_at,
                last_seen_at
            )
            SELECT source.id, opponent.id, old.last_seen_at, old.last_seen_at
            FROM player_edges_legacy old
            JOIN players source
                ON source.username = old.username
            JOIN players opponent
                ON opponent.username = old.opponent
            WHERE source.id <> opponent.id;

            INSERT INTO human_challenges (player_id, last_challenged_at)
            SELECT p.id, old.last_challenged_at
            FROM human_challenges_legacy old
            JOIN players p ON p.username = old.username;

            DROP TABLE players_legacy;
            DROP TABLE player_edges_legacy;
            DROP TABLE human_challenges_legacy;
            "#,
        )?;
        connection.execute_batch("COMMIT;")?;
    }

    Ok(connection)
}

fn unix_time_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/*
 * Insert a discovered player if we have not seen them before.
 *
 * `is_bot` is deliberately nullable because discovering somebody from a game
 * does not itself tell us whether the account is a bot.
 */
fn graph_insert_player(
    db: &Connection,
    username: &str,
    is_bot: Option<bool>,
    source: &str,
) -> Result<(), rusqlite::Error> {
    let now =
        unix_time_now();

    db.execute(
        r#"
        INSERT INTO players (
            username,
            is_bot,
            first_seen_at
        )
        VALUES (?1, ?2, ?3)
        ON CONFLICT(username) DO UPDATE SET
            is_bot = COALESCE(
                excluded.is_bot,
                players.is_bot
            )
        "#,
        params![
            username,
            is_bot.map(|value| {
                if value { 1 } else { 0 }
            }),
            now,
        ],
    )?;

    db.execute(
        "INSERT INTO discovery_sources (source_key)
         VALUES (?1)
         ON CONFLICT(source_key) DO NOTHING",
        params![source],
    )?;

    db.execute(
        r#"
        INSERT INTO player_discoveries (
            player_id,
            source_id,
            first_seen_at,
            last_seen_at
        )
        SELECT p.id, s.id, ?1, ?1
        FROM players p
        CROSS JOIN discovery_sources s
        WHERE p.username = ?2 AND s.source_key = ?3
        ON CONFLICT(player_id, source_id) DO UPDATE SET
            last_seen_at = excluded.last_seen_at
        "#,
        params![now, username, source],
    )?;

    Ok(())
}

fn graph_mark_online(
    db: &Connection,
    username: &str,
) -> Result<(), rusqlite::Error> {
    let now =
        unix_time_now();

    db.execute(
        r#"
        UPDATE players
        SET last_seen_online_at = ?1
        WHERE username = ?2
        "#,
        params![
            now,
            username,
        ],
    )?;

    Ok(())
}

fn graph_mark_expanded(
    db: &Connection,
    username: &str,
) -> Result<(), rusqlite::Error> {
    let now =
        unix_time_now();

    db.execute(
        r#"
        INSERT INTO player_expansions (player_id, last_expanded_at)
        SELECT id, ?1
        FROM players
        WHERE username = ?2
        ON CONFLICT(player_id) DO UPDATE SET
            last_expanded_at = excluded.last_expanded_at
        "#,
        params![
            now,
            username,
        ],
    )?;

    Ok(())
}

fn graph_insert_edge(
    db: &Connection,
    username: &str,
    opponent: &str,
) -> Result<(), rusqlite::Error> {
    let now =
        unix_time_now();

    db.execute(
        r#"
        INSERT INTO player_edges (
            player_id,
            opponent_id,
            first_seen_at,
            last_seen_at
        )
        SELECT source.id, opponent.id, ?3, ?3
        FROM players source
        CROSS JOIN players opponent
        WHERE source.username = ?1
          AND opponent.username = ?2
          AND source.id <> opponent.id
        ON CONFLICT(player_id, opponent_id) DO UPDATE SET
            last_seen_at = excluded.last_seen_at
        "#,
        params![
            username,
            opponent,
            now,
        ],
    )?;

    Ok(())
}

fn graph_get_expansion_candidates(
    db: &Connection,
    limit: usize,
) -> Result<
    Vec<String>,
    rusqlite::Error,
> {
    let cutoff =
        unix_time_now()
            - GRAPH_REEXPAND_INTERVAL_SECS as i64;

    let mut statement =
        db.prepare(
            r#"
            SELECT username
            FROM players
            LEFT JOIN player_expansions
                ON player_expansions.player_id = players.id
            WHERE player_expansions.last_expanded_at IS NULL
               OR player_expansions.last_expanded_at < ?1
            ORDER BY
                CASE
                    WHEN player_expansions.last_expanded_at IS NULL
                    THEN 0
                    ELSE 1
                END,
                player_expansions.last_expanded_at ASC
            LIMIT ?2
            "#,
        )?;

    let rows =
        statement.query_map(
            params![
                cutoff,
                limit as i64,
            ],
            |row| {
                row.get::<_, String>(0)
            },
        )?;

    let mut result =
        Vec::new();

    for row in rows {
        result.push(row?);
    }

    Ok(result)
}

/*
 * Fetch recent games for one player and extract their opponents.
 *
 * Lichess's user-game endpoint returns NDJSON when requested with the
 * application/x-ndjson Accept header.
 */
async fn fetch_player_opponents(
    username: &str,
) -> Result<
    Vec<String>,
    Box<dyn std::error::Error + Send + Sync>,
> {
    let url =
        format!(
            "https://lichess.org/api/games/user/{}?max={}&moves=false&clocks=false&evals=false&opening=false&literate=false&ongoing=false&finished=true",
            username,
            GRAPH_GAMES_PER_PLAYER,
        );

    let response =
        reqwest::Client::new()
            .get(&url)
            .header(
                reqwest::header::USER_AGENT,
                "ZanLing-TrueZero Lichess player discovery",
            )
            .header(
                reqwest::header::ACCEPT,
                "application/x-ndjson",
            )
            .send()
            .await?
            .error_for_status()?;

    let body =
        response.text().await?;

    let mut opponents =
        Vec::new();

    let mut seen =
        HashSet::new();

    for line in body.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let game:
            serde_json::Value =
            match serde_json::from_str(line) {
                Ok(value) => value,

                Err(error) => {
                    eprintln!(
                        "Failed to parse game JSON for {}: {}",
                        username,
                        error
                    );

                    continue;
                }
            };

        let white =
            game
                .get("players")
                .and_then(|players| {
                    players.get("white")
                })
                .and_then(|white| {
                    white.get("user")
                })
                .and_then(|user| {
                    user.get("name")
                })
                .and_then(
                    |name| name.as_str(),
                );

        let black =
            game
                .get("players")
                .and_then(|players| {
                    players.get("black")
                })
                .and_then(|black| {
                    black.get("user")
                })
                .and_then(|user| {
                    user.get("name")
                })
                .and_then(
                    |name| name.as_str(),
                );

        let opponent =
            match (
                white,
                black,
            ) {
                (Some(white), Some(black))
                    if white.eq_ignore_ascii_case(
                        username,
                    ) =>
                {
                    Some(black)
                }

                (Some(white), Some(black))
                    if black.eq_ignore_ascii_case(
                        username,
                    ) =>
                {
                    Some(white)
                }

                /*
                 * This also handles cases where the API gives a
                 * username with different casing.
                 */
                (Some(white), Some(black)) => {
                    if !white.eq_ignore_ascii_case(
                        username,
                    ) {
                        Some(white)
                    } else if !black.eq_ignore_ascii_case(
                        username,
                    ) {
                        Some(black)
                    } else {
                        None
                    }
                }

                _ => None,
            };

        let Some(opponent) =
            opponent
        else {
            continue;
        };

        if opponent.is_empty()
            || opponent.eq_ignore_ascii_case(
                username,
            )
        {
            continue;
        }

        if seen.insert(
            opponent.to_ascii_lowercase(),
        ) {
            opponents.push(
                opponent.to_string(),
            );
        }
    }

    Ok(opponents)
}

/*
 * Expand the graph.
 *
 * Seed source:
 *
 *     currently-online bots
 *
 * Then:
 *
 *     bot -> recent opponents
 *     opponent -> their recent opponents
 *     opponent -> their recent opponents
 *     ...
 *
 * The database makes this persistent across restarts.
 */
async fn expand_player_graph(
    client: &LichessClient,
    our_username: &str,
    graph_db: &Arc<Mutex<Connection>>,
) -> Result<
    (),
    Box<dyn std::error::Error + Send + Sync>,
> {
    println!(
        "Expanding persistent Lichess player graph..."
    );

    /*
     * Seed the graph with both online bots and online humans. Human users
     * discovered here are expanded through the same recent-game API, so a
     * human found in /player can discover additional players on later scans.
     */
    let online_humans =
        fetch_online_player_usernames().await?;

    let bot_api =
        client.bot();

    let mut online_bots =
        bot_api.online(None).await?;

    let mut human_seed_count =
        0usize;

    let mut bot_seed_count =
        0usize;

    {
        let db =
            graph_db.lock().await;

        for username in online_humans {
            if username.eq_ignore_ascii_case(our_username) {
                continue;
            }

            graph_insert_player(
                &db,
                &username,
                None,
                "online_human",
            )?;

            human_seed_count += 1;
        }

        while let Some(bot_result) =
            online_bots.next().await
        {
            let bot =
                match bot_result {
                    Ok(bot) => bot,

                    Err(error) => {
                        eprintln!(
                            "Graph bot-list error: {}",
                            error
                        );

                        continue;
                    }
                };

            if bot.username
                .eq_ignore_ascii_case(
                    our_username,
                )
            {
                continue;
            }

            graph_insert_player(
                &db,
                &bot.username,
                Some(true),
                "online_bot",
            )?;

            bot_seed_count += 1;
        }
    }

    println!(
        "Graph: seeded {} currently-online humans and {} bots.",
        human_seed_count,
        bot_seed_count,
    );

    /*
     * Now take a bounded number of graph nodes and expand them.
     *
     * We intentionally do this sequentially. This makes the graph
     * traversal predictable and avoids hammering the public game API.
     */
    let expansion_candidates =
        {
            let db =
                graph_db.lock().await;

            graph_get_expansion_candidates(
                &db,
                MAX_GRAPH_PLAYERS_PER_SCAN,
            )?
        };

    println!(
        "Graph: {} players selected for expansion.",
        expansion_candidates.len()
    );

    let mut expanded =
        0usize;

    let mut discovered =
        0usize;

    for username in
        expansion_candidates
    {
        println!(
            "Graph: expanding {}...",
            username
        );

        let opponents =
            match fetch_player_opponents(
                &username,
            )
            .await
            {
                Ok(opponents) =>
                    opponents,

                Err(error) => {
                    eprintln!(
                        "Graph: failed to fetch games for {}: {}",
                        username,
                        error
                    );

                    continue;
                }
            };

        {
            let db =
                graph_db.lock().await;

            for opponent in
                &opponents
            {
                if opponent.eq_ignore_ascii_case(
                    our_username,
                ) {
                    continue;
                }

                let existed:
                    bool =
                    db.query_row(
                        r#"
                        SELECT EXISTS(
                            SELECT 1
                            FROM players
                            WHERE username = ?1
                        )
                        "#,
                        params![
                            opponent
                        ],
                        |row| {
                            row.get(0)
                        },
                    )?;

                graph_insert_player(
                    &db,
                    opponent,
                    None,
                    &format!(
                        "game: {}",
                        username
                    ),
                )?;

                graph_insert_edge(
                    &db,
                    &username,
                    opponent,
                )?;

                if !existed {
                    discovered += 1;
                }
            }

            graph_mark_expanded(
                &db,
                &username,
            )?;
        }

        expanded += 1;

        println!(
            "Graph: {} produced {} unique recent opponents.",
            username,
            opponents.len()
        );
    }

    let graph_size:
        usize =
        {
            let db =
                graph_db.lock().await;

            db.query_row(
                "SELECT COUNT(*) FROM players",
                [],
                |row| {
                    row.get(0)
                },
            )?
        };

    let edge_count:
        usize =
        {
            let db =
                graph_db.lock().await;

            db.query_row(
                "SELECT COUNT(*) FROM player_edges",
                [],
                |row| {
                    row.get(0)
                },
            )?
        };

    println!(
        "Player graph expansion complete: \
         {} players expanded, \
         {} newly discovered players, \
         {} total players, \
         {} total edges.",
        expanded,
        discovered,
        graph_size,
        edge_count
    );

    Ok(())
}

/*
 * Return persisted graph users who are currently online.
 *
 * This intentionally uses the status endpoint instead of intersecting with
 * /player: /player is only a discovery source and may not contain everybody
 * already known by the graph.
 */
async fn graph_online_candidates(
    client: &LichessClient,
    graph_db: &Arc<Mutex<Connection>>,
) -> Result<
    Vec<String>,
    Box<dyn std::error::Error + Send + Sync>,
> {
    let graph_usernames:
        Vec<String> =
        {
            let db =
                graph_db.lock().await;

            let mut statement =
                db.prepare(
                    r#"
                    SELECT players.username
                    FROM players
                    LEFT JOIN player_expansions
                        ON player_expansions.player_id = players.id
                    WHERE players.is_bot IS NULL OR players.is_bot = 0
                    ORDER BY players.last_seen_online_at DESC,
                             player_expansions.last_expanded_at ASC
                    LIMIT ?1
                    "#,
                )?;

            let rows =
                statement.query_map(
                    params![MAX_ONLINE_HUMAN_CANDIDATES as i64],
                    |row| {
                        row.get::<_, String>(0)
                    },
                )?;

            let mut result =
                Vec::new();

            for row in rows {
                result.push(row?);
            }

            result
        };

    if graph_usernames.is_empty() {
        return Ok(Vec::new());
    }

    let candidate_ids =
        graph_usernames
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();

    let statuses =
        client
            .users()
            .statuses(
                &candidate_ids,
                Some(false),
                Some(false),
                Some(false),
            )
            .await?;

    let mut result =
        Vec::new();

    let db =
        graph_db.lock().await;

    for status in statuses {
        let username =
            status.user.name;

        let is_bot =
            matches!(
                status.user.title,
                Some(LichessTitle::Bot)
            );

        graph_insert_player(
            &db,
            &username,
            Some(is_bot),
            "graph_status",
        )?;

        if status.online == Some(true) {
            graph_mark_online(
                &db,
                &username,
            )?;

            if !is_bot {
                result.push(username);
            }
        }
    }

    Ok(result)
}

/*
 * Persistent human challenge cooldown. This survives process restarts because
 * the last successful challenge time is stored in the matchmaking database.
 */
fn graph_human_challenge_is_on_cooldown(
    db: &Connection,
    username: &str,
) -> Result<Option<i64>, rusqlite::Error> {
    let last_challenged_at =
        db.query_row(
            r#"
            SELECT human_challenges.last_challenged_at
            FROM human_challenges
            JOIN players
                ON players.id = human_challenges.player_id
            WHERE players.username = ?1
            "#,
            params![username],
            |row| row.get::<_, i64>(0),
        )
        .optional()?;

    let Some(last_challenged_at) = last_challenged_at else {
        return Ok(None);
    };

    let cooldown_seconds =
        HUMAN_CHALLENGE_COOLDOWN_DAYS.saturating_mul(24 * 60 * 60);

    let age = unix_time_now().saturating_sub(last_challenged_at);

    if age < cooldown_seconds {
        Ok(Some(cooldown_seconds.saturating_sub(age)))
    } else {
        Ok(None)
    }
}

fn graph_mark_human_challenged(
    db: &Connection,
    username: &str,
) -> Result<(), rusqlite::Error> {
    db.execute(
        r#"
        INSERT INTO human_challenges (
            player_id,
            last_challenged_at
        )
        SELECT id, ?2
        FROM players
        WHERE username = ?1
        ON CONFLICT(player_id) DO UPDATE SET
            last_challenged_at = excluded.last_challenged_at
        "#,
        params![username, unix_time_now()],
    )?;

    Ok(())
}

/*
 * ============================================================================
 * HUMAN MATCHMAKING
 * ============================================================================
 *
 * The original /player discovery remains the first source.
 *
 * The graph database is an additional source.
 */
async fn challenge_online_humans(
    client: &LichessClient,
    our_username: &str,
    pending_challenges: &mut HashSet<String>,
    api_request_lock: &Arc<Mutex<()>>,
    graph_db: &Arc<Mutex<Connection>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!(
        "Scanning currently-online Lichess humans..."
    );

    let our_rating =
        match current_blitz_rating(client).await? {
            Some(rating) => rating,

            None => {
                println!(
                    "Cannot run human matchmaking: \
                     our account has no Blitz rating."
                );

                return Ok(());
            }
        };

    println!(
        "Our current Blitz rating: {}",
        our_rating
    );

    /*
     * ORIGINAL SOURCE:
     *
     * /player
     */
    let mut online_usernames =
        fetch_online_player_usernames().await?;

    /*
     * NEW SOURCE:
     *
     * persistent graph source. Its status is queried live below, so we do
     * not assume a graph user is still online merely because they were
     * online when discovered.
     */
    let graph_candidates =
        graph_online_candidates(
            client,
            graph_db,
        )
        .await?;

        let mut seen:
            HashSet<String> =
            online_usernames
                .iter()
                .map(
                    |username| {
                        username
                            .to_ascii_lowercase()
                    },
                )
                .collect();

        let mut graph_added =
            0usize;

        for username in
            graph_candidates
        {
            if seen.insert(
                username.to_ascii_lowercase(),
            ) {
                online_usernames
                    .push(username);

                graph_added += 1;
            }
        }

    println!(
        "Graph source added {} online \
         candidates to the direct /player source.",
        graph_added
    );

    println!(
        "Combined online-player source yielded {} \
         unique usernames.",
        online_usernames.len()
    );

    if online_usernames.is_empty() {
        println!(
            "No online-player usernames discovered."
        );

        return Ok(());
    }

    let candidate_usernames =
        online_usernames
            .into_iter()
            .filter(
                |username| {
                    !username
                        .eq_ignore_ascii_case(
                            our_username,
                        )
                },
            )
            .take(
                MAX_ONLINE_HUMAN_CANDIDATES,
            )
            .collect::<Vec<_>>();

    if candidate_usernames.is_empty() {
        println!(
            "No online human candidates after \
             excluding ourselves."
        );

        return Ok(());
    }

    let candidate_ids =
        candidate_usernames
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();

    let users =
        client
            .users()
            .get_many(
                &candidate_ids,
                Some(false),
                Some(false),
            )
            .await?;

    let mut eligible_count =
        0usize;

    let mut challenge_count =
        0usize;

    for user in users {
        if challenge_count >=
            MAX_HUMAN_CHALLENGES_PER_SCAN
        {
            println!(
                "Reached human matchmaking limit of {} \
                 challenge attempts this scan.",
                MAX_HUMAN_CHALLENGES_PER_SCAN
            );

            break;
        }

        let username =
            user.username.clone();

        if username
            .eq_ignore_ascii_case(our_username)
        {
            continue;
        }

        /*
         * Keep the graph database's BOT classification current.
         */
        {
            let db =
                graph_db.lock().await;

            let is_bot =
                matches!(
                    user.title,
                    Some(LichessTitle::Bot)
                );

            graph_insert_player(
                &db,
                &username,
                Some(is_bot),
                "online_profile",
            )?;
        }

        if matches!(
            user.title,
            Some(LichessTitle::Bot)
        ) {
            println!(
                "Skipping {}: account is a BOT.",
                username
            );

            continue;
        }

        let candidate_rating =
            match user
                .perfs
                .as_ref()
                .and_then(
                    |perfs| {
                        perfs.blitz
                            .as_ref()
                            .map(
                                |blitz| {
                                    blitz.rating
                                },
                            )
                    },
                )
            {
                Some(rating) => rating,

                None => {
                    println!(
                        "Skipping {}: no Blitz rating.",
                        username
                    );

                    continue;
                }
            };

        let rating_difference =
            candidate_rating as i32
                - our_rating as i32;

        eligible_count += 1;

        if pending_challenges
            .contains(&username)
        {
            println!(
                "Skipping {}: outgoing human challenge \
                 already pending.",
                username
            );

            continue;
        }

        let cooldown_remaining = {
            let db =
                graph_db.lock().await;

            graph_human_challenge_is_on_cooldown(
                &db,
                &username,
            )?
        };

        if let Some(remaining_seconds) = cooldown_remaining {
            println!(
                "Skipping {}: persistent human challenge \
                 cooldown active ({}s remaining; cooldown={} days).",
                username,
                remaining_seconds,
                HUMAN_CHALLENGE_COOLDOWN_DAYS
            );

            continue;
        }

        pending_challenges.insert(
            username.clone(),
        );

        challenge_count += 1;

        let challenge_color =
            challenge_color_for_bot(
                &username,
            );

        println!(
            "Challenging online human {} \
             (Blitz {}, ours {}, difference {:+}) \
             as {:?} with {}+{} rated ({}/{})",
            username,
            candidate_rating,
            our_rating,
            rating_difference,
            challenge_color,
            HUMAN_CHALLENGE_INITIAL_SECONDS / 60,
            HUMAN_CHALLENGE_INCREMENT_SECONDS,
            challenge_count,
            MAX_HUMAN_CHALLENGES_PER_SCAN
        );

        let result = {
            let _api_guard =
                api_request_lock.lock().await;

            client
                .challenges()
                .challenge(&username)
                .color(challenge_color)
                .rated(HUMAN_CHALLENGE_RATED)
                .clock(
                    HUMAN_CHALLENGE_INITIAL_SECONDS,
                    HUMAN_CHALLENGE_INCREMENT_SECONDS,
                )
                .send()
                .await
        };

        match result {
            Ok(challenge) => {
                println!(
                    "Human challenge sent to {}: \
                     id={}, status={:?}",
                    username,
                    challenge.id,
                    challenge.status
                );

                if matches!(
                    challenge.status,
                    LichessChallengeStatus::Created
                ) {
                    let db =
                        graph_db.lock().await;

                    graph_mark_human_challenged(
                        &db,
                        &username,
                    )?;
                } else {
                    pending_challenges
                        .remove(&username);
                }
            }

            Err(error) => {
                eprintln!(
                    "Failed to challenge human {}: {}",
                    username,
                    error
                );

                pending_challenges
                    .remove(&username);
            }
        }

        if challenge_count <
            MAX_HUMAN_CHALLENGES_PER_SCAN
        {
            sleep(
                Duration::from_millis(
                    HUMAN_CHALLENGE_SEND_DELAY_MS,
                ),
            )
            .await;
        }
    }

    println!(
        "Online human scan complete: \
         {} eligible humans found, \
         {} challenge attempts made, \
         {} outgoing human challenges currently pending.",
        eligible_count,
        challenge_count,
        pending_challenges.len()
    );

    Ok(())
}

async fn current_blitz_rating(
    client: &LichessClient,
) -> Result<
    Option<u32>,
    Box<dyn std::error::Error + Send + Sync>,
> {
    let me =
        client
            .account()
            .profile()
            .await?;

    Ok(
        me.user
            .perfs
            .as_ref()
            .and_then(
                |perfs| {
                    perfs.blitz
                        .as_ref()
                        .map(
                            |blitz| {
                                blitz.rating
                            },
                        )
                },
            ),
    )
}

async fn fetch_online_player_usernames(
) -> Result<
    Vec<String>,
    Box<dyn std::error::Error + Send + Sync>,
> {
    let response =
        reqwest::Client::new()
            .get(ONLINE_HUMAN_PAGE_URL)
            .header(
                reqwest::header::USER_AGENT,
                "ZanLing-TrueZero Lichess matchmaking",
            )
            .send()
            .await?
            .error_for_status()?;

    let html =
        response.text().await?;

    let mut usernames =
        Vec::new();

    let mut seen =
        HashSet::new();

    let marker =
        "href=\"/@/";

    let mut search_start =
        0usize;

    while let Some(relative_start) =
        html[search_start..]
            .find(marker)
    {
        let start =
            search_start
                + relative_start
                + marker.len();

        let remainder =
            &html[start..];

        let end =
            match remainder
                .find('"')
            {
                Some(end) => end,

                None => break,
            };

        let username =
            &remainder[..end];

        if !username.is_empty()
            && !username.contains('/')
            && seen.insert(
                username.to_ascii_lowercase(),
            )
        {
            usernames.push(
                username.to_string(),
            );
        }

        search_start =
            start + end + 1;

        if usernames.len()
            >= MAX_ONLINE_HUMAN_CANDIDATES
        {
            break;
        }
    }

    Ok(usernames)
}

fn challenge_color_for_bot(
    username: &str,
) -> LichessChallengeColor {
    let hash =
        username.bytes().fold(
            0u64,
            |acc, byte| {
                acc.wrapping_mul(31)
                    .wrapping_add(byte as u64)
            },
        );

    if hash % 2 == 0 {
        LichessChallengeColor::White
    } else {
        LichessChallengeColor::Black
    }
}

#[cfg(test)]
mod graph_database_tests {
    use super::*;
    use std::{
        fs::remove_file,
        path::PathBuf,
    };

    fn temporary_database_path() -> PathBuf {
        env::temp_dir().join(format!(
            "truezero-player-graph-{}.sqlite3",
            std::process::id()
        ))
    }

    #[test]
    fn migrates_legacy_graph_and_keeps_normalized_relationships() {
        let path =
            temporary_database_path();
        let _ =
            remove_file(&path);
        let path_string =
            path.to_string_lossy().into_owned();

        {
            let legacy =
                Connection::open(&path_string).unwrap();
            legacy
                .execute_batch(
                    r#"
                    CREATE TABLE players (
                        username TEXT PRIMARY KEY,
                        is_bot INTEGER,
                        first_seen_at INTEGER NOT NULL,
                        last_seen_online_at INTEGER,
                        last_expanded_at INTEGER,
                        discovered_from TEXT
                    );
                    CREATE TABLE player_edges (
                        username TEXT NOT NULL,
                        opponent TEXT NOT NULL,
                        last_seen_at INTEGER NOT NULL,
                        PRIMARY KEY (username, opponent)
                    );
                    CREATE TABLE human_challenges (
                        username TEXT PRIMARY KEY,
                        last_challenged_at INTEGER NOT NULL
                    );
                    INSERT INTO players VALUES
                        ('HumanSeed', NULL, 1, 2, 3, 'online_human'),
                        ('Opponent', 0, 4, NULL, NULL, 'game: HumanSeed');
                    INSERT INTO player_edges VALUES
                        ('HumanSeed', 'Opponent', 5);
                    INSERT INTO human_challenges VALUES
                        ('HumanSeed', 6);
                    "#,
                )
                .unwrap();
        }

        let db =
            open_player_graph_database(&path_string)
                .unwrap();

        let player_count: i64 =
            db.query_row(
                "SELECT COUNT(*) FROM players",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let edge_count: i64 =
            db.query_row(
                "SELECT COUNT(*) FROM player_edges",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let challenge_count: i64 =
            db.query_row(
                "SELECT COUNT(*) FROM human_challenges",
                [],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(player_count, 2);
        assert_eq!(edge_count, 1);
        assert_eq!(challenge_count, 1);

        graph_insert_player(
            &db,
            "HumanSeed",
            None,
            "online_human",
        )
        .unwrap();
        graph_insert_edge(
            &db,
            "HumanSeed",
            "Opponent",
        )
        .unwrap();

        let discovery_count: i64 =
            db.query_row(
                "SELECT COUNT(*) FROM player_discoveries",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(discovery_count, 2);

        drop(db);
        remove_file(path).unwrap();
    }
}

async fn run_game_owned(
    client: Arc<LichessClient>,
    game_id: String,
    our_color: LichessColor,
    tensor_exe_send: Sender<Packet>,
    active_games: Arc<AtomicUsize>,
    api_request_lock: Arc<Mutex<()>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    run_game(
        &client,
        &game_id,
        our_color,
        &tensor_exe_send,
        active_games,
        api_request_lock,
    )
    .await
}

async fn run_game(
    client: &LichessClient,
    game_id: &str,
    our_color: LichessColor,
    tensor_exe_send: &Sender<Packet>,
    active_games: Arc<AtomicUsize>,
    api_request_lock: Arc<Mutex<()>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _guard =
        ActiveGameGuard::new(
            Arc::clone(&active_games),
        );

    let bot_api =
        client.bot();

    let mut cache:
        LruCache<
            CacheEntryKey,
            ZeroEvaluationAbs,
        > =
        LruCache::new(
            NonZeroUsize::new(
                LRU_CACHE_SIZE,
            )
            .unwrap(),
        );

    let mut reconnect_attempts =
        0u32;

    let mut opponent_username:
        Option<String> =
        None;

    let mut rematch_rated:
        Option<bool> =
        None;

    let mut rematch_clock:
        Option<(u32, u32)> =
        None;

    let mut last_processed_moves:
        Option<String> =
        None;

    loop {
        println!(
            "Opening game stream for {}...",
            game_id
        );

        let mut stream =
            match bot_api
                .stream_game(game_id)
                .await
            {
                Ok(stream) => {
                    println!(
                        "Game stream connected for {}.",
                        game_id
                    );

                    reconnect_attempts = 0;

                    stream
                }

                Err(error) => {
                    reconnect_attempts += 1;

                    eprintln!(
                        "Failed to open game stream {} \
                         (attempt {}/{}): {}",
                        game_id,
                        reconnect_attempts,
                        GAME_STREAM_MAX_RECONNECTS,
                        error
                    );

                    if reconnect_attempts >
                        GAME_STREAM_MAX_RECONNECTS
                    {
                        return Err(
                            format!(
                                "Game {} stream could not \
                                 be opened after {} attempts",
                                game_id,
                                GAME_STREAM_MAX_RECONNECTS
                            )
                            .into(),
                        );
                    }

                    sleep(
                        Duration::from_millis(
                            GAME_STREAM_RECONNECT_DELAY_MS,
                        ),
                    )
                    .await;

                    continue;
                }
            };

        let mut initial_fen =
            String::from("startpos");

        let mut game_stream_finished =
            false;

        while let Some(event_result) =
            stream.next().await
        {
            let event =
                match event_result {
                    Ok(event) => event,

                    Err(error) => {
                        eprintln!(
                            "Game {} stream error: {}",
                            game_id,
                            error
                        );

                        game_stream_finished =
                            true;

                        break;
                    }
                };

            match event {
                LichessBoardEvent::GameFull(
                    game,
                ) => {
                    println!(
                        "Game {}: received initial GameFull.",
                        game_id
                    );

                    initial_fen =
                        game
                            .initial_fen
                            .clone()
                            .unwrap_or_else(
                                || "startpos".to_string(),
                            );

                    opponent_username =
                        match our_color {
                            LichessColor::White => {
                                game.black
                                    .as_ref()
                                    .and_then(
                                        |player| {
                                            player
                                                .name
                                                .clone()
                                                .or_else(
                                                    || {
                                                        player
                                                            .id
                                                            .clone()
                                                    },
                                                )
                                        },
                                    )
                            }

                            LichessColor::Black => {
                                game.white
                                    .as_ref()
                                    .and_then(
                                        |player| {
                                            player
                                                .name
                                                .clone()
                                                .or_else(
                                                    || {
                                                        player
                                                            .id
                                                            .clone()
                                                    },
                                                )
                                        },
                                    )
                            }
                        };

                    rematch_rated =
                        game.rated;

                    rematch_clock =
                        game.clock.and_then(
                            |clock| {
                                let initial =
                                    clock
                                        .initial
                                        .unwrap_or(0);

                                let increment =
                                    clock
                                        .increment
                                        .unwrap_or(0);

                                if initial > 0 {
                                    Some((
                                        (initial / 1000)
                                            as u32,
                                        (increment.max(0)
                                            / 1000)
                                            as u32,
                                    ))
                                } else {
                                    None
                                }
                            },
                        );

                    if game_finished(
                        game.state.status,
                    ) {
                        println!(
                            "Game {} already finished: {:?}",
                            game_id,
                            game.state.status
                        );

                        offer_rematch(
                            client,
                            game_id,
                            opponent_username
                                .as_deref(),
                            our_color,
                            rematch_rated,
                            rematch_clock,
                            &api_request_lock,
                        )
                        .await;

                        return Ok(());
                    }

                    let current_moves =
                        game.state.moves.clone();

                    if last_processed_moves
                        .as_deref()
                        == Some(
                            current_moves.as_str(),
                        )
                    {
                        println!(
                            "Game {}: ignoring duplicate \
                             GameFull state with moves: {}",
                            game_id,
                            current_moves
                        );

                        continue;
                    }

                    println!(
                        "Game {} GameFull moves: {}",
                        game_id,
                        current_moves
                    );

                    if draw_offer_exists(
                        &game.state,
                        our_color,
                    ) {
                        println!(
                            "Game {}: opponent has offered a draw.",
                            game_id
                        );

                        if should_accept_low_time_draw(
                            &game.state,
                            our_color,
                        ) {
                            println!(
                                "Game {}: accepting draw due \
                                 to low time.",
                                game_id
                            );

                            let _api_guard =
                                api_request_lock
                                    .lock()
                                    .await;

                            bot_api
                                .handle_draw(
                                    game_id,
                                    true,
                                )
                                .await?;

                            return Ok(());
                        }
                    }

                    if !is_our_turn_from_position(
                        &initial_fen,
                        &current_moves,
                        our_color,
                    )? {
                        println!(
                            "Game {}: not our turn after GameFull.",
                            game_id
                        );

                        continue;
                    }

                    last_processed_moves =
                        Some(current_moves);

                    play_position(
                        &bot_api,
                        game_id,
                        &initial_fen,
                        &game.state,
                        our_color,
                        tensor_exe_send,
                        &mut cache,
                        active_games.load(
                            Ordering::Relaxed,
                        ),
                    )
                    .await?;
                }

                LichessBoardEvent::GameState(
                    state,
                ) => {
                    if game_finished(
                        state.status,
                    ) {
                        println!(
                            "Game {} finished: {:?}",
                            game_id,
                            state.status
                        );

                        offer_rematch(
                            client,
                            game_id,
                            opponent_username
                                .as_deref(),
                            our_color,
                            rematch_rated,
                            rematch_clock,
                            &api_request_lock,
                        )
                        .await;

                        return Ok(());
                    }

                    let current_moves =
                        state.moves.clone();

                    if last_processed_moves
                        .as_deref()
                        == Some(
                            current_moves.as_str(),
                        )
                    {
                        println!(
                            "Game {}: ignoring duplicate \
                             GameState; moves unchanged: {}",
                            game_id,
                            current_moves
                        );

                        continue;
                    }

                    println!(
                        "Game {} GameState moves: {}",
                        game_id,
                        current_moves
                    );

                    if draw_offer_exists(
                        &state,
                        our_color,
                    ) {
                        println!(
                            "Game {}: opponent has offered a draw.",
                            game_id
                        );

                        if should_accept_low_time_draw(
                            &state,
                            our_color,
                        ) {
                            println!(
                                "Game {}: accepting draw due \
                                 to low time.",
                                game_id
                            );

                            let _api_guard =
                                api_request_lock
                                    .lock()
                                    .await;

                            bot_api
                                .handle_draw(
                                    game_id,
                                    true,
                                )
                                .await?;

                            return Ok(());
                        }

                        println!(
                            "Game {}: draw offer detected but \
                             low-time acceptance condition is false.",
                            game_id
                        );
                    }

                    if !is_our_turn_from_position(
                        &initial_fen,
                        &current_moves,
                        our_color,
                    )? {
                        println!(
                            "Game {}: not our turn.",
                            game_id
                        );

                        continue;
                    }

                    last_processed_moves =
                        Some(current_moves);

                    play_position(
                        &bot_api,
                        game_id,
                        &initial_fen,
                        &state,
                        our_color,
                        tensor_exe_send,
                        &mut cache,
                        active_games.load(
                            Ordering::Relaxed,
                        ),
                    )
                    .await?;
                }

                LichessBoardEvent::OpponentGone(
                    opponent,
                ) => {
                    if opponent.gone {
                        println!(
                            "Game {}: opponent has left.",
                            game_id
                        );
                    } else {
                        println!(
                            "Game {}: opponent has returned.",
                            game_id
                        );
                    }
                }

                LichessBoardEvent::ChatLine(
                    chat,
                ) => {
                    println!(
                        "[Game {}][{:?}] {}: {}",
                        game_id,
                        chat.room,
                        chat.username,
                        chat.text
                    );
                }

                _ => {}
            }
        }

        if !game_stream_finished {
            println!(
                "Game {} stream ended unexpectedly.",
                game_id
            );
        }

        reconnect_attempts += 1;

        if reconnect_attempts >
            GAME_STREAM_MAX_RECONNECTS
        {
            return Err(
                format!(
                    "Game {} stream repeatedly \
                     ended/failed after {} reconnect attempts",
                    game_id,
                    GAME_STREAM_MAX_RECONNECTS
                )
                .into(),
            );
        }

        println!(
            "Reconnecting game {} stream in {} ms \
             (attempt {}/{}).",
            game_id,
            GAME_STREAM_RECONNECT_DELAY_MS,
            reconnect_attempts,
            GAME_STREAM_MAX_RECONNECTS
        );

        sleep(
            Duration::from_millis(
                GAME_STREAM_RECONNECT_DELAY_MS,
            ),
        )
        .await;
    }
}

async fn offer_rematch(
    client: &LichessClient,
    game_id: &str,
    opponent_username: Option<&str>,
    our_color: LichessColor,
    rated: Option<bool>,
    clock: Option<(u32, u32)>,
    api_request_lock: &Arc<Mutex<()>>,
) {
    if let Some(opponent_username) =
        opponent_username
    {
        let rematch_color =
            match our_color {
                LichessColor::White =>
                    LichessChallengeColor::Black,

                LichessColor::Black =>
                    LichessChallengeColor::White,
            };

        println!(
            "Offering rematch challenge to {}.",
            opponent_username
        );

        let mut request =
            client
                .challenges()
                .challenge(opponent_username)
                .color(rematch_color);

        if let Some(rated) = rated {
            request =
                request.rated(rated);
        }

        if let Some((limit, increment)) =
            clock
        {
            if limit > 0 {
                request =
                    request.clock(
                        limit,
                        increment,
                    );
            }
        }

        let result = {
            let _api_guard =
                api_request_lock.lock().await;

            request.send().await
        };

        match result {
            Ok(challenge) => {
                println!(
                    "Rematch challenge sent: \
                     id={}, status={:?}",
                    challenge.id,
                    challenge.status
                );
            }

            Err(error) => {
                eprintln!(
                    "Failed to offer rematch to {}: {}",
                    opponent_username,
                    error
                );
            }
        }

        return;
    }

    println!(
        "Opponent username unavailable for game {}. \
         Built-in Lichess AI rematch is unavailable through \
         the bot API authentication.",
        game_id
    );

    println!(
        "Skipping rematch for game {} because \
         /challenge/rematch-of requires the web:mobile \
         scope when called with a bearer token.",
        game_id
    );
}

fn is_no_time_limit_game(
    state: &LichessGameState,
) -> bool {
    let all_zero =
        state.wtime <= 0
            && state.btime <= 0
            && state.winc <= 0
            && state.binc <= 0;

    let massive_time =
        state.wtime >= 86_400_000
            || state.btime >= 86_400_000;

    all_zero || massive_time
}

fn draw_offer_exists(
    state: &LichessGameState,
    our_color: LichessColor,
) -> bool {
    match our_color {
        LichessColor::White =>
            state.bdraw.unwrap_or(false),

        LichessColor::Black =>
            state.wdraw.unwrap_or(false),
    }
}

fn should_accept_low_time_draw(
    state: &LichessGameState,
    our_color: LichessColor,
) -> bool {
    if is_no_time_limit_game(state) {
        return false;
    }

    if !draw_offer_exists(
        state,
        our_color,
    ) {
        return false;
    }

    let our_time_ms =
        match our_color {
            LichessColor::White =>
                state.wtime.max(0) as u64,

            LichessColor::Black =>
                state.btime.max(0) as u64,
        };

    our_time_ms <= DRAW_ACCEPT_TIME_MS
}

async fn play_position(
    bot_api: &litchee::api::gameplay::bot::BotApi<'_>,
    game_id: &str,
    initial_fen: &str,
    state: &LichessGameState,
    our_color: LichessColor,
    tensor_exe_send: &Sender<Packet>,
    cache: &mut LruCache<
        CacheEntryKey,
        ZeroEvaluationAbs,
    >,
    active_games_count: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let fen =
        fen_from_position(
            initial_fen,
            &state.moves,
        )?;

    println!(
        "Game {} current FEN: {}",
        game_id,
        fen
    );

    let clocked_game =
        !is_no_time_limit_game(state);

    let draw_offer_pending =
        draw_offer_exists(
            state,
            our_color,
        );

    let (uci_move, eval) =
        if draw_offer_pending
            && !clocked_game
        {
            println!(
                "Game {}: draw offer pending in \
                 non-timed game. Running quick evaluation.",
                game_id
            );

            let quick_nodes =
                500_000;

            let quick_time =
                5_000;

            let (q_move, q_eval) =
                get_engine_move(
                    &fen,
                    state,
                    our_color,
                    tensor_exe_send,
                    cache,
                    Some(quick_nodes),
                    Some(quick_time),
                    active_games_count,
                )
                .await?;

            if q_eval.is_finite()
                && q_eval <= DRAW_MAX_EVAL
            {
                println!(
                    "Game {}: quick eval {} <= {}. \
                     Accepting draw.",
                    game_id,
                    q_eval,
                    DRAW_MAX_EVAL
                );

                bot_api
                    .handle_draw(
                        game_id,
                        true,
                    )
                    .await?;

                return Ok(());
            }

            println!(
                "Game {}: quick eval {} > {}. \
                 Declining draw and running full search.",
                game_id,
                q_eval,
                DRAW_MAX_EVAL
            );

            let _ = q_move;

            get_engine_move(
                &fen,
                state,
                our_color,
                tensor_exe_send,
                cache,
                None,
                None,
                active_games_count,
            )
            .await?
        } else {
            get_engine_move(
                &fen,
                state,
                our_color,
                tensor_exe_send,
                cache,
                None,
                None,
                active_games_count,
            )
            .await?
        };

    println!(
        "Game {} engine wants to play UCI: {} \
         (eval={})",
        game_id,
        uci_move,
        eval
    );

    let our_time_ms =
        match our_color {
            LichessColor::White =>
                state.wtime.max(0) as u64,

            LichessColor::Black =>
                state.btime.max(0) as u64,
        };

    let low_on_time =
        clocked_game
            && our_time_ms
                <= DRAW_ACCEPT_TIME_MS;

    let not_advantageous =
        eval.is_finite()
            && eval <= DRAW_MAX_EVAL;

    if draw_offer_pending
        && (not_advantageous
            || low_on_time)
    {
        println!(
            "Game {}: accepting pending draw offer \
             (eval={}, low_on_time={})",
            game_id,
            eval,
            low_on_time
        );

        bot_api
            .handle_draw(
                game_id,
                true,
            )
            .await?;

        return Ok(());
    }

    let offer_draw =
        not_advantageous
            || low_on_time;

    let chess =
        chess_from_fen(&fen)?;

    let parsed_move:
        UciMove =
        uci_move.parse()?;

    let _legal_move =
        parsed_move
            .to_move(&chess)
            .map_err(|error| {
                format!(
                    "Engine returned illegal \
                     move {}: {}",
                    uci_move,
                    error
                )
            })?;

    println!(
        "Game {} verified legal move: {}",
        game_id,
        _legal_move
    );

    bot_api
        .make_move(
            game_id,
            &uci_move,
            offer_draw,
        )
        .await?;

    if offer_draw {
        println!(
            "Game {} played UCI move {} \
             and offered draw.",
            game_id,
            uci_move
        );
    } else {
        println!(
            "Game {} played UCI move {}.",
            game_id,
            uci_move
        );
    }

    Ok(())
}

async fn get_engine_move(
    fen: &str,
    state: &LichessGameState,
    our_color: LichessColor,
    tensor_exe_send: &Sender<Packet>,
    cache: &mut LruCache<
        CacheEntryKey,
        ZeroEvaluationAbs,
    >,
    node_limit_override: Option<u128>,
    time_limit_override: Option<u64>,
    active_games_count: usize,
) -> Result<
    (String, f64),
    Box<dyn std::error::Error + Send + Sync>,
> {
    println!(
        "Engine input: {}",
        fen
    );

    let chess =
        chess_from_fen(fen)?;

    let side_to_move =
        chess.turn();

    println!(
        "Side to move: {:?}",
        side_to_move
    );

    let expected_side =
        match our_color {
            LichessColor::White =>
                Color::White,

            LichessColor::Black =>
                Color::Black,
        };

    if side_to_move != expected_side {
        return Err(
            format!(
                "Engine called on wrong side: \
                 FEN says {:?}, bot is {:?}",
                side_to_move,
                expected_side
            )
            .into(),
        );
    }

    let white_time_ms =
        state.wtime.max(0) as u64;

    let black_time_ms =
        state.btime.max(0) as u64;

    let white_increment_ms =
        state.winc.max(0) as u64;

    let black_increment_ms =
        state.binc.max(0) as u64;

    println!(
        "White clock: {:.2}s",
        white_time_ms as f64 / 1000.0
    );

    println!(
        "Black clock: {:.2}s",
        black_time_ms as f64 / 1000.0
    );

    println!(
        "White increment: {:.2}s",
        white_increment_ms as f64 / 1000.0
    );

    println!(
        "Black increment: {:.2}s",
        black_increment_ms as f64 / 1000.0
    );

    let no_time_limit =
        is_no_time_limit_game(state);

    if no_time_limit
        && node_limit_override.is_none()
    {
        println!(
            "NO TIME LIMIT GAME DETECTED"
        );

        println!(
            "Using explicit unlimited-game \
             budget: {} nodes",
            NO_TIME_LIMIT_ENGINE_NODES
        );
    }

    let nodes: u128;
    let wall_timeout_ms: u64;

    let concurrent_games =
        active_games_count
            .max(1) as u128;

    if no_time_limit {
        nodes =
            node_limit_override
                .unwrap_or(
                    NO_TIME_LIMIT_ENGINE_NODES
                        as u128,
                );

        wall_timeout_ms =
            time_limit_override
                .unwrap_or(
                    NO_TIME_LIMIT_WALL_TIMEOUT_MS,
                );
    } else {
        let times:
            [Option<u64>; 2] =
            [
                Some(white_time_ms),
                Some(black_time_ms),
            ];

        let incs:
            [Option<u64>; 2] =
            [
                Some(white_increment_ms),
                Some(black_increment_ms),
            ];

        let stm_cozy =
            match side_to_move {
                Color::Black =>
                    cozy_chess::Color::Black,

                Color::White =>
                    cozy_chess::Color::White,
            };

        let (_time, calculated_nodes) =
            time_to_nodes(
                stm_cozy,
                times,
                incs,
                MOVES_TO_GO,
                MAX_ENGINE_TIME_MS,
            );

        println!(
            "time_to_nodes calculated {} nodes",
            calculated_nodes
        );

        let raw_nodes =
            if let Some(n) =
                node_limit_override
            {
                n
            } else if calculated_nodes
                == 0
            {
                println!(
                    "time_to_nodes returned 0 nodes; \
                     using fallback {} nodes.",
                    FALLBACK_ENGINE_NODES
                );

                FALLBACK_ENGINE_NODES
                    as u128
            } else {
                calculated_nodes
                    .min(
                        MAX_ENGINE_NODES
                            as u128,
                    )
            };

        nodes =
            (raw_nodes
                / concurrent_games)
                .max(100);

        wall_timeout_ms =
            time_limit_override
                .unwrap_or(
                    ENGINE_WALL_TIMEOUT_MS,
                );
    }

    println!(
        "Final node budget: {}",
        nodes
    );

    println!(
        "Final wall timeout: {} ms",
        wall_timeout_ms
    );

    let m_settings =
        MovesLeftSettings {
            moves_left_weight: 0.03,
            moves_left_clip: 20.0,
            moves_left_sharpness: 0.5,
        };

    let batch_size =
        1;

    let settings:
        SearchSettings =
        SearchSettings {
            fpu: FPUSettings {
                root_fpu: 0.5,
                children_fpu: 0.5,
            },

            wdl: EvalMode::Wdl,

            moves_left:
                Some(m_settings),

            c_puct: CPUCTSettings {
                root_c_puct: 3.0,
                children_c_puct: 2.0,
            },

            max_nodes:
                Some(nodes),

            alpha: 0.03,
            eps: 0.25,

            search_type:
                TrainerSearch(None),

            pst: PSTSettings {
                root_pst: 1.75,
                children_pst: 1.5,
            },

            batch_size,
        };

    let board =
        Board::from_fen(
            fen,
            false,
        )
        .map_err(|error| {
            format!(
                "Invalid engine FEN: {}",
                error
            )
        })?;

    let bs =
        BoardStack::new(board);

    println!(
        "Starting engine search: \
         max_nodes={}, wall_timeout={}ms",
        nodes,
        wall_timeout_ms
    );

    let search_result =
        timeout(
            Duration::from_millis(
                wall_timeout_ms,
            ),
            get_move(
                bs,
                tensor_exe_send.clone(),
                settings,
                None,
                cache,
            ),
        )
        .await;

    let (
        best_move,
        _eval,
        _pv,
        root_eval,
        searched_nodes,
    ) =
        match search_result {
            Ok(result) =>
                result,

            Err(_) => {
                eprintln!(
                    "ENGINE TIMEOUT: search exceeded \
                     {} ms",
                    wall_timeout_ms
                );

                let fallback =
                    first_legal_move(fen)?;

                return Ok((
                    fallback,
                    f64::INFINITY,
                ));
            }
        };

    println!(
        "ENGINE: searched {} nodes, \
         best move = {:?}",
        searched_nodes,
        best_move
    );

    let eval: f64 =
        root_eval.values.value
            as f64;

    println!(
        "ENGINE: root evaluation = {}",
        eval
    );

    let mut uci_move =
        format!(
            "{}{}",
            best_move.from,
            best_move.to
        );

    if let Some(promotion) =
        best_move.promotion
    {
        uci_move.push(
            match promotion {
                cozy_chess::Piece::Knight =>
                    'n',

                cozy_chess::Piece::Bishop =>
                    'b',

                cozy_chess::Piece::Rook =>
                    'r',

                cozy_chess::Piece::Queen =>
                    'q',

                cozy_chess::Piece::King =>
                    'k',

                cozy_chess::Piece::Pawn =>
                    'p',
            },
        );
    }

    eprintln!(
        "ENGINE: UCI move = {}",
        uci_move
    );

    let parsed_move:
        UciMove =
        uci_move.parse()
            .map_err(|e| {
                format!(
                    "Invalid generated \
                     UCI move {:?}: {:?}",
                    uci_move,
                    e
                )
            })?;

    if parsed_move
        .to_move(&chess)
        .is_err()
    {
        eprintln!(
            "ENGINE ERROR: generated UCI move {} \
             is illegal in {}",
            uci_move,
            fen
        );

        let fallback =
            first_legal_move(fen)?;

        return Ok((
            fallback,
            f64::INFINITY,
        ));
    }

    Ok((
        uci_move,
        eval,
    ))
}

fn first_legal_move(
    fen: &str,
) -> Result<
    String,
    Box<dyn std::error::Error + Send + Sync>,
> {
    let chess =
        chess_from_fen(fen)?;

    let legal_move =
        chess
            .legal_moves()
            .into_iter()
            .next()
            .ok_or_else(
                || {
                    "Position has no legal moves"
                        .to_string()
                },
            )?;

    let mut uci_move =
        format!(
            "{:?}{}",
            legal_move.from(),
            legal_move.to()
        );

    if let Some(promotion) =
        legal_move.promotion()
    {
        uci_move.push(
            match promotion {
                shakmaty::Role::Knight =>
                    'n',

                shakmaty::Role::Bishop =>
                    'b',

                shakmaty::Role::Rook =>
                    'r',

                shakmaty::Role::Queen =>
                    'q',

                shakmaty::Role::King =>
                    'k',

                shakmaty::Role::Pawn =>
                    'p',
            },
        );
    }

    eprintln!(
        "Using legal fallback UCI move: {}",
        uci_move
    );

    Ok(uci_move)
}

fn chess_from_fen(
    fen_string: &str,
) -> Result<
    Chess,
    Box<dyn std::error::Error + Send + Sync>,
> {
    let fen:
        Fen =
        fen_string.parse()?;

    let chess =
        fen.into_position::<Chess>(
            CastlingMode::Standard,
        )?;

    Ok(chess)
}

fn is_our_turn_from_position(
    initial_fen: &str,
    moves: &str,
    our_color: LichessColor,
) -> Result<
    bool,
    Box<dyn std::error::Error + Send + Sync>,
> {
    let fen =
        fen_from_position(
            initial_fen,
            moves,
        )?;

    let chess =
        chess_from_fen(&fen)?;

    let our_side =
        match our_color {
            LichessColor::White =>
                Color::White,

            LichessColor::Black =>
                Color::Black,
        };

    println!(
        "Turn check: moves='{}', FEN='{}', \
         side_to_move={:?}, our_side={:?}",
        moves,
        fen,
        chess.turn(),
        our_side
    );

    Ok(
        chess.turn()
            == our_side,
    )
}

fn game_finished(
    status: LichessGameStatusName,
) -> bool {
    matches!(
        status,
        LichessGameStatusName::Aborted
            | LichessGameStatusName::Mate
            | LichessGameStatusName::Resign
            | LichessGameStatusName::Stalemate
            | LichessGameStatusName::Timeout
            | LichessGameStatusName::Outoftime
            | LichessGameStatusName::Draw
            | LichessGameStatusName::Cheat
            | LichessGameStatusName::NoStart
            | LichessGameStatusName::UnknownFinish
            | LichessGameStatusName::VariantEnd
            | LichessGameStatusName::InsufficientMaterialClaim
    )
}

fn fen_from_position(
    initial_fen: &str,
    moves: &str,
) -> Result<
    String,
    Box<dyn std::error::Error + Send + Sync>,
> {
    let mut chess =
        if initial_fen == "startpos" {
            Chess::default()
        } else {
            let fen:
                Fen =
                initial_fen.parse()?;

            fen.into_position::<Chess>(
                CastlingMode::Standard,
            )?
        };

    let initial_turn =
        chess.turn();

    let mut ply_count =
        0usize;

    for uci in
        moves.split_whitespace()
    {
        if uci.is_empty() {
            continue;
        }

        let uci_move:
            UciMove =
            uci.parse()?;

        let chess_move =
            uci_move.to_move(
                &chess,
            )?;

        chess.play_unchecked(
            chess_move,
        );

        ply_count += 1;
    }

    let mut fen_string =
        Fen::from_position(
            &chess,
            EnPassantMode::Legal,
        )
        .to_string();

    let expected_turn =
        if ply_count % 2 == 0 {
            initial_turn
        } else {
            match initial_turn {
                Color::White =>
                    Color::Black,

                Color::Black =>
                    Color::White,
            }
        };

    let generated_turn =
        chess.turn();

    if generated_turn
        != expected_turn
    {
        eprintln!(
            "WARNING: shakmaty replay turn mismatch: \
             generated={:?}, expected={:?}, plies={}",
            generated_turn,
            expected_turn,
            ply_count
        );
    }

    let expected_turn_char =
        match expected_turn {
            Color::White =>
                "w",

            Color::Black =>
                "b",
        };

    let mut fields =
        fen_string
            .split_whitespace()
            .collect::<Vec<_>>();

    if fields.len() < 6 {
        return Err(
            format!(
                "Generated invalid FEN: {}",
                fen_string
            )
            .into(),
        );
    }

    fields[1] =
        expected_turn_char;

    fen_string =
        fields.join(" ");

    Ok(fen_string)
}
