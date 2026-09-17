use cozy_chess::Board;
use crossbeam::thread;
use dotenv::dotenv;
use flume::Sender;
use futures_util::StreamExt;
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
    model::LichessColor,
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
    mcts_trainer::{EvalMode, TypeRequest::UCISearch},
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
 * The event stream and matchmaking now run in separate Tokio tasks.
 *
 * Therefore an incoming challenge is consumed immediately even while
 * matchmaking is sleeping or sending outgoing challenges.
 *
 * All mutating Lichess API requests are nevertheless serialized through
 * `api_request_lock`, so we do not intentionally make concurrent API
 * requests.
 */
const ONLINE_BOT_SCAN_INTERVAL_SECS: u64 = 60;
const ONLINE_BOT_CHALLENGE_COOLDOWN_SECS: u64 = 1_800;

const MAX_CHALLENGES_PER_SCAN: usize = 5;
const CHALLENGE_SEND_DELAY_MS: u64 = 2_000;

/*
 * Outgoing bot challenge time control.
 *
 * 3+2 casual.
 */
const BOT_CHALLENGE_INITIAL_SECONDS: u32 = 180;
const BOT_CHALLENGE_INCREMENT_SECONDS: u32 = 2;
const BOT_CHALLENGE_RATED: bool = true;

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
        env::var("LICHESS_API_KEY").expect("LICHESS_API_KEY must be set");

    let net_path = String::from(
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

                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create game-loop Tokio runtime");

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

        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    })
    .expect("Thread scope failed")?;

    let _ = ctrl_send;

    Ok(())
}

/*
 * The important concurrency structure is here:
 *
 *             ┌─────────────────────────┐
 *             │ Lichess event stream    │
 *             │ continuously consumed   │
 *             └────────────┬────────────┘
 *                          │
 *              Challenge/GameStart/etc.
 *                          │
 *                          ▼
 *                    event_loop()
 *
 *             ┌─────────────────────────┐
 *             │ matchmaking_loop()      │
 *             │ independent task        │
 *             └────────────┬────────────┘
 *                          │
 *                          ▼
 *                  challenge requests
 *
 * Both paths use:
 *
 *             Arc<Mutex<()>>
 *
 * for mutating API requests.
 *
 * Thus:
 *
 *   event reception     = concurrent
 *   game execution      = concurrent
 *   matchmaking         = concurrent
 *   HTTP mutations      = serialized
 */
async fn game_loop(
    token: String,
    tensor_exe_send: Sender<Packet>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = Arc::new(
        LichessClient::builder()
            .token(&token)
            .build()
            .expect("Failed to build Lichess client"),
    );

    let me = client
        .account()
        .profile()
        .await
        .expect("Failed to fetch Lichess profile");

    println!(
        "Logged in as {}",
        me.user.username
    );

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

    /*
     * Both tasks are intentionally long-lived.
     *
     * If either task terminates unexpectedly, surface that as the
     * game-loop error instead of silently continuing with only half
     * of the bot's functionality.
     */
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

/*
 * Dedicated global Lichess event-stream task.
 *
 * IMPORTANT:
 *
 * This function NEVER waits for matchmaking.
 *
 * If matchmaking is currently sending challenge #4, this task can
 * still receive:
 *
 *   Challenge
 *   GameStart
 *   other bot events
 *
 * immediately.
 */
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

                    sleep(Duration::from_millis(
                        EVENT_STREAM_RECONNECT_DELAY_MS,
                    ))
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
                            "Ignoring challenge {} \
                             with status {:?}",
                            challenge.id,
                            challenge.status
                        );

                        continue;
                    }

                    if challenge.direction.as_deref()
                        != Some("in")
                    {
                        println!(
                            "Ignoring non-incoming \
                             challenge {} \
                             (direction={:?})",
                            challenge.id,
                            challenge.direction
                        );

                        continue;
                    }

                    println!(
                        "Incoming challenge {} from {:?}. \
                         Waiting for serialized API slot.",
                        challenge.id,
                        challenge.challenger
                    );

                    /*
                     * The event itself was received immediately.
                     *
                     * Only the HTTP mutation waits for the API lock.
                     *
                     * Therefore matchmaking cannot prevent us from
                     * seeing this challenge.
                     */
                    let _api_guard =
                        api_request_lock.lock().await;

                    println!(
                        "Accepting incoming challenge {}.",
                        challenge.id
                    );

                    match client
                        .challenges()
                        .accept(
                            &challenge.id,
                            None,
                        )
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
                                "Failed to accept \
                                 challenge {}: {}",
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

                    let Some(our_color) = game.color else {
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

                    /*
                     * Game execution is independent of:
                     *
                     *   - event processing
                     *   - matchmaking
                     *   - other games
                     *
                     * The game itself receives the shared API lock only
                     * when it needs to perform a mutating API request
                     * such as a rematch.
                     */
                    let game_client =
                        Arc::clone(&client);

                    let game_tensor_send =
                        tensor_exe_send.clone();

                    let game_active_games =
                        Arc::clone(
                            &active_games,
                        );

                    let game_api_lock =
                        Arc::clone(
                            &api_request_lock,
                        );

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

        sleep(Duration::from_millis(
            EVENT_STREAM_RECONNECT_DELAY_MS,
        ))
        .await;
    }
}

/*
 * Dedicated matchmaking task.
 *
 * This task has NO ownership of the event stream.
 *
 * Consequently:
 *
 *     challenge_online_bots()
 *
 * can take several seconds without blocking:
 *
 *     events.next().await
 */
async fn matchmaking_loop(
    client: Arc<LichessClient>,
    our_username: String,
    api_request_lock: Arc<Mutex<()>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut bot_challenge_cooldowns:
        HashMap<String, Instant> =
        HashMap::new();

    let mut pending_challenges:
        HashSet<String> =
        HashSet::new();

    let mut bot_scan_interval =
        tokio::time::interval(
            Duration::from_secs(
                ONLINE_BOT_SCAN_INTERVAL_SECS,
            ),
        );

    /*
     * Tokio's interval fires immediately on its first tick.
     *
     * Therefore the bot performs an initial scan immediately,
     * followed by scans every 60 seconds.
     */
    loop {
        bot_scan_interval.tick().await;

        if let Err(error) =
            challenge_online_bots(
                &client,
                &our_username,
                &mut bot_challenge_cooldowns,
                &mut pending_challenges,
                &api_request_lock,
            )
            .await
        {
            eprintln!(
                "Online-bot matchmaking error: {}",
                error
            );
        }
    }
}

/*
 * Challenge a bounded number of currently-online bots.
 *
 * IMPORTANT:
 *
 * This function may hold the API lock for an individual request,
 * but it NEVER holds the lock while waiting two seconds between
 * challenges.
 *
 * That is deliberate.
 *
 * Example:
 *
 *     send challenge #1
 *     release API lock
 *     wait 2 sec
 *     incoming challenge arrives
 *     event loop accepts it
 *     matchmaking sends challenge #2
 *
 * Thus the pacing delay does not block incoming challenge requests.
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

    /*
     * Remove expired cooldown entries.
     */
    cooldowns.retain(
        |_, last_attempt| {
            now.duration_since(*last_attempt)
                .as_secs()
                < ONLINE_BOT_CHALLENGE_COOLDOWN_SECS
        },
    );

    /*
     * Remove pending entries whose cooldown has expired.
     *
     * This gives us bounded local state even if Lichess never sends
     * an event that tells us exactly what happened to an outgoing
     * challenge.
     */
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
                        "Error while reading online bot list: {}",
                        error
                    );

                    continue;
                }
            };

        online_count += 1;

        if bot.username.eq_ignore_ascii_case(
            our_username,
        ) {
            println!(
                "Skipping ourselves: {}",
                bot.username
            );

            continue;
        }

        if challenge_count
            >= MAX_CHALLENGES_PER_SCAN
        {
            println!(
                "Reached matchmaking limit of {} \
                 challenge attempts this scan.",
                MAX_CHALLENGES_PER_SCAN
            );

            break;
        }

        /*
         * Do not challenge a target for which we already have
         * an outstanding challenge.
         */
        if pending_challenges.contains(
            &bot.username,
        ) {
            println!(
                "Skipping {}: outgoing challenge \
                 already pending.",
                bot.username
            );

            continue;
        }

        /*
         * Do not repeatedly challenge the same bot during the
         * cooldown period.
         */
        if let Some(last_attempt) =
            cooldowns.get(&bot.username)
        {
            let age =
                now.duration_since(*last_attempt)
                    .as_secs();

            if age
                < ONLINE_BOT_CHALLENGE_COOLDOWN_SECS
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

        /*
         * Record both states BEFORE making the request.
         *
         * This prevents another matchmaking iteration from selecting
         * the same target while the HTTP request is in flight.
         */
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
             with {}+{} casual \
             ({}/{})",
            bot.username,
            challenge_color,
            BOT_CHALLENGE_INITIAL_SECONDS / 60,
            BOT_CHALLENGE_INCREMENT_SECONDS,
            challenge_count,
            MAX_CHALLENGES_PER_SCAN
        );

        /*
         * Serialize ONLY the actual API request.
         *
         * We do not hold this lock during the subsequent 2-second
         * pacing delay.
         */
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

                /*
                 * If Lichess immediately tells us the challenge
                 * is no longer active, there is no reason to keep
                 * it in the pending set.
                 *
                 * The cooldown remains.
                 */
                if !matches!(
                    challenge.status,
                    LichessChallengeStatus::Created
                ) {
                    pending_challenges.remove(
                        &bot.username,
                    );
                }
            }

            Err(error) => {
                eprintln!(
                    "Failed to challenge {}: {}",
                    bot.username,
                    error
                );

                /*
                 * Keep the cooldown but remove the "pending"
                 * state because the request itself failed.
                 */
                pending_challenges.remove(
                    &bot.username,
                );
            }
        }

        /*
         * Deliberately pace outgoing challenge requests.
         *
         * IMPORTANT:
         *
         * No API lock is held during this sleep.
         *
         * Therefore an incoming challenge can be accepted during
         * this period.
         */
        if challenge_count
            < MAX_CHALLENGES_PER_SCAN
        {
            sleep(Duration::from_millis(
                CHALLENGE_SEND_DELAY_MS,
            ))
            .await;
        }
    }

    println!(
        "Online-bot scan complete: {} bots seen, \
         {} challenge attempts made, \
         {} outgoing challenges currently pending.",
        online_count,
        challenge_count,
        pending_challenges.len()
    );

    Ok(())
}

/*
 * Pick a deterministic colour from the bot username.
 *
 * There is no dependency on a Random challenge-colour enum.
 */
fn challenge_color_for_bot(
    username: &str,
) -> LichessChallengeColor {
    let hash =
        username
            .bytes()
            .fold(
                0u64,
                |acc, byte| {
                    acc.wrapping_mul(31)
                        .wrapping_add(
                            byte as u64,
                        )
                },
            );

    if hash % 2 == 0 {
        LichessChallengeColor::White
    } else {
        LichessChallengeColor::Black
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
        Option<String> = None;

    let mut rematch_rated:
        Option<bool> = None;

    let mut rematch_clock:
        Option<(u32, u32)> = None;

    /*
     * Lichess can send GameState again when a draw offer,
     * clock event, or other state change occurs without
     * changing the move list.
     *
     * Never search the exact same move list twice.
     */
    let mut last_processed_moves:
        Option<String> = None;

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

                    if reconnect_attempts
                        > GAME_STREAM_MAX_RECONNECTS
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

                    sleep(Duration::from_millis(
                        GAME_STREAM_RECONNECT_DELAY_MS,
                    ))
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

                        game_stream_finished = true;
                        break;
                    }
                };

            match event {
                LichessBoardEvent::GameFull(game) => {
                    println!(
                        "Game {}: received initial GameFull.",
                        game_id
                    );

                    initial_fen =
                        game.initial_fen
                            .clone()
                            .unwrap_or_else(
                                || {
                                    "startpos"
                                        .to_string()
                                },
                            );

                    opponent_username =
                        match our_color {
                            LichessColor::White =>
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
                                    ),

                            LichessColor::Black =>
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
                                    ),
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
                                        (
                                            initial
                                                / 1000
                                        )
                                            as u32,
                                        (
                                            increment
                                                .max(0)
                                                / 1000
                                        )
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

                    if !is_our_turn_from_position(
                        &initial_fen,
                        &current_moves,
                        our_color,
                    )? {
                        println!(
                            "Game {}: not our turn after \
                             GameFull.",
                            game_id
                        );

                        continue;
                    }

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
                            api_request_lock.lock().await;

                        bot_api
                            .handle_draw(
                                game_id,
                                true,
                            )
                            .await?;

                        return Ok(());
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
                            api_request_lock.lock().await;

                        bot_api
                            .handle_draw(
                                game_id,
                                true,
                            )
                            .await?;

                        return Ok(());
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

                        last_processed_moves =
                            Some(current_moves);

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

                LichessBoardEvent::ChatLine(chat) => {
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

        if reconnect_attempts
            > GAME_STREAM_MAX_RECONNECTS
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

        sleep(Duration::from_millis(
            GAME_STREAM_RECONNECT_DELAY_MS,
        ))
        .await;
    }
}

async fn offer_rematch(
    client: &LichessClient,
    opponent_username: Option<&str>,
    our_color: LichessColor,
    rated: Option<bool>,
    clock: Option<(u32, u32)>,
    api_request_lock: &Arc<Mutex<()>>,
) {
    let Some(opponent_username) =
        opponent_username
    else {
        println!(
            "Cannot offer rematch: opponent username unavailable."
        );

        return;
    };

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
            .challenge(
                opponent_username,
            )
            .color(rematch_color);

    if let Some(rated) = rated {
        request =
            request.rated(rated);
    }

    if let Some((
        limit,
        increment,
    )) = clock {
        if limit > 0 {
            request =
                request.clock(
                    limit,
                    increment,
                );
        }
    }

    /*
     * Rematches also go through the same serialized API-request
     * path as incoming challenge acceptance and matchmaking.
     */
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

            let (
                q_move,
                q_eval,
            ) =
                get_engine_move(
                    &fen,
                    state,
                    our_color,
                    tensor_exe_send,
                    cache,
                    Some(
                        quick_nodes,
                    ),
                    Some(
                        quick_time,
                    ),
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

            let _ =
                q_move;

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
            .map_err(
                |error| {
                    format!(
                        "Engine returned illegal \
                         move {}: {}",
                        uci_move,
                        error
                    )
                },
            )?;

    println!(
        "Game {} verified legal move: {}",
        game_id,
        _legal_move
    );

    /*
     * `make_move(..., offer_draw)` sends the move and,
     * if requested, offers a draw atomically.
     *
     * Do NOT call handle_draw(true) after this.
     */
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

    if side_to_move
        != expected_side
    {
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
        white_time_ms as f64
            / 1000.0
    );

    println!(
        "Black clock: {:.2}s",
        black_time_ms as f64
            / 1000.0
    );

    println!(
        "White increment: {:.2}s",
        white_increment_ms as f64
            / 1000.0
    );

    println!(
        "Black increment: {:.2}s",
        black_increment_ms as f64
            / 1000.0
    );

    let no_time_limit =
        is_no_time_limit_game(
            state,
        );

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

        let (
            _time,
            calculated_nodes,
        ) =
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
                Some(
                    m_settings,
                ),

            c_puct:
                CPUCTSettings {
                    root_c_puct: 3.0,
                    children_c_puct: 2.0,
                },

            max_nodes:
                Some(nodes),

            alpha: 0.03,
            eps: 0.25,

            search_type:
                UCISearch,

            pst:
                PSTSettings {
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
        .map_err(
            |error| {
                format!(
                    "Invalid engine FEN: {}",
                    error
                )
            },
        )?;

    let bs =
        BoardStack::new(
            board,
        );

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
                    first_legal_move(
                        fen,
                    )?;

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
        root_eval
            .values
            .value as f64;

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

    if let Some(
        promotion,
    ) = best_move.promotion
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
        uci_move
            .parse()
            .map_err(
                |e| {
                    format!(
                        "Invalid generated \
                         UCI move {:?}: {:?}",
                        uci_move,
                        e
                    )
                },
            )?;

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
            first_legal_move(
                fen,
            )?;

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
        chess_from_fen(
            fen,
        )?;

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

    if let Some(
        promotion,
    ) = legal_move.promotion()
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
        chess_from_fen(
            &fen,
        )?;

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
            uci_move
                .to_move(
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
