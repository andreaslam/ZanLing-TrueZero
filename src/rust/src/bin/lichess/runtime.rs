use super::*;
use tzrust::lichess_graph::*;

pub(super) async fn game_loop(
    token: String,
    tensor_exe_send: Sender<Packet>,
    collector_send: Sender<CollectorMessage>,
    executor_ready_recv: Receiver<bool>,
    num_executors: usize,
    num_generators: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Waiting for the training server to provide and load the first network...");
    for _ in 0..num_executors {
        executor_ready_recv
            .recv_async()
            .await
            .map_err(|error| format!("Executor did not load the first network: {error}"))?;
    }
    println!("Initial network loaded; starting self-play and Lichess services.");

    let (pause_sender, pause_receiver) = watch::channel(false);
    let (discovery_sender, discovery_receiver) = watch::channel(true);

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

    println!("Logged in as {}", me.user.username);

    if let Some(perfs) = me.user.perfs.as_ref() {
        if let Some(blitz) = perfs.blitz.as_ref() {
            println!("Current Blitz rating: {}", blitz.rating);
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

    let api_request_lock = Arc::new(Mutex::new(()));
    let graph_db = Arc::new(Mutex::new(open_player_graph_database(
        PLAYER_GRAPH_DB_PATH,
    )?));

    let active_games = Arc::new(AtomicUsize::new(0));
    let max_concurrent_games = env_usize(
        "TZ_MAX_CONCURRENT_GAMES",
        DEFAULT_MAX_CONCURRENT_GAMES,
    )
    .max(1);
    let game_slots = Arc::new(Semaphore::new(max_concurrent_games));
    println!(
        "Lichess concurrent game limit: {}",
        max_concurrent_games
    );

    spawn_selfplay_generators(
        &tensor_exe_send,
        &collector_send,
        num_generators,
        pause_receiver,
    );

    let event_client = Arc::clone(&client);

    let event_tensor_send = tensor_exe_send.clone();

    let event_api_lock = Arc::clone(&api_request_lock);
    let event_graph_db = Arc::clone(&graph_db);
    let event_discovery_sender = discovery_sender.clone();

    let event_active_games = Arc::clone(&active_games);

    let event_task = tokio::spawn(async move {
        event_loop(
            event_client,
            event_tensor_send,
            collector_send.clone(),
            event_api_lock,
            event_active_games,
            game_slots,
            pause_sender,
            event_graph_db,
            event_discovery_sender,
        )
        .await
    });

    /*
     * Matchmaking has its own task and therefore can never block
     * consumption of the Lichess event stream.
     */
    let matchmaking_client = Arc::clone(&client);

    let matchmaking_api_lock = Arc::clone(&api_request_lock);
    let matchmaking_active_games = Arc::clone(&active_games);
    let matchmaking_graph_db = Arc::clone(&graph_db);
    let matchmaking_discovery_receiver = discovery_receiver;
    let matchmaking_discovery_sender = discovery_sender;

    let username = me.user.username.clone();

    let matchmaking_task = tokio::spawn(async move {
        matchmaking_loop(
            matchmaking_client,
            username,
            matchmaking_api_lock,
            matchmaking_active_games,
            matchmaking_graph_db,
            matchmaking_discovery_receiver,
            matchmaking_discovery_sender,
        )
        .await
    });

    tokio::select! {
        result = event_task => {
            return match result {
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
            };
        }

        result = matchmaking_task => {
            return match result {
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
            };
        }
    }
}

fn spawn_selfplay_generators(
    tensor_exe_send: &Sender<Packet>,
    collector_send: &Sender<CollectorMessage>,
    num_generators: usize,
    pause_receiver: watch::Receiver<bool>,
) {
    for id in 0..num_generators {
        let tensor_send = tensor_exe_send.clone();
        let collector = collector_send.clone();
        let generator_pause_receiver = pause_receiver.clone();
        tokio::spawn(async move {

            let datagen = DataGen { iterations: 1 };
            let settings = SearchSettings {
                fpu: FPUSettings {
                    root_fpu: 1.0,
                    children_fpu: 0.5,
                },
                wdl: EvalMode::Wdl,
                moves_left: Some(MovesLeftSettings {
                    moves_left_weight: 0.05,
                    moves_left_clip: 20.0,
                    moves_left_sharpness: 0.5,
                }),
                c_puct: CPUCTSettings {
                    root_c_puct: 2.0,
                    children_c_puct: 2.0,
                },
                max_nodes: Some(1600),
                alpha: 0.03,
                eps: 0.25,
                search_type: TrainerSearch(None),
                pst: PSTSettings {
                    root_pst: 1.5,
                    children_pst: 1.5,
                },
                batch_size: 1,
            };
            let mut cache = LruCache::new(NonZeroUsize::new(1600).unwrap());

            loop {
                let sim = datagen
                    .play_game(
                        &tensor_send,
                        &collector,
                        &settings,
                        id,
                        &mut cache,
                        None,
                        Some(generator_pause_receiver.clone()),
                    )
                    .await;
                if collector
                    .send_async(CollectorMessage::FinishedGame(sim))
                    .await
                    .is_err()
                {
                    eprintln!("Self-play generator {id} stopped: collector disconnected.");
                    break;
                }
            }
        });
    }
}

pub(super) async fn event_loop(
    client: Arc<LichessClient>,
    tensor_exe_send: Sender<Packet>,
    collector_send: Sender<CollectorMessage>,
    api_request_lock: Arc<Mutex<()>>,
    active_games: Arc<AtomicUsize>,
    game_slots: Arc<Semaphore>,
    pause_sender: watch::Sender<bool>,
    graph_db: Arc<Mutex<Connection>>,
    discovery_sender: watch::Sender<bool>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    loop {
        println!("Opening Lichess bot event stream...");

        let bot_api = client.bot();

        let mut events = match bot_api.stream_events().await {
            Ok(stream) => {
                println!("Lichess bot event stream connected.");

                stream
            }

            Err(error) => {
                eprintln!(
                    "Failed to open Lichess bot \
                         event stream: {}",
                    error
                );

                sleep(Duration::from_millis(EVENT_STREAM_RECONNECT_DELAY_MS)).await;

                continue;
            }
        };

        while let Some(event_result) = events.next().await {
            match event_result {
                Ok(LichessIncomingEvent::Challenge { challenge }) => {
                    println!("Incoming challenge event: {:?}", challenge);

                    if !matches!(challenge.status, LichessChallengeStatus::Created) {
                        println!(
                            "Ignoring challenge {} with status {:?}",
                            challenge.id, challenge.status
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

                    let _api_guard = api_request_lock.lock().await;

                    println!("Accepting incoming challenge {}.", challenge.id);

                    match client.challenges().accept(&challenge.id, None).await {
                        Ok(_) => {
                            println!("Accepted challenge {}", challenge.id);
                        }

                        Err(error) => {
                            eprintln!("Failed to accept challenge {}: {}", challenge.id, error);
                        }
                    }
                }

                Ok(LichessIncomingEvent::GameStart { game }) => {
                    let Some(game_id) = game.id else {
                        println!("GameStart without game ID");

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

                    println!("Game started: {} ({:?})", game_id, our_color);

                    let game_permit = match Arc::clone(&game_slots).try_acquire_owned() {
                        Ok(permit) => permit,
                        Err(_) => {
                            eprintln!(
                                "Game {} exceeds TZ_MAX_CONCURRENT_GAMES; \
                                 not starting another game worker.",
                                game_id
                            );
                            continue;
                        }
                    };

                    let game_client = Arc::clone(&client);

                    let game_tensor_send = tensor_exe_send.clone();

                    let game_active_games = Arc::clone(&active_games);

                    let game_api_lock = Arc::clone(&api_request_lock);
                    let game_collector_send = collector_send.clone();
                    let game_pause_sender = pause_sender.clone();
                    let game_graph_db = Arc::clone(&graph_db);
                    let game_discovery_sender = discovery_sender.clone();
                    let _ = game_pause_sender.send(true);

                    tokio::spawn(async move {
                        match run_game_owned(
                            game_client,
                            game_id.clone(),
                            our_color,
                            game_tensor_send,
                            game_collector_send,
                            Arc::clone(&game_active_games),
                            game_api_lock,
                            game_permit,
                        )
                        .await {
                            Ok(Some(opponent)) => {
                                let db = game_graph_db.lock().await;
                                match graph_remove_player(&db, &opponent) {
                                    Ok(true) => println!(
                                        "Removed successfully played player {} from graph.",
                                        opponent
                                    ),
                                    Ok(false) => println!(
                                        "Successfully played {}, but it was not in the graph.",
                                        opponent
                                    ),
                                    Err(error) => eprintln!(
                                        "Failed to remove successfully played player {}: {}",
                                        opponent, error
                                    ),
                                }
                                let _ = game_discovery_sender.send(true);
                            }
                            Ok(None) => {}
                            Err(error) => eprintln!("Game {} error: {}", game_id, error),
                        }

                        if game_active_games.load(Ordering::Acquire) == 0 {
                            let _ = game_pause_sender.send(false);
                        }

                        println!("Game {} finished.", game_id);
                    });
                }

                Ok(_other) => {
                    println!("Ignoring unrelated bot event.");
                }

                Err(error) => {
                    eprintln!("Lichess bot event stream error: {}", error);

                    break;
                }
            }
        }

        println!(
            "Global Lichess event stream ended; \
             reconnecting in {} ms...",
            EVENT_STREAM_RECONNECT_DELAY_MS
        );

        sleep(Duration::from_millis(EVENT_STREAM_RECONNECT_DELAY_MS)).await;
    }
}

pub(super) async fn matchmaking_loop(
    client: Arc<LichessClient>,
    our_username: String,
    api_request_lock: Arc<Mutex<()>>,
    active_games: Arc<AtomicUsize>,
    graph_db: Arc<Mutex<Connection>>,
    discovery_receiver: watch::Receiver<bool>,
    discovery_sender: watch::Sender<bool>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut bot_challenge_cooldowns: HashMap<String, Instant> = HashMap::new();

    let mut bot_pending_challenges: HashSet<String> = HashSet::new();

    let mut human_pending_challenges: HashSet<String> = HashSet::new();

    let mut scan_interval =
        tokio::time::interval(Duration::from_secs(ONLINE_BOT_SCAN_INTERVAL_SECS));

    loop {
        scan_interval.tick().await;

        wait_for_no_active_games(&active_games, "player discovery").await;

        if !*discovery_receiver.borrow() {
            continue;
        }

        /*
         * EXISTING BOT MATCHMAKING.
         */
        if let Err(error) = challenge_online_bots(
            &client,
            &our_username,
            &mut bot_challenge_cooldowns,
            &mut bot_pending_challenges,
            &api_request_lock,
        )
        .await
        {
            eprintln!("Online bot matchmaking error: {}", error);
        }

        let graph_has_capacity = {
            let db = graph_db.lock().await;
            graph_player_count(&db).map(|count| count < MAX_GRAPH_PLAYERS)
        };

        let graph_has_capacity = match graph_has_capacity {
            Ok(true) => {
                if let Err(error) = expand_player_graph(&client, &our_username, &graph_db).await {
                    eprintln!("Player graph expansion error: {}", error);
                }
                true
            }
            Ok(false) => {
                println!(
                    "Player graph is full; skipping discovery and leaving graph unchanged."
                );
                false
            }
            Err(error) => {
                eprintln!("Could not inspect player graph capacity: {}", error);
                false
            }
        };

        /*
         * EXISTING DIRECT ONLINE-HUMAN SOURCE.
         *
         * This remains unchanged conceptually and is still useful because
         * it discovers currently-online users independently of the graph.
         */
        if graph_has_capacity {
            if let Err(error) = challenge_online_humans(
                &client,
                &our_username,
                &mut human_pending_challenges,
                &api_request_lock,
                &graph_db,
            )
            .await
            {
                eprintln!("Online human matchmaking error: {}", error);
            }
        }

        let _ = discovery_sender.send(false);
    }

}

async fn wait_for_no_active_games(active_games: &AtomicUsize, workload: &str) {
    let mut paused = false;
    while active_games.load(Ordering::Acquire) > 0 {
        if !paused {
            println!("Pausing {workload} while Lichess games are active.");
            paused = true;
        }
        sleep(Duration::from_millis(250)).await;
    }
    if paused {
        println!("Resuming {workload}; no Lichess games are active.");
    }
}

/*
 * EXISTING BOT MATCHMAKING.
 */
pub(super) async fn challenge_online_bots(
    client: &LichessClient,
    our_username: &str,
    cooldowns: &mut HashMap<String, Instant>,
    pending_challenges: &mut HashSet<String>,
    api_request_lock: &Arc<Mutex<()>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Scanning currently-online Lichess bots...");

    let bot_api = client.bot();

    let mut online_bots = bot_api.online(None).await?;

    let now = Instant::now();

    cooldowns.retain(|_, last_attempt| {
        now.duration_since(*last_attempt).as_secs() < ONLINE_BOT_CHALLENGE_COOLDOWN_SECS
    });

    pending_challenges.retain(|username| cooldowns.contains_key(username));

    let mut online_count = 0usize;

    let mut challenge_count = 0usize;

    while let Some(bot_result) = online_bots.next().await {
        let bot = match bot_result {
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

        if bot.username.eq_ignore_ascii_case(our_username) {
            println!("Skipping ourselves: {}", bot.username);

            continue;
        }

        if challenge_count >= MAX_BOT_CHALLENGES_PER_SCAN {
            println!(
                "Reached bot matchmaking limit of {} \
                 challenge attempts this scan.",
                MAX_BOT_CHALLENGES_PER_SCAN
            );

            break;
        }

        if pending_challenges.contains(&bot.username) {
            println!(
                "Skipping {}: outgoing challenge \
                 already pending.",
                bot.username
            );

            continue;
        }

        if let Some(last_attempt) = cooldowns.get(&bot.username) {
            let age = now.duration_since(*last_attempt).as_secs();

            if age < ONLINE_BOT_CHALLENGE_COOLDOWN_SECS {
                println!(
                    "Skipping {}: challenge cooldown \
                     active ({}s old).",
                    bot.username, age
                );

                continue;
            }
        }

        cooldowns.insert(bot.username.clone(), now);

        pending_challenges.insert(bot.username.clone());

        challenge_count += 1;

        let challenge_color = challenge_color_for_bot(&bot.username);

        let time_control = random_time_control();
        let (initial_seconds, increment_seconds) = time_control.clock();

        println!(
            "Challenging online bot {} as {:?} with \
             {:?} ({}+{} rated) ({}/{})",
            bot.username,
            challenge_color,
            time_control,
            initial_seconds / 60,
            increment_seconds,
            challenge_count,
            MAX_BOT_CHALLENGES_PER_SCAN
        );

        let result = {
            let _api_guard = api_request_lock.lock().await;

            client
                .challenges()
                .challenge(&bot.username)
                .color(challenge_color)
                .rated(BOT_CHALLENGE_RATED)
                .clock(initial_seconds, increment_seconds)
                .send()
                .await
        };

        match result {
            Ok(challenge) => {
                println!(
                    "Challenge sent to {}: \
                     id={}, status={:?}",
                    bot.username, challenge.id, challenge.status
                );

                if !matches!(challenge.status, LichessChallengeStatus::Created) {
                    pending_challenges.remove(&bot.username);
                }
            }

            Err(error) => {
                eprintln!("Failed to challenge {}: {}", bot.username, error);

                pending_challenges.remove(&bot.username);
            }
        }

        if challenge_count < MAX_BOT_CHALLENGES_PER_SCAN {
            sleep(Duration::from_millis(BOT_CHALLENGE_SEND_DELAY_MS)).await;
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
