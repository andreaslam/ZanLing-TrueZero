use super::*;
use tzrust::lichess_graph::*;

pub(super) async fn game_loop(
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

    let active_games = Arc::new(AtomicUsize::new(0));

    let event_client = Arc::clone(&client);

    let event_tensor_send = tensor_exe_send.clone();

    let event_api_lock = Arc::clone(&api_request_lock);

    let event_active_games = Arc::clone(&active_games);

    let event_task = tokio::spawn(async move {
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
    let matchmaking_client = Arc::clone(&client);

    let matchmaking_api_lock = Arc::clone(&api_request_lock);

    let username = me.user.username.clone();

    let matchmaking_task = tokio::spawn(async move {
        matchmaking_loop(matchmaking_client, username, matchmaking_api_lock).await
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

pub(super) async fn event_loop(
    client: Arc<LichessClient>,
    tensor_exe_send: Sender<Packet>,
    api_request_lock: Arc<Mutex<()>>,
    active_games: Arc<AtomicUsize>,
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

                    let game_client = Arc::clone(&client);

                    let game_tensor_send = tensor_exe_send.clone();

                    let game_active_games = Arc::clone(&active_games);

                    let game_api_lock = Arc::clone(&api_request_lock);

                    tokio::spawn(async move {
                        if let Err(error) = run_game_owned(
                            game_client,
                            game_id.clone(),
                            our_color,
                            game_tensor_send,
                            game_active_games,
                            game_api_lock,
                        )
                        .await
                        {
                            eprintln!("Game {} error: {}", game_id, error);
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
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    /*
     * Open the graph database once for the lifetime of the matchmaking
     * task.
     */
    let graph_db = Arc::new(Mutex::new(open_player_graph_database(
        PLAYER_GRAPH_DB_PATH,
    )?));

    let mut bot_challenge_cooldowns: HashMap<String, Instant> = HashMap::new();

    let mut bot_pending_challenges: HashSet<String> = HashSet::new();

    let mut human_pending_challenges: HashSet<String> = HashSet::new();

    let mut scan_interval =
        tokio::time::interval(Duration::from_secs(ONLINE_BOT_SCAN_INTERVAL_SECS));

    loop {
        scan_interval.tick().await;

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

        /*
         * NEW GRAPH DISCOVERY SOURCE.
         *
         * This happens independently of the actual human challenge scan.
         */
        if let Err(error) = expand_player_graph(&client, &our_username, &graph_db).await {
            eprintln!("Player graph expansion error: {}", error);
        }

        /*
         * EXISTING DIRECT ONLINE-HUMAN SOURCE.
         *
         * This remains unchanged conceptually and is still useful because
         * it discovers currently-online users independently of the graph.
         */
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
