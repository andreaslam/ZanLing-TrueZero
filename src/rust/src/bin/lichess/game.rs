use super::*;

pub(super) async fn run_game_owned(
    client: Arc<LichessClient>,
    game_id: String,
    our_color: LichessColor,
    tensor_exe_send: Sender<Packet>,
    collector_send: Sender<CollectorMessage>,
    active_games: Arc<AtomicUsize>,
    api_request_lock: Arc<Mutex<()>>,
    _game_permit: OwnedSemaphorePermit,
) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
    run_game(
        &client,
        &game_id,
        our_color,
        &tensor_exe_send,
        collector_send,
        active_games,
        api_request_lock,
        _game_permit,
    )
    .await
}

pub(super) async fn run_game(
    client: &LichessClient,
    game_id: &str,
    our_color: LichessColor,
    tensor_exe_send: &Sender<Packet>,
    collector_send: Sender<CollectorMessage>,
    active_games: Arc<AtomicUsize>,
    api_request_lock: Arc<Mutex<()>>,
    _game_permit: OwnedSemaphorePermit,
) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
    let _guard = ActiveGameGuard::new(Arc::clone(&active_games));

    let bot_api = client.bot();

    let mut cache: LruCache<CacheEntryKey, ZeroEvaluationAbs> =
        LruCache::new(NonZeroUsize::new(LRU_CACHE_SIZE).unwrap());

    let mut reconnect_attempts = 0u32;

    let mut opponent_username: Option<String> = None;

    let mut rematch_rated: Option<bool> = None;

    let mut rematch_clock: Option<(u32, u32)> = None;

    let mut last_processed_moves: Option<String> = None;
    let mut positions = Vec::new();

    loop {
        println!("Opening game stream for {}...", game_id);

        let mut stream = match bot_api.stream_game(game_id).await {
            Ok(stream) => {
                println!("Game stream connected for {}.", game_id);

                reconnect_attempts = 0;

                stream
            }

            Err(error) => {
                reconnect_attempts += 1;

                eprintln!(
                    "Failed to open game stream {} \
                         (attempt {}/{}): {}",
                    game_id, reconnect_attempts, GAME_STREAM_MAX_RECONNECTS, error
                );

                if reconnect_attempts > GAME_STREAM_MAX_RECONNECTS {
                    return Err(format!(
                        "Game {} stream could not \
                                 be opened after {} attempts",
                        game_id, GAME_STREAM_MAX_RECONNECTS
                    )
                    .into());
                }

                sleep(Duration::from_millis(GAME_STREAM_RECONNECT_DELAY_MS)).await;

                continue;
            }
        };

        let mut initial_fen = String::from("startpos");

        let mut game_stream_finished = false;

        while let Some(event_result) = stream.next().await {
            let event = match event_result {
                Ok(event) => event,

                Err(error) => {
                    eprintln!("Game {} stream error: {}", game_id, error);

                    game_stream_finished = true;

                    break;
                }
            };

            match event {
                LichessBoardEvent::GameFull(game) => {
                    println!("Game {}: received initial GameFull.", game_id);

                    initial_fen = game
                        .initial_fen
                        .clone()
                        .unwrap_or_else(|| "startpos".to_string());

                    opponent_username =
                        match our_color {
                            LichessColor::White => game.black.as_ref().and_then(|player| {
                                player.name.clone().or_else(|| player.id.clone())
                            }),

                            LichessColor::Black => game.white.as_ref().and_then(|player| {
                                player.name.clone().or_else(|| player.id.clone())
                            }),
                        };

                    rematch_rated = game.rated;

                    rematch_clock = game.clock.and_then(|clock| {
                        let initial = clock.initial.unwrap_or(0);

                        let increment = clock.increment.unwrap_or(0);

                        if initial > 0 {
                            Some(((initial / 1000) as u32, (increment.max(0) / 1000) as u32))
                        } else {
                            None
                        }
                    });

                    if game_finished(game.state.status) {
                        println!("Game {} already finished: {:?}", game_id, game.state.status);

                        record_finished_game(
                            &collector_send,
                            &mut positions,
                            &initial_fen,
                            &game.state.moves,
                        )
                        .await?;

                        offer_rematch(
                            client,
                            game_id,
                            opponent_username.as_deref(),
                            our_color,
                            rematch_rated,
                            rematch_clock,
                            &api_request_lock,
                        )
                        .await;

                        return Ok(opponent_username.clone());
                    }

                    let current_moves = game.state.moves.clone();

                    if last_processed_moves.as_deref() == Some(current_moves.as_str()) {
                        println!(
                            "Game {}: ignoring duplicate \
                             GameFull state with moves: {}",
                            game_id, current_moves
                        );

                        continue;
                    }

                    println!("Game {} GameFull moves: {}", game_id, current_moves);

                    if draw_offer_exists(&game.state, our_color) {
                        println!("Game {}: opponent has offered a draw.", game_id);

                        if should_accept_low_time_draw(&game.state, our_color) {
                            println!(
                                "Game {}: accepting draw due \
                                 to low time.",
                                game_id
                            );

                            let _api_guard = api_request_lock.lock().await;

                            bot_api.handle_draw(game_id, true).await?;

                            record_finished_game(
                                &collector_send,
                                &mut positions,
                                &initial_fen,
                                &game.state.moves,
                            )
                            .await?;

                            return Ok(opponent_username.clone());
                        }
                    }

                    if !is_our_turn_from_position(&initial_fen, &current_moves, our_color)? {
                        println!("Game {}: not our turn after GameFull.", game_id);

                        continue;
                    }

                    last_processed_moves = Some(current_moves);

                    play_position(
                        &bot_api,
                        game_id,
                        &initial_fen,
                        &game.state,
                        our_color,
                        tensor_exe_send,
                        &mut cache,
                        active_games.load(Ordering::Relaxed),
                        &mut positions,
                    )
                    .await?;
                }

                LichessBoardEvent::GameState(state) => {
                    if game_finished(state.status) {
                        println!("Game {} finished: {:?}", game_id, state.status);
                        record_finished_game(
                            &collector_send,
                            &mut positions,
                            &initial_fen,
                            &state.moves,
                        )
                        .await?;

                        offer_rematch(
                            client,
                            game_id,
                            opponent_username.as_deref(),
                            our_color,
                            rematch_rated,
                            rematch_clock,
                            &api_request_lock,
                        )
                        .await;

                        return Ok(opponent_username.clone());
                    }

                    let current_moves = state.moves.clone();

                    if last_processed_moves.as_deref() == Some(current_moves.as_str()) {
                        println!(
                            "Game {}: ignoring duplicate \
                             GameState; moves unchanged: {}",
                            game_id, current_moves
                        );

                        continue;
                    }

                    println!("Game {} GameState moves: {}", game_id, current_moves);

                    if draw_offer_exists(&state, our_color) {
                        println!("Game {}: opponent has offered a draw.", game_id);

                        if should_accept_low_time_draw(&state, our_color) {
                            println!(
                                "Game {}: accepting draw due \
                                 to low time.",
                                game_id
                            );

                            let _api_guard = api_request_lock.lock().await;

                            bot_api.handle_draw(game_id, true).await?;

                            record_finished_game(
                                &collector_send,
                                &mut positions,
                                &initial_fen,
                                &state.moves,
                            )
                            .await?;

                            return Ok(opponent_username.clone());
                        }

                        println!(
                            "Game {}: draw offer detected but \
                             low-time acceptance condition is false.",
                            game_id
                        );
                    }

                    if !is_our_turn_from_position(&initial_fen, &current_moves, our_color)? {
                        println!("Game {}: not our turn.", game_id);

                        continue;
                    }

                    last_processed_moves = Some(current_moves);

                    play_position(
                        &bot_api,
                        game_id,
                        &initial_fen,
                        &state,
                        our_color,
                        tensor_exe_send,
                        &mut cache,
                        active_games.load(Ordering::Relaxed),
                        &mut positions,
                    )
                    .await?;
                }

                LichessBoardEvent::OpponentGone(opponent) => {
                    if opponent.gone {
                        println!("Game {}: opponent has left.", game_id);
                    } else {
                        println!("Game {}: opponent has returned.", game_id);
                    }
                }

                LichessBoardEvent::ChatLine(chat) => {
                    println!(
                        "[Game {}][{:?}] {}: {}",
                        game_id, chat.room, chat.username, chat.text
                    );
                }

                _ => {}
            }
        }

        if !game_stream_finished {
            println!("Game {} stream ended unexpectedly.", game_id);
        }

        reconnect_attempts += 1;

        if reconnect_attempts > GAME_STREAM_MAX_RECONNECTS {
            return Err(format!(
                "Game {} stream repeatedly \
                     ended/failed after {} reconnect attempts",
                game_id, GAME_STREAM_MAX_RECONNECTS
            )
            .into());
        }

        println!(
            "Reconnecting game {} stream in {} ms \
             (attempt {}/{}).",
            game_id, GAME_STREAM_RECONNECT_DELAY_MS, reconnect_attempts, GAME_STREAM_MAX_RECONNECTS
        );

        sleep(Duration::from_millis(GAME_STREAM_RECONNECT_DELAY_MS)).await;
    }
}

async fn record_finished_game(
    collector_send: &Sender<CollectorMessage>,
    positions: &mut Vec<TrainingPosition>,
    initial_fen: &str,
    moves: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let fen = fen_from_position(initial_fen, moves)?;
    let board = Board::from_fen(&fen, false)
        .map_err(|error| format!("Invalid final Lichess board FEN: {error}"))?;

    collector_send
        .send_async(CollectorMessage::FinishedGame(Simulation {
            positions: std::mem::take(positions),
            final_board: BoardStack::new(board),
        }))
        .await
        .map_err(|error| format!("Failed to send Lichess game to collector: {error}"))?;

    Ok(())
}

pub(super) async fn offer_rematch(
    client: &LichessClient,
    game_id: &str,
    opponent_username: Option<&str>,
    our_color: LichessColor,
    rated: Option<bool>,
    clock: Option<(u32, u32)>,
    api_request_lock: &Arc<Mutex<()>>,
) {
    if let Some(opponent_username) = opponent_username {
        let rematch_color = match our_color {
            LichessColor::White => LichessChallengeColor::Black,

            LichessColor::Black => LichessChallengeColor::White,
        };

        println!("Offering rematch challenge to {}.", opponent_username);

        let mut request = client
            .challenges()
            .challenge(opponent_username)
            .color(rematch_color);

        if let Some(rated) = rated {
            request = request.rated(rated);
        }

        if let Some((limit, increment)) = clock {
            if limit > 0 {
                request = request.clock(limit, increment);
            }
        }

        let result = {
            let _api_guard = api_request_lock.lock().await;

            request.send().await
        };

        match result {
            Ok(challenge) => {
                println!(
                    "Rematch challenge sent: \
                     id={}, status={:?}",
                    challenge.id, challenge.status
                );
            }

            Err(error) => {
                eprintln!(
                    "Failed to offer rematch to {}: {}",
                    opponent_username, error
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

pub(super) fn is_no_time_limit_game(state: &LichessGameState) -> bool {
    let all_zero = state.wtime <= 0 && state.btime <= 0 && state.winc <= 0 && state.binc <= 0;

    let massive_time = state.wtime >= 86_400_000 || state.btime >= 86_400_000;

    all_zero || massive_time
}

pub(super) fn draw_offer_exists(state: &LichessGameState, our_color: LichessColor) -> bool {
    match our_color {
        LichessColor::White => state.bdraw.unwrap_or(false),

        LichessColor::Black => state.wdraw.unwrap_or(false),
    }
}

pub(super) fn should_accept_low_time_draw(
    state: &LichessGameState,
    our_color: LichessColor,
) -> bool {
    if is_no_time_limit_game(state) {
        return false;
    }

    if !draw_offer_exists(state, our_color) {
        return false;
    }

    let our_time_ms = match our_color {
        LichessColor::White => state.wtime.max(0) as u64,

        LichessColor::Black => state.btime.max(0) as u64,
    };

    our_time_ms <= DRAW_ACCEPT_TIME_MS
}

pub(super) async fn play_position(
    bot_api: &litchee::api::gameplay::bot::BotApi<'_>,
    game_id: &str,
    initial_fen: &str,
    state: &LichessGameState,
    our_color: LichessColor,
    tensor_exe_send: &Sender<Packet>,
    cache: &mut LruCache<CacheEntryKey, ZeroEvaluationAbs>,
    active_games_count: usize,
    positions: &mut Vec<TrainingPosition>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let fen = fen_from_position(initial_fen, &state.moves)?;

    println!("Game {} current FEN: {}", game_id, fen);

    let clocked_game = !is_no_time_limit_game(state);

    let draw_offer_pending = draw_offer_exists(state, our_color);

    let (uci_move, eval, search_position) = if draw_offer_pending && !clocked_game {
        println!(
            "Game {}: draw offer pending in \
                 non-timed game. Running quick evaluation.",
            game_id
        );

        let quick_nodes = 500_000;

        let quick_time = 5_000;

        let (q_move, q_eval, _) = get_engine_move(
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

        if q_eval.is_finite() && q_eval <= DRAW_MAX_EVAL {
            println!(
                "Game {}: quick eval {} <= {}. \
                     Accepting draw.",
                game_id, q_eval, DRAW_MAX_EVAL
            );

            bot_api.handle_draw(game_id, true).await?;

            return Ok(());
        }

        println!(
            "Game {}: quick eval {} > {}. \
                 Declining draw and running full search.",
            game_id, q_eval, DRAW_MAX_EVAL
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
        game_id, uci_move, eval
    );

    let our_time_ms = match our_color {
        LichessColor::White => state.wtime.max(0) as u64,

        LichessColor::Black => state.btime.max(0) as u64,
    };

    let low_on_time = clocked_game && our_time_ms <= DRAW_ACCEPT_TIME_MS;

    let not_advantageous = eval.is_finite() && eval <= DRAW_MAX_EVAL;

    if draw_offer_pending && (not_advantageous || low_on_time) {
        println!(
            "Game {}: accepting pending draw offer \
             (eval={}, low_on_time={})",
            game_id, eval, low_on_time
        );

        bot_api.handle_draw(game_id, true).await?;

        return Ok(());
    }

    if let Some(training_position) = search_position {
        positions.push(training_position);
    }

    let offer_draw = not_advantageous || low_on_time;

    let chess = chess_from_fen(&fen)?;

    let parsed_move: UciMove = uci_move.parse()?;

    let _legal_move = parsed_move.to_move(&chess).map_err(|error| {
        format!(
            "Engine returned illegal \
                     move {}: {}",
            uci_move, error
        )
    })?;

    println!("Game {} verified legal move: {}", game_id, _legal_move);

    bot_api.make_move(game_id, &uci_move, offer_draw).await?;

    if offer_draw {
        println!(
            "Game {} played UCI move {} \
             and offered draw.",
            game_id, uci_move
        );
    } else {
        println!("Game {} played UCI move {}.", game_id, uci_move);
    }

    Ok(())
}

pub(super) async fn get_engine_move(
    fen: &str,
    state: &LichessGameState,
    our_color: LichessColor,
    tensor_exe_send: &Sender<Packet>,
    cache: &mut LruCache<CacheEntryKey, ZeroEvaluationAbs>,
    node_limit_override: Option<u128>,
    time_limit_override: Option<u64>,
    active_games_count: usize,
) -> Result<(String, f64, Option<TrainingPosition>), Box<dyn std::error::Error + Send + Sync>> {
    println!("Engine input: {}", fen);

    let chess = chess_from_fen(fen)?;

    let side_to_move = chess.turn();

    println!("Side to move: {:?}", side_to_move);

    let expected_side = match our_color {
        LichessColor::White => Color::White,

        LichessColor::Black => Color::Black,
    };

    if side_to_move != expected_side {
        return Err(format!(
            "Engine called on wrong side: \
                 FEN says {:?}, bot is {:?}",
            side_to_move, expected_side
        )
        .into());
    }

    let white_time_ms = state.wtime.max(0) as u64;

    let black_time_ms = state.btime.max(0) as u64;

    let white_increment_ms = state.winc.max(0) as u64;

    let black_increment_ms = state.binc.max(0) as u64;

    println!("White clock: {:.2}s", white_time_ms as f64 / 1000.0);

    println!("Black clock: {:.2}s", black_time_ms as f64 / 1000.0);

    println!(
        "White increment: {:.2}s",
        white_increment_ms as f64 / 1000.0
    );

    println!(
        "Black increment: {:.2}s",
        black_increment_ms as f64 / 1000.0
    );

    let no_time_limit = is_no_time_limit_game(state);

    if no_time_limit && node_limit_override.is_none() {
        println!("NO TIME LIMIT GAME DETECTED");

        println!(
            "Using explicit unlimited-game \
             budget: {} nodes",
            NO_TIME_LIMIT_ENGINE_NODES
        );
    }

    let nodes: u128;
    let wall_timeout_ms: u64;

    let concurrent_games = active_games_count.max(1) as u128;

    if no_time_limit {
        nodes = node_limit_override.unwrap_or(NO_TIME_LIMIT_ENGINE_NODES as u128);

        wall_timeout_ms = time_limit_override.unwrap_or(NO_TIME_LIMIT_WALL_TIMEOUT_MS);
    } else {
        let times: [Option<u64>; 2] = [Some(white_time_ms), Some(black_time_ms)];

        let incs: [Option<u64>; 2] = [Some(white_increment_ms), Some(black_increment_ms)];

        let stm_cozy = match side_to_move {
            Color::Black => cozy_chess::Color::Black,

            Color::White => cozy_chess::Color::White,
        };

        let (_time, calculated_nodes) =
            time_to_nodes(stm_cozy, times, incs, MOVES_TO_GO, MAX_ENGINE_TIME_MS);

        println!("time_to_nodes calculated {} nodes", calculated_nodes);

        let raw_nodes = if let Some(n) = node_limit_override {
            n
        } else if calculated_nodes == 0 {
            println!(
                "time_to_nodes returned 0 nodes; \
                     using fallback {} nodes.",
                FALLBACK_ENGINE_NODES
            );

            FALLBACK_ENGINE_NODES as u128
        } else {
            calculated_nodes.min(MAX_ENGINE_NODES as u128)
        };

        let allocated_ms = (_time.unwrap_or(0) as u64) / concurrent_games.max(1) as u64;
        let reserve_ms = if allocated_ms < 1_000 {
            SHORT_TIME_RESERVE_MS.min(allocated_ms.saturating_div(3))
        } else {
            ((allocated_ms as f64) * LONG_TIME_RESERVE_FRACTION) as u64
        };
        let search_budget_ms = allocated_ms
            .saturating_sub(reserve_ms)
            .max(MIN_SEARCH_TIME_MS);
        let time_budget_ms = time_limit_override
            .unwrap_or(search_budget_ms.min(ENGINE_WALL_TIMEOUT_MS))
            .min(search_budget_ms.max(MIN_SEARCH_TIME_MS));

        nodes = (raw_nodes / concurrent_games.max(1) as u128)
            .min((time_budget_ms as u128).saturating_div(5).max(1))
            .max(1);

        wall_timeout_ms = time_budget_ms;
    }

    println!("Final node budget: {}", nodes);

    println!("Final wall timeout: {} ms", wall_timeout_ms);

    let m_settings = MovesLeftSettings {
        moves_left_weight: 0.03,
        moves_left_clip: 20.0,
        moves_left_sharpness: 0.5,
    };

    let batch_size = 1;

    let settings: SearchSettings = SearchSettings {
        fpu: FPUSettings {
            root_fpu: 0.5,
            children_fpu: 0.5,
        },

        wdl: EvalMode::Wdl,

        moves_left: Some(m_settings),

        c_puct: CPUCTSettings {
            root_c_puct: 3.0,
            children_c_puct: 2.0,
        },

        max_nodes: Some(nodes),

        alpha: 0.03,
        eps: 0.25,

        search_type: TrainerSearch(None),

        pst: PSTSettings {
            root_pst: 1.75,
            children_pst: 1.5,
        },

        batch_size,
    };

    let board =
        Board::from_fen(fen, false).map_err(|error| format!("Invalid engine FEN: {}", error))?;

    let bs = BoardStack::new(board);

    println!(
        "Starting engine search: \
         max_nodes={}, wall_timeout={}ms",
        nodes, wall_timeout_ms
    );

    let (stop_sender, stop_receiver) = flume::bounded(1);
    let stop_task = tokio::spawn(async move {
        sleep(Duration::from_millis(wall_timeout_ms)).await;
        let _ = stop_sender.send(UCIMsg::UCIStopMessage);
    });

    let search_result = timeout(
        Duration::from_millis(wall_timeout_ms.saturating_add(ENGINE_TIMEOUT_GRACE_MS)),
        get_move(
            bs.clone(),
            tensor_exe_send.clone(),
            settings,
            Some(stop_receiver),
            cache,
        ),
    )
    .await;
    stop_task.abort();

    let (best_move, net_evaluation, _pv, search_data, searched_nodes) = match search_result {
        Ok(result) => result,

        Err(_) => {
            eprintln!(
                "ENGINE HARD TIMEOUT: search did not return within \
                     {} ms (+{} ms grace; node budget={}, active games={}); \
                     using legal fallback",
                wall_timeout_ms, ENGINE_TIMEOUT_GRACE_MS, nodes, active_games_count
            );

            let fallback = first_legal_move(fen)?;

            return Ok((fallback, f64::INFINITY, None));
        }
    };

    println!(
        "ENGINE: searched {} nodes, \
         best move = {:?}",
        searched_nodes, best_move
    );

    let eval: f64 = search_data.values.value as f64;

    println!("ENGINE: root evaluation = {}", eval);

    let from = best_move.from;
    let to = best_move.to;
    let mut uci_move = format!("{}{}", from, to);

    if let Some(promotion) = best_move.promotion {
        uci_move.push(match promotion {
            cozy_chess::Piece::Knight => 'n',

            cozy_chess::Piece::Bishop => 'b',

            cozy_chess::Piece::Rook => 'r',

            cozy_chess::Piece::Queen => 'q',

            cozy_chess::Piece::King => 'k',

            cozy_chess::Piece::Pawn => 'p',
        });
    }

    eprintln!("ENGINE: UCI move = {}", uci_move);

    let parsed_move: UciMove = uci_move.parse().map_err(|e| {
        format!(
            "Invalid generated \
                     UCI move {:?}: {:?}",
            uci_move, e
        )
    })?;

    if parsed_move.to_move(&chess).is_err() {
        eprintln!(
            "ENGINE ERROR: generated UCI move {} \
             is illegal in {}",
            uci_move, fen
        );

        let fallback = first_legal_move(fen)?;

        return Ok((fallback, f64::INFINITY, None));
    }

    let training_position = TrainingPosition {
        board: bs.clone(),
        is_full_search: true,
        played_mv: best_move,
        zero_visits: searched_nodes as u64,
        zero_evaluation: ZeroEvaluationPov {
            values: search_data.values.to_relative(bs.board().side_to_move()),
            policy: search_data.policy,
        },
        net_evaluation: ZeroEvaluationPov {
            values: net_evaluation.values.to_relative(bs.board().side_to_move()),
            policy: net_evaluation.policy,
        },
    };
    Ok((uci_move, eval, Some(training_position)))
}

pub(super) fn first_legal_move(
    fen: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let chess = chess_from_fen(fen)?;

    let legal_move = chess
        .legal_moves()
        .into_iter()
        .next()
        .ok_or_else(|| "TrainingPosition has no legal moves".to_string())?;

    let from = legal_move
        .from()
        .ok_or_else(|| "Legal move has no source square".to_string())?;
    let mut uci_move = format!("{}{}", from, legal_move.to());

    if let Some(promotion) = legal_move.promotion() {
        uci_move.push(match promotion {
            shakmaty::Role::Knight => 'n',

            shakmaty::Role::Bishop => 'b',

            shakmaty::Role::Rook => 'r',

            shakmaty::Role::Queen => 'q',

            shakmaty::Role::King => 'k',

            shakmaty::Role::Pawn => 'p',
        });
    }

    eprintln!("Using legal fallback UCI move: {}", uci_move);

    Ok(uci_move)
}

pub(super) fn chess_from_fen(
    fen_string: &str,
) -> Result<Chess, Box<dyn std::error::Error + Send + Sync>> {
    let fen: Fen = fen_string.parse()?;

    let chess = fen.into_position::<Chess>(CastlingMode::Standard)?;

    Ok(chess)
}

pub(super) fn is_our_turn_from_position(
    initial_fen: &str,
    moves: &str,
    our_color: LichessColor,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    let fen = fen_from_position(initial_fen, moves)?;

    let chess = chess_from_fen(&fen)?;

    let our_side = match our_color {
        LichessColor::White => Color::White,

        LichessColor::Black => Color::Black,
    };

    println!(
        "Turn check: moves='{}', FEN='{}', \
         side_to_move={:?}, our_side={:?}",
        moves,
        fen,
        chess.turn(),
        our_side
    );

    Ok(chess.turn() == our_side)
}

pub(super) fn game_finished(status: LichessGameStatusName) -> bool {
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

pub(super) fn fen_from_position(
    initial_fen: &str,
    moves: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mut chess = if initial_fen == "startpos" {
        Chess::default()
    } else {
        let fen: Fen = initial_fen.parse()?;

        fen.into_position::<Chess>(CastlingMode::Standard)?
    };

    let initial_turn = chess.turn();

    let mut ply_count = 0usize;

    for uci in moves.split_whitespace() {
        if uci.is_empty() {
            continue;
        }

        let uci_move: UciMove = uci.parse()?;

        let chess_move = uci_move.to_move(&chess)?;

        chess.play_unchecked(chess_move);

        ply_count += 1;
    }

    let fen_string = Fen::from_position(&chess, EnPassantMode::Legal).to_string();

    let expected_turn = if ply_count % 2 == 0 {
        initial_turn
    } else {
        match initial_turn {
            Color::White => Color::Black,

            Color::Black => Color::White,
        }
    };

    let generated_turn = chess.turn();

    if generated_turn != expected_turn {
        eprintln!(
            "WARNING: shakmaty replay turn mismatch: \
             generated={:?}, expected={:?}, plies={}",
            generated_turn, expected_turn, ply_count
        );
    }

    Ok(fen_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_preserves_side_to_move() {
        assert_eq!(
            fen_from_position("startpos", "").unwrap(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
        assert_eq!(
            fen_from_position("startpos", "e2e4").unwrap(),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        );
        assert_eq!(
            fen_from_position("startpos", "e2e4 e7e5").unwrap(),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        );
    }

    #[test]
    fn replay_handles_black_to_move_initial_fen() {
        let fen = fen_from_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "",
        )
        .unwrap();
        assert_eq!(fen.split_whitespace().nth(1), Some("b"));
    }
}
