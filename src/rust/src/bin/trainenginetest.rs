use cozy_chess::{Board, GameStatus};

use crossbeam::thread;

use flume::{Receiver, Sender};

use futures::executor::ThreadPool;

use lru::LruCache;

use rand::seq::SliceRandom;

use sha2::{Digest, Sha256};

use std::{
    collections::HashSet,
    env,
    fs::{self, File},
    io::{self, BufRead, BufReader, Write},
    net::TcpStream,
    num::NonZeroUsize,
    panic,
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use tzrust::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    data_path, data_path_str,
    dataformat::ZeroEvaluationAbs,
    debug_print,
    elo::elo_wld,
    executor::{executor_main, Packet},
    mcts::get_move,
    mcts_trainer::{EvalMode, TypeRequest::NonTrainerSearch},
    message_types::{Entity, GateTesterResult, MessageServer, MessageType},
    settings::{CPUCTSettings, FPUSettings, MovesLeftSettings, PSTSettings, SearchSettings},
    utils::directory_exists,
};


// A paired gated test.
// Both games use the same opening, with the candidate and benchmark
// swapping colours. The pair is the unit dispatched to a worker.
#[derive(Clone, Debug)]
struct GamePair {
    test_id: usize,
    fen: String,
}


// Outcome of one game, reported back to the collector.
#[derive(Clone, Debug)]
struct GameOutcome {
    test_id: usize,
    engine0_white: bool,

    // None = draw
    // Some(true) = White won
    // Some(false) = Black won
    white_won: Option<bool>,
}


const NO_CANCELLED_TEST: usize = usize::MAX;

const OVERDISPATCH_MULTIPLIER: usize = 2;

const EARLY_REJECT_MIN_GAMES: usize = 100;

// H0: candidate has a 50% decisive-game win rate.
// H1: candidate has a 55% decisive-game win rate.
const SPRT_H0_SCORE: f64 = 0.50;
const SPRT_H1_SCORE: f64 = 0.55;

const SPRT_ALPHA: f64 = 0.05;
const SPRT_BETA: f64 = 0.05;


fn net_checksum(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}


/// Decisive-game SPRT log-likelihood ratio.
///
/// Draws are deliberately excluded.
///
/// H0: p(candidate win | decisive game) = 0.50
/// H1: p(candidate win | decisive game) = 0.55
///
/// For W decisive wins and L decisive losses:
///
/// LLR = W * ln(p1 / p0)
///     + L * ln((1-p1) / (1-p0))
fn sprt_log_likelihood_ratio(wins: u32, losses: u32) -> f64 {
    wins as f64 * (SPRT_H1_SCORE / SPRT_H0_SCORE).ln()
        + losses as f64
            * ((1.0 - SPRT_H1_SCORE) / (1.0 - SPRT_H0_SCORE)).ln()
}


/// Returns true if the candidate has crossed the lower SPRT boundary.
///
/// Draws do not contribute to the LLR and therefore do not count toward
/// the minimum number of decisive games.
fn should_early_reject(wins: u32, losses: u32) -> bool {
    let decisive_games = wins + losses;

    if decisive_games < EARLY_REJECT_MIN_GAMES as u32 {
        return false;
    }

    let reject_bound =
        (SPRT_BETA / (1.0 - SPRT_ALPHA)).ln();

    sprt_log_likelihood_ratio(wins, losses) <= reject_bound
}


fn main() {
    let pool = ThreadPool::builder()
        .pool_size(6)
        .create()
        .unwrap();

    env::set_var("RUST_BACKTRACE", "2");

    panic::set_hook(Box::new(|panic_info| {
        eprintln!("Panic occurred: {:?}", panic_info);
        std::process::exit(1);
    }));


    let mut stream = loop {
        match TcpStream::connect("127.0.0.1:38475") {
            Ok(s) => break s,
            Err(_) => continue,
        }
    };


    let message = MessageServer {
        purpose: MessageType::Initialise(Entity::GateTesterRunner),
    };

    let serialised =
        serde_json::to_string(&message).expect("serialisation failed");

    let serialised = serialised + "\n";

    stream
        .write_all(serialised.as_bytes())
        .expect("Failed to send data");

    println!("Connected to server!");


    let num_executors = 2;

    // Number of concurrent pair-playing workers.
    //
    // Each worker receives a GamePair and plays both colour-swapped games
    // before taking another pair.
    let num_workers = 2048;

    // Number of counted games = 2 * num_game_pairs.
    //
    // Each GamePair contains exactly two games.
    let num_game_pairs = 500;

    // Executor batch size.
    let executor_batch_size = 256;

    assert!(num_executors == 2);


    thread::scope(|s| {

        // ---------------------------------------------------------------
        // Per-executor net channels
        // ---------------------------------------------------------------

        let mut vec_communicate_exe_send: Vec<Sender<String>> = Vec::new();

        let mut vec_communicate_exe_recv: Vec<Receiver<String>> = Vec::new();

        for _ in 0..num_executors {
            let (send, recv) = flume::bounded::<String>(1);

            vec_communicate_exe_send.push(send);
            vec_communicate_exe_recv.push(recv);
        }


        // ---------------------------------------------------------------
        // Per-executor tensor channels
        // ---------------------------------------------------------------

        let (tensor_exe_send_0, tensor_exe_recv_0) =
            flume::bounded::<Packet>(num_workers);

        let (tensor_exe_send_1, tensor_exe_recv_1) =
            flume::bounded::<Packet>(num_workers);


        // ---------------------------------------------------------------
        // Executor ready signals
        // ---------------------------------------------------------------

        let (exe_send_signal_0, exe_recv_signal_0) =
            flume::bounded::<bool>(1);

        let (exe_send_signal_1, exe_recv_signal_1) =
            flume::bounded::<bool>(1);


        // ---------------------------------------------------------------
        // Pair dispatch and result collection
        // ---------------------------------------------------------------

        let (job_send, job_recv) =
            flume::bounded::<GamePair>(num_workers);

        let (result_send, result_recv) =
            flume::bounded::<GameOutcome>(num_workers);

        let cancelled_test =
            Arc::new(AtomicUsize::new(NO_CANCELLED_TEST));


        // ---------------------------------------------------------------
        // Commander
        // ---------------------------------------------------------------

        let commander_stream =
            stream.try_clone().expect("clone failed");

        let commander_exe_senders =
            vec_communicate_exe_send.clone();

        let commander_job_send =
            job_send.clone();

        let commander_ready0 =
            exe_recv_signal_0.clone();

        let commander_ready1 =
            exe_recv_signal_1.clone();


        s.builder()
            .name("commander".to_string())
            .spawn(move |_| {
                commander_main(
                    commander_exe_senders,
                    commander_stream,
                    commander_ready0,
                    commander_ready1,
                    commander_job_send,
                    num_game_pairs,
                )
            })
            .unwrap();


        // ---------------------------------------------------------------
        // Collector
        // ---------------------------------------------------------------

        let collector_stream =
            stream.try_clone().expect("clone failed");

        let collector_cancelled_test =
            cancelled_test.clone();


        s.builder()
            .name("collector".to_string())
            .spawn(move |_| {
                collector_main(
                    &result_recv,
                    collector_stream,
                    num_game_pairs,
                    collector_cancelled_test,
                )
            })
            .unwrap();


        // ---------------------------------------------------------------
        // Pair-playing workers
        // ---------------------------------------------------------------

        for n in 0..num_workers {
            let job_recv = job_recv.clone();

            let result_send = result_send.clone();

            let engine0 =
                tensor_exe_send_0.clone();

            let engine1 =
                tensor_exe_send_1.clone();

            let worker_cancelled_test =
                cancelled_test.clone();


            let fut = async move {
                worker_main(
                    job_recv,
                    result_send,
                    engine0,
                    engine1,
                    worker_cancelled_test,
                    n,
                )
                .await;
            };

            pool.spawn_ok(fut);
        }


        // ---------------------------------------------------------------
        // Executors
        // ---------------------------------------------------------------

        let mut recv_iter =
            vec_communicate_exe_recv.into_iter();

        let communicate_exe_recv_0 =
            recv_iter.next().unwrap();

        let communicate_exe_recv_1 =
            recv_iter.next().unwrap();


        s.builder()
            .name("executor_0".to_string())
            .spawn(move |_| {
                debug_print!("Spawning executor_0 thread");

                executor_main(
                    communicate_exe_recv_0,
                    tensor_exe_recv_0,
                    executor_batch_size,
                    None,
                    Some(exe_send_signal_0),
                    0,
                )
            })
            .unwrap();


        s.builder()
            .name("executor_1".to_string())
            .spawn(move |_| {
                debug_print!("Spawning executor_1 thread");

                executor_main(
                    communicate_exe_recv_1,
                    tensor_exe_recv_1,
                    executor_batch_size,
                    None,
                    Some(exe_send_signal_1),
                    1,
                )
            })
            .unwrap();
    })
    .unwrap();
}


fn read_epd_file(file_path: &str) -> io::Result<Vec<String>> {
    debug_print!("Reading EPD file: {}", file_path);

    let file = File::open(file_path)?;

    let reader = io::BufReader::new(file);

    let positions: Vec<String> =
        reader.lines().filter_map(|line| line.ok()).collect();

    Ok(positions)
}


async fn worker_main(
    job_recv: Receiver<GamePair>,
    result_send: Sender<GameOutcome>,
    tensor_exe_send_0: Sender<Packet>,
    tensor_exe_send_1: Sender<Packet>,
    cancelled_test: Arc<AtomicUsize>,
    id: usize,
) {
    debug_print!("Initialised worker-{}", id);


    let m_settings = MovesLeftSettings {
        moves_left_weight: 0.03,
        moves_left_clip: 20.0,
        moves_left_sharpness: 0.5,
    };


    let settings: SearchSettings = SearchSettings {
        fpu: FPUSettings {
            root_fpu: 0.6,
            children_fpu: 0.6,
        },

        wdl: EvalMode::Wdl,

        moves_left: Some(m_settings),

        c_puct: CPUCTSettings {
            root_c_puct: 4.0,
            children_c_puct: 2.0,
        },

        max_nodes: Some(100),

        alpha: 0.03,

        eps: 0.25,

        search_type: NonTrainerSearch,

        pst: PSTSettings {
            root_pst: 1.75,
            children_pst: 1.5,
        },

        batch_size: 1,
    };


    let engines = [
        tensor_exe_send_0.clone(),
        tensor_exe_send_1.clone(),
    ];


    while let Ok(pair) = job_recv.recv_async().await {

        if cancelled_test.load(Ordering::Acquire) == pair.test_id {
            continue;
        }


        // ===============================================================
        // Game 1: candidate = White, benchmark = Black
        // ===============================================================

        let outcome = play_game(
            &pair.fen,
            pair.test_id,
            true,
            &engines,
            settings,
            &cancelled_test,
            id,
        )
        .await;


        if let Some(outcome) = outcome {
            debug_print!(
                "worker-{} finished first game of pair: {:?}",
                id,
                outcome
            );

            if result_send.send_async(outcome).await.is_err() {
                return;
            }
        } else {
            // The test was cancelled during the first game.
            continue;
        }


        // ===============================================================
        // Game 2: benchmark = White, candidate = Black
        // ===============================================================

        if cancelled_test.load(Ordering::Acquire) == pair.test_id {
            continue;
        }


        let outcome = play_game(
            &pair.fen,
            pair.test_id,
            false,
            &engines,
            settings,
            &cancelled_test,
            id,
        )
        .await;


        if let Some(outcome) = outcome {
            debug_print!(
                "worker-{} finished second game of pair: {:?}",
                id,
                outcome
            );

            if result_send.send_async(outcome).await.is_err() {
                return;
            }
        }
    }
}


/// Play one game belonging to a GamePair.
///
/// Returns None if the current test was cancelled before the game
/// completed. A completed draw is returned as Some(GameOutcome) with
/// white_won = None.
async fn play_game(
    fen: &str,
    test_id: usize,
    engine0_white: bool,
    engines: &[Sender<Packet>; 2],
    settings: SearchSettings,
    cancelled_test: &Arc<AtomicUsize>,
    id: usize,
) -> Option<GameOutcome> {

    if cancelled_test.load(Ordering::Acquire) == test_id {
        return None;
    }


    let board =
        Board::from_fen(fen, false).unwrap();

    let mut bs =
        BoardStack::new(board);


    // engine0 moves first when candidate is White.
    //
    // engine0_white = true:
    //   White -> engine 0
    //   Black -> engine 1
    //
    // engine0_white = false:
    //   White -> engine 1
    //   Black -> engine 0
    let mut mover: usize =
        if engine0_white { 0 } else { 1 };


    let mut plies: usize = 0;


    let cache_0: LruCache<CacheEntryKey, ZeroEvaluationAbs> =
        LruCache::new(
            NonZeroUsize::new(
                settings.max_nodes.unwrap() as usize
            )
            .unwrap(),
        );


    let cache_1: LruCache<CacheEntryKey, ZeroEvaluationAbs> =
        LruCache::new(
            NonZeroUsize::new(
                settings.max_nodes.unwrap() as usize
            )
            .unwrap(),
        );


    let mut caches =
        [cache_0, cache_1];


    while bs.status() == GameStatus::Ongoing {

        if cancelled_test.load(Ordering::Acquire) == test_id {
            debug_print!(
                "worker-{} cancelled test {} during search",
                id,
                test_id
            );

            return None;
        }


        let engine =
            &engines[mover];

        let cache =
            &mut caches[mover];


        let (mv, _, _, _, _) =
            get_move(
                bs.clone(),
                engine.clone(),
                settings,
                None,
                cache,
            )
            .await;


        if cancelled_test.load(Ordering::Acquire) == test_id {
            debug_print!(
                "worker-{} cancelled test {} after search",
                id,
                test_id
            );

            return None;
        }


        bs.play(mv);

        mover = 1 - mover;

        plies += 1;
    }


    if bs.status() == GameStatus::Ongoing {
        return None;
    }


    // At a terminal position, the side to move has either been
    // checkmated or stalemated.
    //
    // plies = number of moves actually played.
    //
    // If plies is even, White made the last move.
    // If plies is odd, Black made the last move.
    let white_won: Option<bool> =
        match bs.status() {
            GameStatus::Drawn => None,

            GameStatus::Won =>
                Some(plies % 2 == 0),

            GameStatus::Ongoing =>
                panic!("Game is still ongoing!"),
        };


    Some(GameOutcome {
        test_id,
        engine0_white,
        white_won,
    })
}


fn collector_main(
    receiver: &Receiver<GameOutcome>,
    mut server_handle: TcpStream,
    num_game_pairs: usize,
    cancelled_test: Arc<AtomicUsize>,
) {
    // num_game_pairs means counted pairs.
    //
    // Each pair contains two games, so the fixed-count test contains
    // exactly 2 * num_game_pairs games.
    let games_per_test =
        num_game_pairs * 2;


    // (wins, losses, draws)
    //
    // All three are retained for reporting.
    // Only wins and losses enter the SPRT.
    let mut results =
        (0u32, 0u32, 0u32);


    let mut counter =
        0usize;


    let mut active_test_id:
        Option<usize> = None;


    let mut completed_tests:
        HashSet<usize> = HashSet::new();


    debug_print!("Collector main started");


    loop {
        let msg =
            receiver.recv().unwrap();


        if completed_tests.contains(&msg.test_id) {
            println!(
                "[sprt] Ignoring late result for completed test {}",
                msg.test_id
            );

            continue;
        }


        match active_test_id {

            Some(test_id) if test_id != msg.test_id => {
                println!(
                    "[sprt] Ignoring out-of-order result for test {} while test {} is active",
                    msg.test_id,
                    test_id
                );

                continue;
            }


            None => {
                active_test_id =
                    Some(msg.test_id);

                results =
                    (0, 0, 0);

                counter =
                    0;

                println!(
                    "[sprt] Collecting results for test {}",
                    msg.test_id
                );
            }


            _ => {}
        }


        // ---------------------------------------------------------------
        // Convert the result to the candidate's perspective.
        //
        // Draws are recorded but are NOT converted into half a win.
        // ---------------------------------------------------------------

        match msg.white_won {

            None => {
                results.2 += 1;
            }


            Some(white_won) => {
                let engine0_won =
                    white_won == msg.engine0_white;


                if engine0_won {
                    results.0 += 1;
                } else {
                    results.1 += 1;
                }
            }
        }


        counter += 1;


        if counter % 50 == 0 {
            let decisive =
                results.0 + results.1;


            let decisive_win_rate =
                if decisive > 0 {
                    results.0 as f64
                        / decisive as f64
                } else {
                    0.0
                };


            println!(
                "[sprt] progress {}/{} games  W:{} L:{} D:{} decisive:{} decisive-WR:{:.3}",
                counter,
                games_per_test,
                results.0,
                results.1,
                results.2,
                decisive,
                decisive_win_rate
            );
        }


        // ---------------------------------------------------------------
        // Sequential rejection.
        //
        // Only decisive games contribute.
        // ---------------------------------------------------------------

        let early_reject =
            should_early_reject(
                results.0,
                results.1,
            );


        // ---------------------------------------------------------------
        // Fixed-count completion.
        //
        // The fixed count is in TOTAL GAMES, including draws.
        // This means a draw consumes one game of the test but contributes
        // no evidence to the decisive-game SPRT.
        // ---------------------------------------------------------------

        if early_reject || counter >= games_per_test {

            let decisive =
                results.0 + results.1;


            let decisive_win_rate =
                if decisive > 0 {
                    results.0 as f64
                        / decisive as f64
                } else {
                    0.0
                };


            let elo =
                elo_wld(
                    results.0,
                    results.1,
                    results.2,
                );


            // No draw-as-half-point calculation here.
            //
            // A candidate passes the fixed-count gate only if:
            //
            //   W / (W + L) > 0.55
            //
            // Draws are ignored for this criterion.
            let accept_new_net =
                !early_reject
                    && decisive > 0
                    && decisive_win_rate > SPRT_H1_SCORE;


            let res =
                GateTesterResult {
                    elo,
                    accept_new_net,
                };


            if let Some(test_id) =
                active_test_id
            {
                cancelled_test.store(
                    test_id,
                    Ordering::Release,
                );
            }


            println!(
                "[sprt] Test complete: W:{} L:{} D:{} decisive:{} decisive-WR:{:.3} elo:{:?} accept:{} reason:{}",
                results.0,
                results.1,
                results.2,
                decisive,
                decisive_win_rate,
                elo,
                res.accept_new_net,
                if early_reject {
                    "early-reject"
                } else {
                    "fixed-count"
                }
            );


            let message =
                MessageServer {
                    purpose:
                        MessageType::TestResult(res),
                };


            let mut serialised =
                serde_json::to_string(&message)
                    .expect("serialisation failed");

            serialised += "\n";


            server_handle
                .write_all(serialised.as_bytes())
                .unwrap();


            // -----------------------------------------------------------
            // Reset for next candidate.
            // -----------------------------------------------------------

            results =
                (0, 0, 0);

            counter =
                0;


            if let Some(test_id) =
                active_test_id.take()
            {
                completed_tests.insert(test_id);
            }
        }
    }
}


fn commander_main(
    vec_exe_sender: Vec<Sender<String>>,
    server_handle: TcpStream,
    ready0: Receiver<bool>,
    ready1: Receiver<bool>,
    job_send: Sender<GamePair>,
    num_game_pairs: usize,
) {
    let mut is_initialised =
        false;


    let mut cloned_handle =
        server_handle.try_clone().unwrap();


    let mut reader =
        BufReader::new(
            server_handle.try_clone().unwrap()
        );


    let mut net_path_counter =
        0usize;


    let generator_id:
        usize = 0;


    // H0 = current benchmark net.
    // H1 = candidate net under test.
    let mut h0_path =
        String::new();

    let mut h1_path =
        String::new();

    let mut h0_checksum =
        String::new();

    let mut h1_checksum =
        String::new();


    // Benchmark net has been loaded into both executors.
    let mut h0_loaded =
        false;


    let mut last_h0_file =
        String::new();


    let mut net_save_timestamp =
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_nanos();


    // Candidate has been loaded but jobs have not yet been dispatched.
    let mut pending_jobs =
        false;


    let mut active_test =
        false;


    let mut next_test_id =
        0usize;


    loop {

        if !directory_exists(
            &data_path_str("nets")
        ) {
            fs::create_dir(
                data_path("nets")
            )
            .unwrap();
        }


        let mut recv_msg =
            String::new();


        if reader
            .read_line(&mut recv_msg)
            .is_err()
        {
            return;
        }


        let message =
            match serde_json::from_str::<MessageServer>(
                &recv_msg
            ) {
                Ok(message) =>
                    message,

                Err(_) => {
                    recv_msg.clear();
                    continue;
                }
            };


        if is_initialised {

            match message.purpose {

                // =======================================================
                // New candidate network
                // =======================================================

                MessageType::NewNetworkData(data) => {

                    let checksum =
                        net_checksum(&data);


                    if checksum == h0_checksum
                        || checksum == h1_checksum
                    {
                        println!(
                            "[sprt] Ignoring duplicate net {}",
                            checksum
                        );

                        recv_msg.clear();
                        continue;
                    }


                    if active_test
                        || pending_jobs
                    {
                        println!(
                            "[sprt][Warning] Ignoring candidate {} while test {} is still active",
                            checksum,
                            next_test_id.saturating_sub(1)
                        );

                        recv_msg.clear();
                        continue;
                    }


                    let new_path =
                        data_path_str(
                            &format!(
                                "nets/tz_sprt_temp_net_{}_{}_{}.pt",
                                generator_id,
                                net_path_counter,
                                net_save_timestamp
                            )
                        );


                    let mut file =
                        File::create(
                            new_path.clone()
                        )
                        .expect(
                            "Unable to create data"
                        );


                    file.write_all(&data)
                        .expect(
                            "Unable to write data"
                        );


                    net_save_timestamp =
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("Time went backwards")
                            .as_nanos();


                    net_path_counter += 1;


                    // ===================================================
                    // Bootstrap H0.
                    // ===================================================

                    if !h0_loaded {

                        h0_path =
                            new_path;

                        h0_checksum =
                            checksum;


                        println!(
                            "[sprt] H0 (benchmark) set to {} ({})",
                            h0_path,
                            h0_checksum
                        );


                        // Bootstrap both engines with the same net.
                        vec_exe_sender[0]
                            .send(h0_path.clone())
                            .unwrap();

                        vec_exe_sender[1]
                            .send(h0_path.clone())
                            .unwrap();


                        h0_loaded =
                            true;


                        // Consume bootstrap ready signals.
                        ready0.recv().unwrap();
                        ready1.recv().unwrap();

                    } else {

                        // ===============================================
                        // Candidate H1.
                        // ===============================================

                        h1_path =
                            new_path;

                        h1_checksum =
                            checksum;


                        println!(
                            "[sprt] H1 (candidate) set to {} ({})  (benchmark H0 = {} / {})",
                            h1_path,
                            h1_checksum,
                            h0_path,
                            h0_checksum
                        );


                        // Drain any stale candidate-ready signal.
                        let _ =
                            ready0.try_recv();


                        vec_exe_sender[0]
                            .send(h1_path.clone())
                            .unwrap();


                        pending_jobs =
                            true;
                    }
                }


                // =======================================================
                // Test result
                // =======================================================

                MessageType::TestResult(result) => {

                    if result.accept_new_net {

                        println!(
                            "[sprt] Candidate ACCEPTED, promoting H1 -> H0 benchmark"
                        );


                        let old_h0 =
                            std::mem::replace(
                                &mut h0_path,
                                h1_path.clone()
                            );


                        h0_checksum =
                            h1_checksum.clone();


                        if !old_h0.is_empty() {
                            last_h0_file =
                                old_h0;
                        }


                        // Load the accepted candidate as the new
                        // benchmark into engine_1.
                        vec_exe_sender[1]
                            .send(h0_path.clone())
                            .unwrap();


                        ready1.recv().unwrap();


                        if !last_h0_file.is_empty()
                            && Path::new(
                                &last_h0_file
                            )
                            .is_file()
                        {

                            match fs::remove_file(
                                last_h0_file.clone()
                            ) {

                                Ok(_) => {
                                    println!(
                                        "[sprt] Deleted superseded benchmark {}",
                                        last_h0_file
                                    );
                                }


                                Err(e) => {
                                    eprintln!(
                                        "[sprt] Error deleting {}: {}",
                                        last_h0_file,
                                        e
                                    );
                                }
                            }


                            last_h0_file.clear();
                        }

                    } else {

                        println!(
                            "[sprt] Candidate REJECTED, keeping H0 = {}",
                            h0_path
                        );


                        if !h1_path.is_empty()
                            && Path::new(
                                &h1_path
                            )
                            .is_file()
                        {

                            match fs::remove_file(
                                h1_path.clone()
                            ) {

                                Ok(_) => {
                                    println!(
                                        "[sprt] Deleted rejected candidate {}",
                                        h1_path
                                    );
                                }


                                Err(e) => {
                                    eprintln!(
                                        "[sprt] Error deleting {}: {}",
                                        h1_path,
                                        e
                                    );
                                }
                            }
                        }
                    }


                    h1_path.clear();
                    h1_checksum.clear();

                    active_test =
                        false;
                }


                _ => {}
            }

        } else if let MessageType::IdentityConfirmation(
            (entity, _)
        ) = message.purpose {

            match entity {

                Entity::GateTesterRunner => {
                    is_initialised =
                        true;
                }

                _ => {
                    println!(
                        "[Warning] Wrong entity, got {:?}",
                        entity
                    );
                }
            }
        }


        // ===============================================================
        // Request initial H0 if necessary.
        // ===============================================================

        if !h0_loaded {

            let message =
                MessageServer {
                    purpose:
                        MessageType::RequestingNet(),
                };


            let mut serialised =
                serde_json::to_string(&message)
                    .expect(
                        "serialisation failed"
                    );


            serialised += "\n";


            cloned_handle
                .write_all(
                    serialised.as_bytes()
                )
                .unwrap();
        }


        // ===============================================================
        // Dispatch the test as PAIRS.
        //
        // Each queued item is one GamePair.
        //
        // A worker takes the pair and plays:
        //
        //   Game 1: candidate White
        //   Game 2: candidate Black
        //
        // from the exact same opening.
        // ===============================================================

        if pending_jobs
            && !h1_path.is_empty()
        {

            // Wait for candidate engine to acknowledge its new net.
            ready0.recv().unwrap();


            // Drain any stale benchmark-ready signal.
            let _ =
                ready1.try_recv();


            let total_pairs_dispatched =
                num_game_pairs
                    * OVERDISPATCH_MULTIPLIER;


            println!(
                "[sprt] Both engines ready. Dispatching test {}: {} counted game pairs, {} queued game pairs ({} queued games)",
                next_test_id,
                num_game_pairs,
                total_pairs_dispatched,
                total_pairs_dispatched * 2
            );


            let openings =
                read_epd_file(
                    &data_path_str(
                        "hidden/8moves_v3.epd"
                    )
                )
                .expect(
                    "EPD file missing"
                );


            let mut rng =
                rand::thread_rng();


            for _ in 0..total_pairs_dispatched {

                let fen =
                    openings
                        .choose(&mut rng)
                        .unwrap()
                        .to_string();


                // -------------------------------------------------------
                // ONE queue item = ONE paired test.
                //
                // The worker itself plays both games before accepting
                // another pair.
                // -------------------------------------------------------

                job_send
                    .send(GamePair {
                        test_id:
                            next_test_id,
                        fen,
                    })
                    .unwrap();
            }


            pending_jobs =
                false;

            active_test =
                true;

            next_test_id += 1;
        }


        recv_msg.clear();
    }
}
