use cozy_chess::{Board, GameStatus};
use crossbeam::thread;
use flume::{Receiver, Sender};
use futures::executor::ThreadPool;
use lru::LruCache;
use rand::seq::SliceRandom;
use std::{
    env,
    fs::File,
    io::{self, BufRead},
    num::NonZeroUsize,
    panic, process,
};
use tzrust::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    dataformat::ZeroEvaluationAbs,
    debug_print,
    elo::elo_wld,
    executor::{executor_static, Message, Packet},
    mcts::get_move,
    mcts_trainer::{EvalMode, TypeRequest::NonTrainerSearch},
    selfplay::CollectorMessage,
    settings::{CPUCTSettings, FPUSettings, MovesLeftSettings, SearchSettings},
};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    panic::set_hook(Box::new(|panic_info| {
        // print panic information
        eprintln!("Panic occurred: {:?}", panic_info);
        std::process::exit(1);
    }));

    let (game_sender, game_receiver) = flume::bounded::<CollectorMessage>(1);
    let num_games = 1000000;
    let num_threads = 2048;
    let engine_0: String = "nets/tz_304.pt".to_string(); // new engine
                                                         // let engine_0: String = "tz_6515.pt".to_string(); // new engine
                                                         //  let engine_1: String = "chess_16x128_gen3634.pt".to_string(); // old engine
    let engine_1: String = "tz_6515.pt".to_string();
    // let engine_1: String = "nets/tz_296.pt".to_string(); // new engine
    // let engine_1: String = "nets/tz_151.pt".to_string();
    let num_executors = 2; // always be 2, 2 players, one each (one for each neural net)
    let (ctrl_sender, ctrl_recv) = flume::bounded::<Message>(1);
    let pool = ThreadPool::builder().pool_size(6).create().unwrap();
    thread::scope(|s| {
        let mut vec_communicate_exe_send: Vec<Sender<String>> = Vec::new();
        let mut vec_communicate_exe_recv: Vec<Receiver<String>> = Vec::new();
        assert!(num_executors == 2);
        for _ in 0..num_executors {
            let (communicate_exe_send, communicate_exe_recv) = flume::bounded::<String>(1);
            vec_communicate_exe_send.push(communicate_exe_send);
            vec_communicate_exe_recv.push(communicate_exe_recv);
        }

        let (tensor_exe_send_0, tensor_exe_recv_0) = flume::bounded::<Packet>(num_threads);
        let (tensor_exe_send_1, tensor_exe_recv_1) = flume::bounded::<Packet>(num_threads);

        for n in 0..num_threads {
            let sender_clone = game_sender.clone();
            let tensor_exe_send_clone_0 = tensor_exe_send_0.clone();
            let tensor_exe_send_clone_1 = tensor_exe_send_1.clone();
            let fut_generator = async move {
                generator_main(
                    sender_clone,
                    tensor_exe_send_clone_0.clone(),
                    tensor_exe_send_clone_1.clone(),
                    n,
                )
                .await;
            };
            pool.spawn_ok(fut_generator);
        }

        s.builder()
            .name("collector".to_string())
            .spawn(|_| {
                debug_print!("Spawning collector thread");
                collector_main(
                    &game_receiver,
                    num_games,
                    ctrl_sender,
                    engine_0.clone(),
                    engine_1.clone(),
                )
            })
            .unwrap();

        let ctrl_recv_0 = ctrl_recv.clone();
        let ctrl_recv_1 = ctrl_recv.clone();

        let engine_0_clone = engine_0.clone();
        let engine_1_clone = engine_1.clone();
        s.builder()
            .name("executor_0".to_string())
            .spawn(move |_| {
                debug_print!("Spawning executor_0 thread");
                executor_static(
                    engine_0_clone,
                    tensor_exe_recv_0,
                    ctrl_recv_0,
                    num_threads / num_executors,
                )
            })
            .unwrap();

        s.builder()
            .name("executor_1".to_string())
            .spawn(move |_| {
                debug_print!("Spawning executor_1 thread");
                executor_static(
                    engine_1_clone,
                    tensor_exe_recv_1,
                    ctrl_recv_1,
                    num_threads / num_executors,
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
    let positions: Vec<String> = reader.lines().filter_map(|line| line.ok()).collect();
    Ok(positions)
}

async fn generator_main(
    sender_collector: Sender<CollectorMessage>,
    tensor_exe_send_0: Sender<Packet>,
    tensor_exe_send_1: Sender<Packet>,
    generator_id: usize,
) {
    let m_settings = MovesLeftSettings {
        moves_left_weight: 0.03,
        moves_left_clip: 20.0,
        moves_left_sharpness: 0.5,
    };

    let settings: SearchSettings = SearchSettings {
        fpu: FPUSettings {
            root_fpu: Some(0.1),
            children_fpu: Some(0.1),
        },
        wdl: EvalMode::Wdl,
        moves_left: Some(m_settings),
        c_puct: CPUCTSettings {
            root_c_puct: Some(2.0),
            children_c_puct: Some(2.0),
        },
        max_nodes: Some(400),
        alpha: 0.03,
        eps: 0.25,
        search_type: NonTrainerSearch,
        pst: 1.3,
        batch_size: 1,
    };
    let _thread_name = format!("sprt-generator-{}", generator_id);
    debug_print!("{} Generator settings initialized", _thread_name);

    let openings = read_epd_file("./8moves_v3.epd").unwrap();
    let engines = [tensor_exe_send_0.clone(), tensor_exe_send_1.clone()];
    let mut swap_count = 0;
    let mut fen = openings.choose(&mut rand::thread_rng()).unwrap();
    loop {
        let mut moves_list: Vec<String> = Vec::new();
        if swap_count % 2 == 0 {
            fen = openings.choose(&mut rand::thread_rng()).unwrap();
        }
        let board = Board::from_fen(fen, false).unwrap();
        let mut bs = BoardStack::new(board);
        let mut move_counter = swap_count % 2;
        let cache_0: LruCache<CacheEntryKey, ZeroEvaluationAbs> =
            LruCache::new(NonZeroUsize::new(settings.max_nodes.unwrap() as usize).unwrap());
        let cache_1: LruCache<CacheEntryKey, ZeroEvaluationAbs> =
            LruCache::new(NonZeroUsize::new(settings.max_nodes.unwrap() as usize).unwrap());

        let mut caches = [cache_0, cache_1];
        while bs.status() == GameStatus::Ongoing {
            let engine = &engines[move_counter % 2];
            let cache = &mut caches[move_counter % 2];
            let (mv, _, _, _, _) =
                get_move(bs.clone(), engine.clone(), settings, None, cache).await;
            bs.play(mv);
            moves_list.push(format!("{:#}", mv));
            move_counter += 1;
        }
        let outcome: Option<bool> = match bs.status() {
            GameStatus::Drawn => None,
            GameStatus::Won => Some((move_counter - 1) % 2 == 0),
            GameStatus::Ongoing => panic!("Game is still ongoing!"),
        };
        let moves_list_str = moves_list.join(" ");
        println!(
            "{}",
            &format!(
                "first move engine_{} opening {}, moves {}",
                swap_count % 2,
                fen,
                moves_list_str
            )
        );
        swap_count += 1;
        sender_collector
            .send(CollectorMessage::TestingResult(outcome))
            .unwrap();
    }
}

fn collector_main(
    receiver: &Receiver<CollectorMessage>,
    games: usize,
    ctrl_sender: Sender<Message>,
    engine_0_path: String,
    engine_1_path: String,
) {
    let mut results = (0, 0, 0); // (w,l,d) in the perspective of engine_0
    let mut counter = 0;

    debug_print!("Collector main started");

    loop {
        let msg = receiver.recv().unwrap();
        match msg {
            CollectorMessage::FinishedGame(_)
            | CollectorMessage::GeneratorStatistics(_)
            | CollectorMessage::ExecutorStatistics(_)
            | CollectorMessage::GameResult(_) => {
                panic!("not possible! this is to test engine changes");
            }
            CollectorMessage::TestingResult(result) => {
                if counter == games {
                    let (elo_min, elo_actual, elo_max) = elo_wld(results.0, results.1, results.2);
                    println!("===");
                    println!("{} vs {}", engine_0_path, engine_1_path);
                    println!("{} games", counter);
                    println!("w: {}, l: {}, d: {}", results.0, results.1, results.2);
                    println!(
                        "elo_min={}, elo_actual={}, elo_max={}, +/- {}",
                        elo_min,
                        elo_actual,
                        elo_max,
                        elo_max - elo_min
                    );
                    println!("===");
                    ctrl_sender.send(Message::StopServer).unwrap();
                    process::exit(0)
                } else {
                    match result {
                        Some(winner) => match winner {
                            true => results.0 += 1,
                            false => results.1 += 1,
                        },
                        None => results.2 += 1,
                    }
                    let (elo_min, elo_actual, elo_max) = elo_wld(results.0, results.1, results.2);
                    println!("w: {}, l: {}, d: {}", results.0, results.1, results.2);
                    println!(
                        "elo_min={}, elo_actual={}, elo_max={}, +/- {}",
                        elo_min,
                        elo_actual,
                        elo_max,
                        elo_max - elo_min
                    );
                    counter += 1;
                }
            }
        }
    }
}
