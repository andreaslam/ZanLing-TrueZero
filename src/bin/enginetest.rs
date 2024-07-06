use cozy_chess::{Board, Color, GameStatus, Move};
use crossbeam::thread;
use flume::{Receiver, Sender};
use lru::LruCache;
use rand::seq::SliceRandom;
use std::{
    env,
    fs::File,
    io::{self, BufRead},
    num::NonZeroUsize,
    panic, process,
};
use tokio::runtime::Runtime;
use tz_rust::debug_print;
use tz_rust::{
    boardmanager::BoardStack,
    cache::{CacheEntryKey, CacheEntryValue},
    elo::elo_wld,
    executor::{executor_static, Message, Packet},
    mcts::get_move,
    mcts_trainer::{EvalMode, TypeRequest::NonTrainerSearch},
    selfplay::CollectorMessage,
    settings::SearchSettings,
    utils::TimeStampDebugger,
};
fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    panic::set_hook(Box::new(|panic_info| {
        // print panic information
        eprintln!("Panic occurred: {:?}", panic_info);
        // exit the program immediately
        std::process::exit(1);
    }));

    let (game_sender, game_receiver) = flume::bounded::<CollectorMessage>(1);
    let num_games = 1000000;
    let num_threads = 1024;
    let engine_0: String = "nets/tz_1848.pt".to_string(); // new engine
    let engine_1: String = "./chess_16x128_gen3634.pt".to_string(); // old engine
    let num_executors = 2; // always be 2, 2 players, one each (one for each neural net)
    let (ctrl_sender, ctrl_recv) = flume::bounded::<Message>(1);

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
            s.builder()
                .name(format!("generator_{}", n.to_string()))
                .spawn(move |_| {
                    generator_main(
                        &sender_clone,
                        tensor_exe_send_clone_0.clone(),
                        tensor_exe_send_clone_1.clone(),
                    )
                })
                .unwrap();
        }

        s.builder()
            .name("collector".to_string())
            .spawn(|_| {
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
    let file = File::open(file_path)?;
    let reader = io::BufReader::new(file);
    let positions: Vec<String> = reader.lines().filter_map(|line| line.ok()).collect();
    Ok(positions)
}

fn generator_main(
    sender_collector: &Sender<CollectorMessage>,
    tensor_exe_send_0: Sender<Packet>,
    tensor_exe_send_1: Sender<Packet>,
) {
    let settings: SearchSettings = SearchSettings {
        fpu: 0.0,
        wdl: EvalMode::Value,
        moves_left: None,
        c_puct: 2.0,
        max_nodes: 400,
        alpha: 0.0,
        eps: 0.0,
        search_type: NonTrainerSearch,
        pst: 0.0,
    };
    let openings = read_epd_file("./8moves_v3.epd").unwrap();
    let engines = vec![tensor_exe_send_0.clone(), tensor_exe_send_1.clone()];
    let mut swap_count = 0;
    let mut fen = openings.choose(&mut rand::thread_rng()).unwrap();
    loop {
        let mut moves_list: Vec<String> = Vec::new();
        if swap_count % 2 == 0 {
            fen = openings.choose(&mut rand::thread_rng()).unwrap();
        } else {
        }
        let board = Board::from_fen(fen, false).unwrap();
        let mut bs = BoardStack::new(board);
        let rt = Runtime::new().unwrap();
        let mut move_counter = swap_count % 2;
        let cache_0: LruCache<CacheEntryKey, CacheEntryValue> =
            LruCache::new(NonZeroUsize::new(settings.max_nodes as usize).unwrap());
        let cache_1: LruCache<CacheEntryKey, CacheEntryValue> =
            LruCache::new(NonZeroUsize::new(settings.max_nodes as usize).unwrap());
        let mut caches = vec![cache_0, cache_1];

        while bs.status() == GameStatus::Ongoing {
            let engine = &engines[move_counter % 2];
            let (mv, _, _, _, _) = rt.block_on(async {
                get_move(
                    bs.clone(),
                    engine.clone(),
                    settings.clone(),
                    None,
                    &mut caches[move_counter % 2],
                )
                .await
            });
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
        debug_print!(
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
    loop {
        let msg = receiver.recv().unwrap();
        match msg {
            CollectorMessage::FinishedGame(_) => {
                panic!("not possible! this is to test engine changes");
            }
            CollectorMessage::GeneratorStatistics(_) => {
                panic!("not possible! this is to test engine changes");
            }
            CollectorMessage::ExecutorStatistics(_) => {
                panic!("not possible! this is to test engine changes");
            }
            CollectorMessage::GameResult(_) => {
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
