use cozy_chess::{Board, Color, GameStatus, Move};
use crossbeam::thread;
use flume::{Receiver, Sender};
use rand::seq::SliceRandom;
use std::{
    env,
    fs::File,
    io::{self, BufRead},
    panic, process,
};
use tz_rust::{
    boardmanager::BoardStack,
    elo::elo_wld,
    executor::{executor_static, Message, Packet},
    mcts::get_move,
    mcts_trainer::TypeRequest::NonTrainerSearch,
    selfplay::CollectorMessage,
    settings::SearchSettings,
};
fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    // panic::set_hook(Box::new(|panic_info| {
    //     // print panic information
    //     eprintln!("Panic occurred: {:?}", panic_info);
    //     // exit the program immediately
    //     std::process::exit(1);
    // }));

    let (game_sender, game_receiver) = flume::bounded::<CollectorMessage>(1);
    let num_games = 10000; // number of games to generate
    let num_threads = 128;
    let engine_0: String = "./nets/tz_5780.pt".to_string(); // new engine
                                                            // let engine_1: String = "./nets/tz_5128.pt".to_string(); // old engine
    let engine_1: String = "./chess_16x128_gen3634.pt".to_string(); // old engine
    let engine_0_clone = engine_0.clone();
    let engine_1_clone = engine_1.clone();
    let num_executors = 2; // always be 2, 2 players, one each (one for each neural net)
    let (ctrl_sender, ctrl_recv) = flume::bounded::<Message>(1);

    thread::scope(|s| {
        // let mut selfplay_masters: Vec<DataGen> = Vec::new();
        // commander

        let mut vec_communicate_exe_send: Vec<Sender<String>> = Vec::new();
        let mut vec_communicate_exe_recv: Vec<Receiver<String>> = Vec::new();

        assert!(num_executors == 2);

        for _ in 0..num_executors {
            let (communicate_exe_send, communicate_exe_recv) = flume::bounded::<String>(1);
            vec_communicate_exe_send.push(communicate_exe_send);
            vec_communicate_exe_recv.push(communicate_exe_recv);
        }

        // send-recv pair between commander and collector

        // selfplay threads
        let (tensor_exe_send_0, tensor_exe_recv_0) = flume::bounded::<Packet>(1); // mcts to executor
        let (tensor_exe_send_1, tensor_exe_recv_1) = flume::bounded::<Packet>(1); // mcts to executor
        for n in 0..num_threads {
            // // executor
            // sender-receiver pair to communicate for each thread instance to the executor
            let sender_clone = game_sender.clone();
            // let mut selfplay_master = DataGen { iterations: 1 };
            let tensor_exe_send_clone_0 = tensor_exe_send_0.clone();
            let tensor_exe_send_clone_1 = tensor_exe_send_1.clone();
            // let nps_sender = game_sender.clone();
            s.builder()
                .name(format!("generator_{}", n.to_string()))
                .spawn(move |_| {
                    generator_main(
                        &sender_clone,
                        tensor_exe_send_clone_0,
                        tensor_exe_send_clone_1,
                    )
                })
                .unwrap();

            // selfplay_masters.push(selfplay_master.clone());
        }
        // collector

        s.builder()
            .name("collector".to_string())
            .spawn(|_| {
                collector_main(
                    &game_receiver,
                    num_games,
                    ctrl_sender,
                    engine_0_clone,
                    engine_1_clone,
                )
            })
            .unwrap();
        // executor
        // send/recv pair between executor and commander
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
        wdl: None,
        moves_left: None,
        c_puct: 0.0,
        max_nodes: 2,
        alpha: 0.0,
        eps: 0.0,
        search_type: NonTrainerSearch,
        pst: 0.0
    };

    let openings = read_epd_file("./8moves_v3.epd").unwrap();
    let engines = vec![tensor_exe_send_0.clone(), tensor_exe_send_1.clone()];
    loop {
        let mut swap_count = 0; // so that each engine can play same opening as black and white

        let fen = openings.choose(&mut rand::thread_rng()).unwrap();
        for (engine_idx, engine) in engines.iter().enumerate() {
            let board = Board::from_fen(fen, false).unwrap();
            // println!("starting fen: {}", fen);
            let mut bs = BoardStack::new(board);

            let mut counter = 0;
            while bs.status() == GameStatus::Ongoing {
                let mv: Move;
                if counter % 2 == 0 {
                    // white
                    (mv, _, _, _, _) = get_move(bs.clone(), engine.clone(), settings.clone());
                } else {
                    // swap the engine for black
                    let opponent_engine = engines[(engine_idx + 1) % engines.len()].clone();
                    (mv, _, _, _, _) = get_move(bs.clone(), opponent_engine, settings.clone());
                }
                bs.play(mv);
                counter += 1;
            }

            let outcome: Option<Color> = match bs.status() {
                GameStatus::Drawn => None,
                GameStatus::Won => Some(!bs.board().side_to_move()),
                GameStatus::Ongoing => panic!("Game is still ongoing!"),
            };
            // handle outcome based on engine and move colour (engine_0 POV)
            let outcome: Option<Color> = match outcome {
                Some(colour) => {
                    if swap_count == 0 {
                        Some(colour)
                    } else if swap_count == 1 {
                        Some(!colour)
                    } else {
                        unreachable!()
                    }
                }
                None => None,
            };

            swap_count += 1;
            sender_collector
                .send(CollectorMessage::GameResult(outcome))
                .unwrap();
        }
    }
}

fn collector_main(
    receiver: &Receiver<CollectorMessage>,
    games: usize,
    ctrl_sender: Sender<Message>,
    engine_0_path: String,
    engine_1_path: String,
) {
    let mut results = (0, 0, 0); // win, loss, draw (engine_0 POV)
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
            CollectorMessage::GameResult(result) => {
                if counter == games {
                    // print elo stats
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
                    let _ = ctrl_sender.send(Message::StopServer());
                    process::exit(0)
                } else {
                    // print elo stats

                    match result {
                        Some(colour) => match colour {
                            Color::White => results.0 += 1,
                            Color::Black => results.1 += 1,
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
