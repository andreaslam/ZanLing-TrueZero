use cozy_chess::{Board, Color, GameStatus};
use crossbeam::thread;
use flume::Sender;
use lru::LruCache;
use std::{env, num::NonZeroUsize, panic};
use tokio::runtime::Runtime;
use tzrust::{
    boardmanager::BoardStack,
    executor::{
        executor_static,
        Message::{self, StopServer},
        Packet,
    },
    mcts::get_move,
    mcts_trainer::{EvalMode, TypeRequest::TrainerSearch},
    settings::{CPUCTSettings, FPUSettings, MovesLeftSettings, PSTSettings, SearchSettings},
};

fn main() {
    // test game with MCTS
    env::set_var("RUST_BACKTRACE", "1");
    panic::set_hook(Box::new(|panic_info| {
        // print panic information
        eprintln!("Panic occurred: {:?}", panic_info);
        // exit the program immediately

        std::process::exit(1);
    }));
    let (tensor_exe_send_1, tensor_exe_recv_1) = flume::bounded::<Packet>(1);
    let (tensor_exe_send_0, tensor_exe_recv_0) = flume::bounded::<Packet>(1);

    let mut scores = (0, 0, 0); // Win, Draw, Loss (White POV)

    let mut games_count = 0;

    let target_games = 10;

    // set up executor and sender pairs

    let (ctrl_sender, ctrl_recv) = flume::bounded::<Message>(1);
    let m_settings = MovesLeftSettings {
        moves_left_weight: 0.03,
        moves_left_clip: 20.0,
        moves_left_sharpness: 0.5,
    };
    let settings: SearchSettings = SearchSettings {
        fpu: FPUSettings {
            root_fpu: 0.1,
            children_fpu: 0.1,
        },

        wdl: EvalMode::Wdl,
        moves_left: Some(m_settings),
        c_puct: CPUCTSettings {
            root_c_puct: 2.0,
            children_c_puct: 2.0,
        },

        max_nodes: Some(10),
        alpha: 0.03,
        eps: 0.25,
        search_type: TrainerSearch(None),
        pst: PSTSettings {
            root_pst: 1.75,
            children_pst: 1.5,
        },
        batch_size: 1,
    };
    thread::scope(|s| {
        while games_count < target_games {
            let board = Board::default();
            let mut bs = BoardStack::new(board);
            let tensor_exe_recv_clone_0 = tensor_exe_recv_0.clone();
            let ctrl_recv_clone = ctrl_recv.clone();
            s.builder()
                .name("executor_net_0".to_string())
                .spawn(move |_| {
                    executor_static(
                        "./nets/tz_1.pt".to_string(),
                        // "chess_16x128_gen3634.pt".to_string(),
                        tensor_exe_recv_clone_0,
                        ctrl_recv_clone,
                        1,
                    )
                })
                .unwrap();
            let tensor_exe_recv_clone_1 = tensor_exe_recv_1.clone();
            let ctrl_recv_clone = ctrl_recv.clone();
            s.builder()
                .name("executor_net_1".to_string())
                .spawn(move |_| {
                    executor_static(
                        "chess_16x128_gen3634.pt".to_string(),
                        // "./nets/tz_4483.pt".to_string(),
                        tensor_exe_recv_clone_1,
                        ctrl_recv_clone,
                        1,
                    )
                })
                .unwrap();
            let mut counter = 0;
            let mut tensor_exe_send: Sender<Packet>;
            while bs.status() == GameStatus::Ongoing {
                if counter % 2 == 0 {
                    tensor_exe_send = tensor_exe_send_0.clone();
                } else {
                    tensor_exe_send = tensor_exe_send_1.clone();
                }
                let rt = Runtime::new().unwrap();
                let mut cache = LruCache::new(NonZeroUsize::new(1000).unwrap());
                let (mv, _, _, _, _) = rt.block_on(async {
                    get_move(
                        bs.clone(),
                        tensor_exe_send.clone(),
                        settings,
                        None,
                        &mut cache,
                    )
                    .await
                });
                bs.play(mv);
                println!("{:#}", mv);

                counter += 1;
            }
            let result = match bs.status() {
                GameStatus::Ongoing => panic!("game not over yet!"),
                GameStatus::Won => Some(!bs.board().side_to_move()),
                GameStatus::Drawn => None,
            };

            match result {
                Some(winner) => match winner {
                    Color::White => {
                        println!("1-0");
                        scores.0 += 1;
                    }
                    Color::Black => {
                        println!("0-1");
                        scores.2 += 1;
                    }
                },
                None => {
                    println!("1/2-1/2");
                    scores.1 += 1;
                }
            }

            games_count += 1;
        }
        println!("{:?}", scores);
        ctrl_sender.send(StopServer).unwrap();
    })
    .unwrap();
}
