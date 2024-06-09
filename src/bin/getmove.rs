use cozy_chess::Board;
use crossbeam::thread;
use std::{env, panic, time::Instant};
use tokio::runtime::Runtime;
use tz_rust::{
    boardmanager::BoardStack,
    executor::{
        executor_static,
        Message::{self, StopServer},
        Packet,
    },
    mcts::get_move,
    mcts_trainer::TypeRequest::NonTrainerSearch,
    settings::SearchSettings,
};

fn main() {
    // test MCTS move outputs
    panic::set_hook(Box::new(|panic_info| {
        // print panic information
        eprintln!("Panic occurred: {:?}", panic_info);
        // exit the program immediately
        std::process::exit(1);
    }));
    env::set_var("RUST_BACKTRACE", "1");
    let board = Board::default();
    // let board = Board::from_fen(
    //     "6rk/p3p2p/1p2Pp2/2p2P2/2P1nBr1/1P6/P6P/3R1R1K b - - 0 1",
    //     false,
    // )
    // .unwrap(); // black M2
    // let board = Board::from_fen(
    //     "r2q1k1r/3bnp2/p1n1pNp1/3pP1Qp/Pp1P4/2PB4/5PPP/R1B2RK1 w - - 1 1",
    //     false,
    // )
    // .unwrap(); // white M2
    // let board = Board::from_fen("8/6Q1/k7/8/3P4/1P2P1P1/rB3K1P/7q w - - 3 44", false).unwrap();
    let mut move_list = Vec::new();
    board.generate_moves(|moves| {
        // Unpack dense move set into move list
        move_list.extend(moves);
        false
    });

    let total_moves = move_list.len();
    // set up executor and sender pairs

    let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Packet>(1);
    let (ctrl_sender, ctrl_recv) = flume::bounded::<Message>(1);
    thread::scope(|s| {
        s.builder()
            .name("executor".to_string())
            .spawn(move |_| {
                executor_static(
                    r"nets/tz_162.pt".to_string(),
                    // r"C:\Users\andre\RemoteFolder\ZanLing-TrueZero\chess_16x128_gen3634.pt"
                    //     .to_string(),
                    tensor_exe_recv,
                    ctrl_recv,
                    1,
                )
            })
            .unwrap();

        println!("Number of legal moves: {}", total_moves);
        let bs = BoardStack::new(board);
        let sw = Instant::now();
        let settings: SearchSettings = SearchSettings {
            fpu: 0.0,
            wdl: None,
            moves_left: None,
            c_puct: 2.0,
            max_nodes: 10,
            alpha: 0.0,
            eps: 0.0,
            search_type: NonTrainerSearch,
            pst: 1.05,
        };
        let rt = Runtime::new().unwrap();
        let (best_move, nn_data, _, _, _) = rt.block_on(async {
            get_move(bs, tensor_exe_send.clone(), settings.clone(), None).await
        });
        for (mv, score) in move_list.iter().zip(nn_data.policy.iter()) {
            println!("{:#}, {}", mv, score);
        }
        println!("{:#}", best_move);
        println!("{:?}", nn_data);
        println!("Elapsed time: {}ms", sw.elapsed().as_nanos() as f32 / 1e6);
        let nps = settings.max_nodes as f32 / (sw.elapsed().as_nanos() as f32 / 1e9);
        println!("Nodes per second: {}nps", nps);
        ctrl_sender.send(StopServer).unwrap();
    })
    .unwrap();
}
