use cozy_chess::Board;
use crossbeam::thread;
use lru::LruCache;
use std::{env, num::NonZeroUsize, panic, process, time::Instant};
use tokio::runtime::Runtime;
use tzrust::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    dataformat::ZeroEvaluationAbs,
    debug_print,
    executor::{
        executor_static,
        Message::{self, StopServer},
        Packet,
    },
    mcts::get_move,
    mcts_trainer::{EvalMode, TypeRequest::NonTrainerSearch},
    settings::{CPUCTSettings, FPUSettings, MovesLeftSettings, PSTSettings, SearchSettings},
};
fn main() {
    // test MCTS move outputs
    panic::set_hook(Box::new(|panic_info| {
        // print panic information
        eprintln!("Panic occurred: {:?}", panic_info);
        // exit the program immediately
        std::process::exit(2);
    }));
    env::set_var("RUST_BACKTRACE", "1");
    let board = Board::default();
    // let board = Board::from_fen(
    //     "6rk/p3p2p/1p2Pp2/2p2P2/2P1nBr1/1P6/P6P/3R1R1K b - - 0 1",
    //     false,
    // )
    // .unwrap(); // black M2
    // let board = Board::from_fen("8/7k/5KR1/8/8/8/8/8 w - - 0 1", false).unwrap(); // white M2
    // let board = Board::from_fen("5r1k/6pp/8/8/1Pq2r2/P3Q2P/6P1/2R1R2K b - - 4 34", false).unwrap();
    let mut move_list = Vec::new();
    board.generate_moves(|moves| {
        // Unpack dense move set into move list
        move_list.extend(moves);
        false
    });

    let _total_moves = move_list.len();
    // set up executor and sender pairs

    let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Packet>(1);
    let (ctrl_sender, ctrl_recv) = flume::bounded::<Message>(1);
    thread::scope(|s| {
        s.builder()
            .name("executor".to_string())
            .spawn(move |_| {
                executor_static(
                    // r"nets/tz_4924.pt".to_string(),
                    // r"tz_6515.pt".to_string(),
                    r"chess_16x128_gen3634.pt".to_string(),
                    tensor_exe_recv,
                    ctrl_recv,
                    1,
                )
            })
            .unwrap();

        debug_print!("{}", &format!("Number of legal moves: {}", _total_moves));
        let bs = BoardStack::new(board);
        let sw = Instant::now();
        let m_settings = MovesLeftSettings {
            moves_left_weight: 0.03,
            moves_left_clip: 20.0,
            moves_left_sharpness: 0.5,
        };
        let max_nodes = 1000;
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
            max_nodes: Some(max_nodes),
            alpha: 0.03,
            eps: 0.25,
            search_type: NonTrainerSearch,
            pst: PSTSettings {
                root_pst: 1.75,
                children_pst: 1.5,
            },
            batch_size: 1,
        };
        let rt = Runtime::new().unwrap();
        let mut cache: LruCache<CacheEntryKey, ZeroEvaluationAbs> =
            LruCache::new(NonZeroUsize::new(100000).unwrap());
        let (best_move, nn_data, _, _, _) = rt.block_on(async {
            get_move(bs, tensor_exe_send.clone(), settings, None, &mut cache).await
        });
        for (mv, score) in move_list.iter().zip(nn_data.policy.iter()) {
            println!("{:#}: {}%", mv, score);
        }
        println!("{:#}", best_move);
        println!("{:?}", nn_data);
        println!("Elapsed time: {}ms", sw.elapsed().as_nanos() as f32 / 1e6);
        let nps = settings.max_nodes.unwrap() as f32 / (sw.elapsed().as_nanos() as f32 / 1e9);
        println!("{}", &format!("Nodes per second: {}nps", nps));
        ctrl_sender.send(StopServer).unwrap();
        process::exit(0);
    })
    .unwrap();
}
