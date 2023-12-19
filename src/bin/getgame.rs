use cozy_chess::{Board, GameStatus};
use crossbeam::thread;
use flume::Sender;
use std::{env, panic};
use tz_rust::{
    boardmanager::BoardStack,
    executor::{
        executor_static,
        Message::{self, StopServer},
        Packet,
    },
    mcts_trainer::get_move,
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

    let board = Board::default();
    let mut bs = BoardStack::new(board);

    // set up executor and sender pairs

    let (tensor_exe_send_0, tensor_exe_recv_0) = flume::bounded::<Packet>(1);
    let (tensor_exe_send_1, tensor_exe_recv_1) = flume::bounded::<Packet>(1);
    let (ctrl_sender, ctrl_recv) = flume::bounded::<Message>(1);
    let _ = thread::scope(|s| {
        let tensor_exe_recv_clone_0 = tensor_exe_recv_0.clone();
        let ctrl_recv_clone = ctrl_recv.clone();
        s.builder()
            .name("executor_net_0".to_string())
            .spawn(move |_| {
                executor_static(
                    "./nets/tz_0.pt".to_string(),
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
                    "./nets/tz_0.pt".to_string(),
                    // "chess_16x128_gen3634.pt".to_string(),
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
            let (mv, _, _, _, _) = get_move(bs.clone(), tensor_exe_send.clone());
            bs.play(mv);
            println!("{:#}", mv);

            counter += 1;
        }
        ctrl_sender.send(StopServer()).unwrap();
    });
}
