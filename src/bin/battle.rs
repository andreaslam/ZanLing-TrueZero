use cozy_chess::{Board, GameStatus, Move};
use crossbeam::thread;
use lru::LruCache;
use std::{env, io, num::NonZeroUsize, panic, str::FromStr};
use tokio::runtime::Runtime;
use tzrust::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    dataformat::ZeroEvaluationAbs,
    executor::{
        executor_static,
        Message::{self, StopServer},
        Packet,
    },
    mcts::get_move,
    mcts_trainer::{EvalMode, TypeRequest::NonTrainerSearch},
    settings::SearchSettings,
};

fn get_input(bs: &BoardStack) -> Move {
    let mut mv;
    // get all legal moves and print them

    let mut move_list = Vec::new();

    bs.board().generate_moves(|moves| {
        // Unpack dense move set into move list
        move_list.extend(moves);
        false
    });

    let mut result = String::new();

    for (index, item) in move_list.iter().enumerate() {
        result.push_str(&format!("{}", item));

        if index < move_list.len() - 1 {
            result.push_str(", ");
        }
    }
    loop {
        let mut input = String::new();
        println!("Legal moves available: {}", result);
        println!("Enter move:");
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");
        let tb = bs.clone(); // test legality, play move
        let input = input.replace("\r\n", "");
        mv = Move::from_str(&input);
        match mv {
            Ok(valid_move) => {
                let result = tb.board().clone().try_play(valid_move);
                match result {
                    Ok(_) => break,
                    Err(_) => continue,
                }
            }
            Err(_) => continue,
        }
    }

    mv.unwrap()
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    panic::set_hook(Box::new(|panic_info| {
        // print panic information
        eprintln!("Panic occurred: {:?}", panic_info);
        // exit the program immediately
        std::process::exit(1);
    }));

    let board = Board::default();
    let mut bs = BoardStack::new(board);
    let mut input = String::new();

    // set up executor and sender pairs
    let settings: SearchSettings = SearchSettings {
        fpu: 0.0,
        wdl: EvalMode::Value,
        moves_left: None,
        c_puct: 2.0,
        max_nodes: Some(400),
        alpha: 0.0,
        eps: 0.0,
        search_type: NonTrainerSearch,
        pst: 0.0,
    };
    let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Packet>(1);
    let (ctrl_sender, ctrl_recv) = flume::bounded::<Message>(1);
    let _ = thread::scope(|s| {
        s.builder()
            .name("executor".to_string())
            .spawn(move |_| {
                executor_static("nets/tz_5524.pt".to_string(), tensor_exe_recv, ctrl_recv, 1)
            })
            .unwrap();

        println!("Player (p) or bot (b) first: ");
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");
        let input = input.replace("\r\n", "");
        let mut cache: LruCache<CacheEntryKey, ZeroEvaluationAbs> =
            LruCache::new(NonZeroUsize::new(10000).unwrap());
        if input == *"p" {
            while bs.status() == GameStatus::Ongoing {
                let mv = get_input(&bs);
                bs.play(mv);
                let rt = Runtime::new().unwrap();
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
                println!("{:#}", mv);
                bs.play(mv);
            }
        } else if input == *"b" {
            // bot plays first
            while bs.status() == GameStatus::Ongoing {
                let rt = Runtime::new().unwrap();
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
                let mv = get_input(&bs);
                bs.play(mv);
            }
        }
        ctrl_sender.send(StopServer).unwrap();
    });
}
