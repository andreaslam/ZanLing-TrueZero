use crate::{
    boardmanager::BoardStack,
    cache::{CacheEntryKey, CacheEntryValue},
    decoder::{convert_board, eval_state},
    executor::{executor_static, Message, Packet},
    mcts::get_move,
    mcts_trainer::{EvalMode, Net, TypeRequest::UCISearch},
    settings::SearchSettings,
};
use cozy_chess::{Board, Color, Move, Piece, Square};
use crossbeam::thread;
use flume::{Receiver, Sender};
use lru::LruCache;
use std::{cmp::max, io, num::NonZeroUsize, panic, process, str::FromStr};
use tokio::runtime::Runtime;
const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub fn eval_in_cp(eval: f32) -> f32 {
    let cps = if eval > 0.5 {
        18. * (eval - 0.5) + 1.
    } else if eval < -0.5 {
        18. * (eval + 0.5) - 1.
    } else {
        2. * eval
    };
    cps
}

fn mb_to_items(mb: usize) -> usize {
    let key_size = std::mem::size_of::<CacheEntryKey>();
    let val_size = std::mem::size_of::<CacheEntryValue>();
    (mb * 1000000) / (key_size + val_size)
}

pub fn run_uci(net_path: &str) {
    panic::set_hook(Box::new(|panic_info| {
        eprintln!("Panic occurred: {:?}", panic_info);
        std::process::exit(1);
    }));

    let board = Board::default();
    let mut bs = BoardStack::new(board);
    let mut stack = Vec::new();
    let mut threads = 1;

    let mut stored_message: Option<String> = None;
    let net = Net::new(net_path);
    let (ctrl_sender, ctrl_recv) = flume::bounded::<Message>(1);

    // Spawn a thread to listen for stop and quit commands
    let (cmd_sender, cmd_receiver) = flume::unbounded();
    std::thread::spawn(move || loop {
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let commands = input.split_whitespace().collect::<Vec<_>>();

        match *commands.first().unwrap_or(&"oops") {
            "stop" => {
                let _ = cmd_sender.send("stop");
                // debug_print(&format!("Debug: Sent 'stop' command"));
            }
            "quit" => {
                process::exit(0);
            }
            _ => {}
        }
    });

    loop {
        let input = if let Some(msg) = stored_message {
            msg.clone()
        } else {
            let mut input = String::new();
            let bytes_read = io::stdin().read_line(&mut input).unwrap();
            if bytes_read == 0 {
                break;
            }
            input
        };

        stored_message = None;
        let commands = input.split_whitespace().collect::<Vec<_>>();

        match *commands.first().unwrap_or(&"oops") {
            "uci" => {
                // debug_print(&format!("Debug: Received 'uci' command"));
                preamble();
            }
            "isready" => {
                // debug_print(&format!("Debug: Received 'isready' command"));
                println!("readyok");
            }
            "ucinewgame" => {}
            "go" => {
                // debug_print(&format!("Debug: Received 'go' command"));
                handle_go(
                    &commands,
                    &bs,
                    &net_path,
                    &ctrl_recv,
                    &cmd_receiver,
                    &ctrl_sender,
                );
            }
            "position" => {
                // debug_print(&format!("Debug: Received 'position' command"));
                set_position(commands, &mut bs, &mut stack);
            }
            "eval" => {
                // debug_print(&format!("Debug: Received 'eval' command"));
                let (value, _) = eval_state(convert_board(&bs), &net).unwrap();
                let value = value.squeeze();
                let value_raw: Vec<f32> = Vec::try_from(value).expect("Error");
                let value: f32 = value_raw[0].tanh();
                let cps = eval_in_cp(value);
                println!(
                    "eval: {}",
                    (cps * 100.).round().max(-1000.).min(1000.) as i64,
                );
            }
            _ => {}
        }
    }
}

fn preamble() {
    println!("id name TrueZero-latest {}", env!("CARGO_PKG_VERSION"));
    println!("id author Andreas Lam");
    println!("uciok");
}

fn check_castling_move(bs: &BoardStack, mut mv: Move) -> Move {
    // debug_print(&format!("Debug: Checking castling move"));
    if bs.board().piece_on(mv.from) == Some(Piece::King) {
        mv.to = match (mv.from, mv.to) {
            (Square::E1, Square::G1) => Square::H1,
            (Square::E8, Square::G8) => Square::H8,
            (Square::E1, Square::C1) => Square::A1,
            (Square::E8, Square::C8) => Square::A8,
            _ => mv.to,
        };
    }
    mv
}

fn set_position(commands: Vec<&str>, bs: &mut BoardStack, stack: &mut Vec<u64>) {
    // debug_print(&format!("Debug: Setting position"));
    let mut fen = String::new();
    let mut move_list = Vec::new();
    let mut moves = false;

    for cmd in commands {
        match cmd {
            "position" | "startpos" | "fen" => {}
            "moves" => moves = true,
            _ => {
                if moves {
                    move_list.push(cmd)
                } else {
                    fen.push_str(&format!("{cmd} "))
                }
            }
        }
    }
    let fenstr = if fen.is_empty() {
        STARTPOS
    } else {
        &fen.trim()
    };
    let board = Board::from_fen(fenstr, false).unwrap_or(Board::default());
    *bs = BoardStack::new(board.clone());
    stack.clear();

    for m in move_list {
        stack.push(bs.board().hash());
        let mut legal_moves = Vec::new();
        bs.board().generate_moves(|moves| {
            legal_moves.extend(moves);
            false
        });
        for mov in legal_moves.iter() {
            let mv = Move::from_str(&m).unwrap();
            let mv = check_castling_move(&bs, mv);
            if mv == *mov {
                bs.play(*mov);
            }
        }
    }
}

pub fn handle_go(
    commands: &[&str],
    bs: &BoardStack,
    net_path: &str,
    ctrl_recv: &Receiver<Message>,
    cmd_receiver: &Receiver<&str>,
    ctrl_sender: &Sender<Message>,
) {
    // debug_print(&format!("Debug: Handling 'go' command"));
    let mut nodes = 1600;
    let mut max_time = None;
    let mut max_depth = 256;

    let mut times = [None; 2];
    let mut incs = [None; 2];
    let mut movestogo = 30;

    let mut mode = "";

    for cmd in commands {
        match *cmd {
            "nodes" => mode = "nodes",
            "movetime" => mode = "mov",
            "movetime" => mode = "movetime",
            "depth" => mode = "depth",
            "wtime" => mode = "wtime",
            "btime" => mode = "btime",
            "winc" => mode = "winc",
            "binc" => mode = "binc",
            "movestogo" => mode = "movestogo",
            _ => match mode {
                "nodes" => {
                    nodes = cmd.parse().unwrap_or(nodes);
                    // debug_print(&format!("Debug: Set 'nodes' to {}", nodes));
                }
                "movetime" => {
                    max_time = cmd.parse().ok();
                    // debug_print(&format!("Debug: Set 'max_time' to {:?}", max_time));
                }
                "depth" => {
                    max_depth = cmd.parse().unwrap_or(max_depth);
                    // debug_print(&format!("Debug: Set 'max_depth' to {}", max_depth));
                }
                "wtime" => {
                    times[0] = Some(cmd.parse().unwrap_or(0));
                    // debug_print(&format!("Debug: Set 'wtime' to {:?}", times[0]));
                }
                "btime" => {
                    times[1] = Some(cmd.parse().unwrap_or(0));
                    // debug_print(&format!("Debug: Set 'btime' to {:?}", times[1]));
                }
                "winc" => {
                    incs[0] = Some(cmd.parse().unwrap_or(0));
                    // debug_print(&format!("Debug: Set 'winc' to {:?}", incs[0]));
                }
                "binc" => {
                    incs[1] = Some(cmd.parse().unwrap_or(0));
                    // debug_print(&format!("Debug: Set 'binc' to {:?}", incs[1]));
                }
                "movestogo" => {
                    movestogo = cmd.parse().unwrap_or(30);
                    // debug_print(&format!("Debug: Set 'movestogo' to {}", movestogo));
                }
                _ => mode = "none",
            },
        }
    }

    let mut time: Option<u128> = None;

    let stm = bs.board().side_to_move();
    let stm_num = match stm {
        Color::White => Some(0),
        Color::Black => Some(1),
    };
    if let Some(t) = times[stm_num.unwrap()] {
        let mut base = t / movestogo.max(1);

        if let Some(i) = incs[stm_num.unwrap()] {
            base += i * 3 / 4;
        }
        time = Some(base.try_into().unwrap());
        nodes = time.unwrap() as u128 / 120;
    }
    nodes = max(1, nodes);
    if let Some(max) = max_time {
        time = Some(time.unwrap_or(u128::MAX).min(max));
        nodes = time.unwrap() as u128 / 120;
    }

    if let Some(t) = time.as_mut() {
        *t = t.saturating_sub(5);
    }
    let settings: SearchSettings = SearchSettings {
        fpu: 0.0,
        wdl: EvalMode::Value,
        moves_left: None,
        c_puct: 2.0,
        max_nodes: nodes,
        alpha: 0.0,
        eps: 0.0,
        search_type: UCISearch,
        pst: 1.5,
    };
    let rt = Runtime::new().unwrap();
    let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Packet>(1);
    thread::scope(|s| {
        s.builder()
            .name("executor".to_string())
            .spawn(move |_| {
                executor_static(net_path.to_string(), tensor_exe_recv, ctrl_recv.clone(), 1)
            })
            .unwrap();
        let mut cache: LruCache<CacheEntryKey, CacheEntryValue> =
            LruCache::new(NonZeroUsize::new(1000000).unwrap());
        let (best_move, _, _, _, _) = rt.block_on(async {
            get_move(
                bs.clone(),
                tensor_exe_send.clone(),
                settings.clone(),
                Some(cmd_receiver.clone()),
                &mut cache,
            )
            .await
        });

        println!("bestmove {:#}", best_move);
        let _ = ctrl_sender.send(Message::StopServer);
        // debug_print(&format!("sent termination message"));
    })
    .unwrap();
    // debug_print(&format!("function done"));
}
