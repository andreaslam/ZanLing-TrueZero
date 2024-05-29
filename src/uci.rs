use crate::{
    boardmanager::BoardStack,
    decoder::{convert_board, eval_state},
    executor::{executor_static, Message, Packet},
    mcts::get_move,
    mcts_trainer::{Net, TypeRequest::UCISearch},
    settings::SearchSettings,
};
use cozy_chess::{Board, Color, Move, Piece, Square};
use crossbeam::thread;
use std::{
    cmp::max,
    io::{self, BufRead},
    panic, process,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
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

pub fn run_uci(net_path: &str) {
    panic::set_hook(Box::new(|_| {}));

    // initialise engine
    let board = Board::default();
    let mut bs = BoardStack::new(board);
    let mut stack = Vec::new();
    let mut threads = 1;

    let mut stored_message: Option<String> = None;
    let net = Net::new(net_path);
    // main uci loop
    loop {
        let input = if let Some(msg) = stored_message {
            msg.clone()
        } else {
            let mut input = String::new();
            let bytes_read = io::stdin().read_line(&mut input).unwrap();

            // got EOF, exit (for OpenBench).
            if bytes_read == 0 {
                break;
            }

            input
        };

        stored_message = None;

        let commands = input.split_whitespace().collect::<Vec<_>>();

        match *commands.first().unwrap_or(&"oops") {
            "uci" => preamble(),
            "isready" => println!("readyok"),
            "ucinewgame" => {}
            "go" => handle_go(&commands, &bs, &net_path),
            "position" => set_position(commands, &mut bs, &mut stack),
            "quit" => process::exit(0),
            "eval" => {
                let (value, _) = eval_state(convert_board(&bs), &net).unwrap();
                let value = value.squeeze();
                let value_raw: Vec<f32> = Vec::try_from(value).expect("Error");
                let value: f32 = value_raw[0].tanh();
                let cps = eval_in_cp(value);
                println!(
                    "eval: {}",
                    (cps * 100.).round().max(-1000.).min(1000.) as i64
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
            // Unpack dense move set into move list
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

pub fn handle_go(commands: &[&str], bs: &BoardStack, net_path: &str) {
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
            "movetime" => mode = "movetime",
            "depth" => mode = "depth",
            "wtime" => mode = "wtime",
            "btime" => mode = "btime",
            "winc" => mode = "winc",
            "binc" => mode = "binc",
            "movestogo" => mode = "movestogo",
            _ => match mode {
                "nodes" => nodes = cmd.parse().unwrap_or(nodes),
                "movetime" => max_time = cmd.parse().ok(),
                "depth" => max_depth = cmd.parse().unwrap_or(max_depth),
                "wtime" => times[0] = Some(cmd.parse().unwrap_or(0)),
                "btime" => times[1] = Some(cmd.parse().unwrap_or(0)),
                "winc" => incs[0] = Some(cmd.parse().unwrap_or(0)),
                "binc" => incs[1] = Some(cmd.parse().unwrap_or(0)),
                "movestogo" => movestogo = cmd.parse().unwrap_or(30),
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

    let settings = SearchSettings {
        fpu: 0.0,
        wdl: None,
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
    let (ctrl_sender, ctrl_recv) = flume::bounded::<Message>(1);

    // Add an atomic boolean to signal stopping
    let stop_signal = Arc::new(AtomicBool::new(false));
    let stop_signal_clone = stop_signal.clone();

    // Thread to listen for stop command

    thread::scope(|s| {
        s.builder()
            .name("stop-listener".to_string())
            .spawn(move |_| listen_stop(stop_signal_clone))
            .unwrap();
        s.builder()
            .name("executor".to_string())
            .spawn(move |_| executor_static(net_path.to_string(), tensor_exe_recv, ctrl_recv, 1))
            .unwrap();

        let (best_move, _, _, _, _) = rt.block_on(async {
            get_move(
                bs.clone(),
                tensor_exe_send.clone(),
                settings.clone(),
                Some(stop_signal.clone()),
            )
            .await
        });

        println!("bestmove {:#}", best_move);
        let _ = ctrl_sender.send(Message::StopServer);
    })
    .unwrap();
}

fn listen_stop(stop_signal_clone: Arc<AtomicBool>) {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        if let Ok(line) = line {
            if line.trim() == "stop" {
                stop_signal_clone.store(true, Ordering::SeqCst);
                break;
            }
        }
    }
}
