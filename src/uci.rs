use cozy_chess::{Board, Move, Color, Piece, Square};
use crossbeam::thread;
use std::str::FromStr;
use crate::decoder::{eval_board, eval_state, convert_board};
use crate::boardmanager::BoardStack;
use crate::mcts::get_move;
use crate::mcts_trainer::Net;
use crate::executor::executor_static;
use crate::executor::{Message, Packet};

use std::{io, process, sync::atomic::AtomicBool, time::Instant};

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn eval_in_cp(eval: f32) -> f32 {
    let cps = if eval > 0.5 {
        18. * (eval - 0.5) + 1.
    } else if eval < -0.5 {
        18. * (eval + 0.5) - 1.
    } else {
        2. * eval
    };
    cps
}

pub fn run_uci() {
    // initialise engine
    let mut board = Board::default();
    let mut bs = BoardStack::new(board);
    let mut stack = Vec::new();
    let mut threads = 1;


    let mut stored_message: Option<String> = None;
    let net_path = r"C:\Users\andre\RemoteFolder\ZanLing-TrueZero\nets\tz_2074.pt";
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
            "ucinewgame" => {
            }
            "go" => handle_go(
                &commands,
                &bs,
                &net_path,
            ),
            "position" => set_position(commands, &mut bs, &mut stack),
            "quit" => process::exit(0),
            "eval" => {
            let (value, _) = eval_state(convert_board(&bs), &net).unwrap();
            let value_raw: Vec<f32>  = Vec::try_from(value).expect("Error");
            let value: f32 = value_raw[0].tanh();
            let cps = eval_in_cp(value);
            println!("eval: {}", (cps * 100.).round().max(-1000.).min(1000.) as i64);
        },
            _ => {}
        }
    }
}

fn preamble() {
    println!("id name TrueZero {}", env!("CARGO_PKG_VERSION"));
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
    let board = Board::from_fen(fenstr, false).unwrap();
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


pub fn handle_go(
    commands: &[&str],
    bs: &BoardStack,
    net_path: &str
) {
    let mut nodes = 10_000_000;
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
                "winc" =>  incs[0]= Some(cmd.parse().unwrap_or(0)),
                "binc" =>  incs[1]= Some(cmd.parse().unwrap_or(0)),
                "movestogo" => movestogo = cmd.parse().unwrap_or(30),
                _ => mode = "none"
            },
        }
    }
    println!("max_time {:?}", max_time);
    let mut time: Option<u128> = None;
    let mut nodes: u128 = 1600; // TODO: dynamically set max_nodes using this setting thru SearchSettings struct
    // `go wtime <wtime> btime <btime> winc <winc> binc <binc>``
    let stm = bs.board().side_to_move();
    let stm_num = match stm {
        Color::White => {Some(0)},
        Color::Black => {Some(1)},
    };
    if let Some(t) = times[stm_num.unwrap()] {
        println!("{}",t);
        let mut base = t / movestogo.max(1);

        if let Some(i) = incs[stm_num.unwrap()] {
            base += i * 3 / 4;
        }
        time = Some(base.try_into().unwrap());
        nodes = time.unwrap() as u128 /20;
    }

    // `go movetime <time>`
    if let Some(max) = max_time {
        // if both movetime and increment time controls given, use
        time = Some(time.unwrap_or(u128::MAX).min(max));
        nodes = time.unwrap() as u128 /20;
    }

    // 5ms move overhead
    if let Some(t) = time.as_mut() {
        *t = t.saturating_sub(5);
    }
    let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Packet>(1);
    let (ctrl_sender, ctrl_recv) = flume::bounded::<Message>(1);
    let _ = thread::scope(|s| {
        s.builder()
            .name("executor".to_string())
            .spawn(move |_| {
                executor_static(net_path.to_string(), tensor_exe_recv, ctrl_recv, 1)
            })
            .unwrap();

    let (best_move, _, _, _, _) = get_move(bs.clone(), tensor_exe_send.clone());

    println!("bestmove {:#}", best_move);
    let _ = ctrl_sender.send(Message::StopServer());
}).unwrap();
}
