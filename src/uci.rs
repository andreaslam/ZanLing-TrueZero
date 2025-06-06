// UCI code based on https://github.com/jw1912/monty
use crate::dataformat::ZeroEvaluationAbs;
use crate::settings::{CPUCTSettings, FPUSettings, MovesLeftSettings, PSTSettings};
use crate::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    debug_print,
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

const DEFAULT_CACHE_SIZE: usize = 64; // default cache size in megabytes

const DEFAULT_BATCH_SIZE: usize = 8;

struct UCIRequest {
    board: BoardStack,
    tensor_exe_send: Sender<Packet>,
    search_settings: SearchSettings,
    stop_signal: Receiver<UCIMsg>,
    cache_size: usize,
    cache_reset: bool,
}

#[derive(Debug, Clone)]
pub enum UCIMsg {
    UCIStopMessage,
}

pub fn eval_in_cp(eval: f32) -> f32 {
    if eval > 0.5 {
        18. * (eval - 0.5) + 1.
    } else if eval < -0.5 {
        18. * (eval + 0.5) - 1.
    } else {
        2. * eval
    }
}
fn get_cache_entry_size() -> (usize, usize) {
    (
        std::mem::size_of::<CacheEntryKey>(),
        std::mem::size_of::<ZeroEvaluationAbs>(),
    )
}

fn mb_to_items(mb: usize) -> usize {
    let (key_size, val_size) = get_cache_entry_size();
    debug_print!("cache size: {}", (mb * 1000000) / (key_size + val_size));
    (mb * 1000000) / (key_size + val_size)
}

pub fn run_uci(net_path: &str) {
    panic::set_hook(Box::new(|panic_info| {
        eprintln!("Panic occurred: {:?}", panic_info);
        std::process::exit(1);
    }));

    let board = Board::default();
    let bs = BoardStack::new(board);
    let stack = Vec::new();
    let _threads = 1;

    let net = Net::new(net_path);
    let (ctrl_sender, ctrl_recv) = flume::bounded::<Message>(1);

    let (cmd_send, cmd_recv) = flume::bounded::<UCIMsg>(1); // termination message
    let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Packet>(1);

    let (job_send, job_recv) = flume::bounded::<UCIRequest>(1);

    let (finished_move_send, finished_move_recv) = flume::bounded::<Move>(1);

    let (user_input_send, user_input_recv) = flume::bounded::<String>(1);

    thread::scope(|s| {
        // spawn a thread to listen for stop and quit commands

        s.builder()
            .name("thread-listener-uci".to_string())
            .spawn(move |_| listen_to_user_inputs(cmd_send, user_input_send, ctrl_sender))
            .unwrap();

        // spawn permanent executor
        s.builder()
            .name("executor-uci".to_string())
            .spawn(move |_| {
                executor_static(net_path.to_string(), tensor_exe_recv, ctrl_recv.clone(), 1)
            })
            .unwrap();

        // spawn permanent thread to process search requests
        s.builder()
            .name("job-listener-uci".to_string())
            .spawn(move |_| {
                job_listener(job_recv, finished_move_send);
            })
            .unwrap();

        // run main uci loop to handle most commands
        s.builder()
            .name("main-loop-uci".to_string())
            .spawn(move |_| {
                uci_command_parse_loop(
                    bs,
                    cmd_recv,
                    tensor_exe_send,
                    job_send,
                    finished_move_recv,
                    stack,
                    net,
                    user_input_recv,
                );
            })
            .unwrap();
    })
    .unwrap();
}

fn uci_command_parse_loop(
    mut bs: BoardStack,
    cmd_recv: Receiver<UCIMsg>,
    tensor_exe_send: Sender<Packet>,
    job_send: Sender<UCIRequest>,
    finished_move_recv: Receiver<Move>,
    mut stack: Vec<u64>,
    net: Net,
    user_input_recv: Receiver<String>,
) {
    let mut cache_in_items = mb_to_items(DEFAULT_CACHE_SIZE); // cache size in number of items in `LruCache`
    let mut cache_reset = false;
    let mut batch_size = DEFAULT_BATCH_SIZE;
    loop {
        debug_print!("Debug: run_uci_loop ready");
        let input = user_input_recv.recv().unwrap();
        debug_print!("Debug: user input: {}", &input);
        let commands = input.split_whitespace().collect::<Vec<_>>();

        match *commands.first().unwrap_or(&"oops") {
            "uci" => {
                debug_print!("{}", &"Debug: Received 'uci' command".to_string());
                preamble();
            }
            "isready" => {
                debug_print!("{}", &"Debug: Received 'isready' command".to_string());
                println!("readyok");
            }
            "ucinewgame" => {
                cache_reset = true;
                set_position(vec!["startpos"], &mut bs, &mut stack);
            }
            "go" => {
                debug_print!("{}", &"Debug: Received 'go' command".to_string());
                handle_go(
                    &commands,
                    &bs,
                    &cmd_recv,
                    &tensor_exe_send,
                    &job_send,
                    &finished_move_recv,
                    cache_in_items,
                    cache_reset,
                    batch_size,
                );
                cache_reset = false;
            }
            "position" => {
                debug_print!("{}", &"Debug: Received 'position' command".to_string());
                set_position(commands, &mut bs, &mut stack);
            }
            "eval" => {
                debug_print!("{}", &"Debug: Received 'eval' command".to_string());
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
            "setoption" => match commands[..] {
                ["setoption", "name", "Clear", "Hash"] => cache_reset = true,
                ["setoption", "name", "Hash", "value", x] => {
                    cache_in_items = mb_to_items(x.parse().unwrap_or(DEFAULT_CACHE_SIZE))
                }
                ["setoption", "name", "Threads", "value", x] => {
                    batch_size = x.parse().unwrap_or(DEFAULT_BATCH_SIZE)
                }
                _ => {}
            },
            _ => {}
        }
        debug_print!("Debug: Finished 1 iteration of listening to user messages");
    }
}

fn listen_to_user_inputs(
    cmd_sender: Sender<UCIMsg>,
    user_input_send: Sender<String>,
    ctrl_sender: Sender<Message>,
) {
    loop {
        debug_print!("Debug: listen_to_user_inputs ready");
        let mut input = String::new();
        let bytes_read = io::stdin().read_line(&mut input).unwrap();
        if bytes_read == 0 {
            break;
        }

        debug_print!("Debug: user input: {}", &input);
        let commands = input.split_whitespace().collect::<Vec<_>>();

        match *commands.first().unwrap_or(&"oops") {
            "stop" => {
                let _ = cmd_sender.send(UCIMsg::UCIStopMessage);
                debug_print!("{}", &"Debug: Sent 'stop' command".to_string());
            }
            "quit" => {
                ctrl_sender.send(Message::StopServer).unwrap();
                process::exit(0);
            }
            _ => user_input_send.send(input).unwrap(),
        }
        debug_print!("Debug: Finished 1 iteration of listen_to_user_inputs");
    }
}

fn preamble() {
    println!("id name TrueZero-latest {}", env!("CARGO_PKG_VERSION"));
    println!("id author Andreas Lam");
    println!("uciok");
    let cache_entry_size = get_cache_entry_size();
    println!(
        "option name Hash type spin default {} min 1 max {}",
        DEFAULT_CACHE_SIZE,
        (i32::MAX / (cache_entry_size.0 + cache_entry_size.1) as i32)
    );
    println!(
        "option name Threads type spin default {} min 1 max {}",
        DEFAULT_BATCH_SIZE,
        i32::MAX
    );
}

fn check_castling_move_on_input(bs: &BoardStack, mut mv: Move) -> Move {
    debug_print!("{}", &"Debug: Checking castling move".to_string());
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

pub fn check_castling_move_on_output(bs: &BoardStack, mut mv: Move) -> Move {
    debug_print!("{}", &"Debug: Checking castling move".to_string());
    if bs.board().piece_on(mv.from) == Some(Piece::King) {
        mv.to = match (mv.from, mv.to) {
            (Square::E1, Square::H1) => Square::G1,
            (Square::E8, Square::H8) => Square::G8,
            (Square::E1, Square::A1) => Square::C1,
            (Square::E8, Square::A8) => Square::C8,
            _ => mv.to,
        };
    }
    mv
}

fn set_position(commands: Vec<&str>, bs: &mut BoardStack, stack: &mut Vec<u64>) {
    debug_print!("{}", &"Debug: Setting position".to_string());
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
        debug_print!("Debug: setting FEN to startpos");
        STARTPOS
    } else {
        fen.trim()
    };
    let board = Board::from_fen(fenstr, false).unwrap_or_default();
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
            let mv = Move::from_str(m).unwrap();
            let mv = check_castling_move_on_input(bs, mv);
            if mv == *mov {
                bs.play(*mov);
            }
        }
    }
}

fn handle_go(
    commands: &[&str],
    bs: &BoardStack,
    cmd_receiver: &Receiver<UCIMsg>,
    tensor_exe_send: &Sender<Packet>,
    job_sender: &Sender<UCIRequest>,
    finished_move_receiver: &Receiver<Move>,
    cache_size: usize,
    cache_reset: bool,
    batch_size: usize,
) {
    debug_print!("{}", &"Debug: Handling 'go' command".to_string());

    let mut finite_search = true;

    let mut nodes = 1600;
    let mut max_time = None;
    let mut max_depth = 256;

    let mut times = [None; 2];
    let mut incs = [None; 2];
    let mut movestogo = 30;
    let mut mode = "";

    for cmd in commands {
        match *cmd {
            "infinite" => {
                finite_search = false;
                mode = "infinite"
            }
            "nodes" => mode = "nodes",
            "movetime" => mode = "movetime",
            "depth" => mode = "depth",
            "wtime" => mode = "wtime",
            "btime" => mode = "btime",
            "winc" => mode = "winc",
            "binc" => mode = "binc",
            "movestogo" => mode = "movestogo",
            _ => match mode {
                "infinite" => {
                    debug_print!(
                        "{}",
                        &format!("Debug: Initiating infinite search {}", movestogo)
                    )
                }
                "nodes" => {
                    nodes = cmd.parse().unwrap_or(nodes);
                    debug_print!("{}", &format!("Debug: Set 'nodes' to {}", nodes));
                }
                "movetime" => {
                    max_time = cmd.parse().ok();
                    debug_print!("{}", &format!("Debug: Set 'max_time' to {:?}", max_time));
                }
                "depth" => {
                    max_depth = cmd.parse().unwrap_or(max_depth);
                    debug_print!("{}", &format!("Debug: Set 'max_depth' to {}", max_depth));
                }
                "wtime" => {
                    times[0] = Some(cmd.parse().unwrap_or(0));
                    debug_print!("{}", &format!("Debug: Set 'wtime' to {:?}", times[0]));
                }
                "btime" => {
                    times[1] = Some(cmd.parse().unwrap_or(0));
                    debug_print!("{}", &format!("Debug: Set 'btime' to {:?}", times[1]));
                }
                "winc" => {
                    incs[0] = Some(cmd.parse().unwrap_or(0));
                    debug_print!("{}", &format!("Debug: Set 'winc' to {:?}", incs[0]));
                }
                "binc" => {
                    incs[1] = Some(cmd.parse().unwrap_or(0));
                    debug_print!("{}", &format!("Debug: Set 'binc' to {:?}", incs[1]));
                }
                "movestogo" => {
                    movestogo = cmd.parse().unwrap_or(30);
                    debug_print!("{}", &format!("Debug: Set 'movestogo' to {}", movestogo));
                }
                _ => mode = "none",
            },
        }
    }
    debug_print!("mode: {}", mode);
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
        nodes = time.unwrap() / 50;
    }
    nodes = max(1, nodes);
    if let Some(max) = max_time {
        time = Some(time.unwrap_or(u128::MAX).min(max));
        nodes = time.unwrap() / 50;
    }

    if let Some(t) = time.as_mut() {
        *t = t.saturating_sub(5);
    }

    let m_settings = MovesLeftSettings {
        moves_left_weight: 0.03,
        moves_left_clip: 20.0,
        moves_left_sharpness: 0.5,
    };

    let search_nodes: Option<u128>;

    if finite_search {
        debug_print!("Visit limit {}", nodes);
        search_nodes = Some(nodes);
    } else {
        debug_print!("Visit limit: None");
        search_nodes = None;
    }

    let settings: SearchSettings = SearchSettings {
        fpu: FPUSettings {
            root_fpu: 1.0,
            children_fpu: 0.75,
        },
        wdl: EvalMode::Wdl,
        moves_left: Some(m_settings),
        c_puct: CPUCTSettings {
            root_c_puct: 4.0,
            children_c_puct: 2.0,
        },
        max_nodes: search_nodes,
        alpha: 0.03,
        eps: 0.25,
        search_type: UCISearch,
        pst: PSTSettings {
            root_pst: 1.75,
            children_pst: 1.5,
        },
        batch_size,
    };

    let search_request = UCIRequest {
        board: bs.clone(),
        tensor_exe_send: tensor_exe_send.clone(),
        search_settings: settings,
        stop_signal: cmd_receiver.clone(),
        cache_size: cache_size.max(1),
        cache_reset,
    };

    debug_print!("Debug: Gathered search request");
    job_sender.send(search_request).unwrap();

    let best_move = finished_move_receiver.recv().unwrap();

    println!("bestmove {:#}", best_move);
    debug_print!("{}", "sent termination message");
    debug_print!("{}", "handle go done");
}
fn job_listener(job_receiver: Receiver<UCIRequest>, finished_move_sender: Sender<Move>) {
    // listen to jobs and run search requests
    let rt = Runtime::new().unwrap();

    let mut cache: LruCache<CacheEntryKey, ZeroEvaluationAbs> =
        LruCache::new(NonZeroUsize::new(mb_to_items(DEFAULT_CACHE_SIZE)).unwrap());
    debug_print!("Debug: Job Listener ready");
    loop {
        let received_message = job_receiver.recv();
        match received_message {
            Ok(search_request) => {
                debug_print!("Debug: Received new UCI search request");
                let old_cache_cap = cache.cap();
                if search_request.cache_reset {
                    cache.clear();
                    assert!(cache.is_empty());
                }
                if old_cache_cap != NonZeroUsize::new(search_request.cache_size).unwrap() {
                    cache.resize(NonZeroUsize::new(search_request.cache_size).unwrap());
                    debug_print!(
                        "Debug: Resized cache from {} to {}",
                        old_cache_cap,
                        cache.cap()
                    )
                }
                let castling_board = search_request.board.clone();
                let (mut best_move, _, _, _, _) = rt.block_on(async {
                    get_move(
                        search_request.board,
                        search_request.tensor_exe_send,
                        search_request.search_settings,
                        Some(search_request.stop_signal),
                        &mut cache,
                    )
                    .await
                });
                best_move = check_castling_move_on_output(&castling_board, best_move);
                debug_print!("Debug: Finished UCI Search. Best move: {:#}", best_move);
                // send the best move back to main thread
                finished_move_sender.send(best_move).unwrap();
            }
            Err(_msg) => {
                debug_print!("Error: Failed to receive message: {}", _msg);
            }
        };
    }
}
