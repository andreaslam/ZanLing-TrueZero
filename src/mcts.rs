use crate::{
    boardmanager::BoardStack, cache::CacheEntryKey, dataformat::ZeroEvaluationAbs, debug_print,
    executor::Packet, mcts_trainer::Tree, settings::SearchSettings, uci::UCIMsg,
};
use cozy_chess::Move;
use flume::{Receiver, Sender, TryRecvError};
use lru::LruCache;
use std::time::Instant;

pub async fn get_move(
    bs: BoardStack,
    tensor_exe_send: Sender<Packet>,
    settings: SearchSettings,
    stop_signal: Option<Receiver<UCIMsg>>,
    cache: &mut LruCache<CacheEntryKey, ZeroEvaluationAbs>,
) -> (
    Move,
    ZeroEvaluationAbs,
    Option<Vec<usize>>,
    ZeroEvaluationAbs,
    u32,
) {
    let sw = Instant::now();
    debug_print!("Debug: Start of get_move function");

    let mut tree = Tree::new(bs, settings);
    debug_print!("Debug: Tree initialized");

    if tree.board.is_terminal() {
        panic!("No valid move!/Board is already game over!");
    }
    match settings.max_nodes {
        Some(max_nodes) => {
            while tree.nodes[0].visits < max_nodes as u32 {
                debug_print!("step {}", tree.nodes[0].visits);
                match stop_signal {
                    Some(ref signal) => match signal.try_recv() {
                        Ok(_) => {
                            if tree.nodes[0].visits < 1 {
                                debug_print!("Debug: Insufficient search to produce output");
                                tree.step(&tensor_exe_send, sw, 0, cache).await;
                            } else {
                                debug_print!("Debug: Stop signal received, breaking loop");
                                break;
                            }
                        }
                        Err(TryRecvError::Empty) => {
                            debug_print!("Debug: No stop signal, stepping tree");
                            tree.step(&tensor_exe_send, sw, 0, cache).await;
                        }
                        Err(TryRecvError::Disconnected) => {
                            debug_print!("Debug: Stop signal disconnected, breaking loop");
                            break;
                        }
                    },
                    None => {
                        tree.step(&tensor_exe_send, sw, 0, cache).await;
                    }
                }
            }
        }
        None => {
            // go infinite/infinite search

            loop {
                debug_print!("step {}", tree.nodes[0].visits);
                match stop_signal {
                    Some(ref signal) => match signal.try_recv() {
                        Ok(_) => {
                            if tree.nodes[0].visits < 1 {
                                debug_print!("Debug: Insufficient search to produce output");
                                tree.step(&tensor_exe_send, sw, 0, cache).await;
                            } else {
                                debug_print!("Debug: Stop signal received, breaking loop");
                                break;
                            }
                        }
                        Err(TryRecvError::Empty) => {
                            debug_print!("Debug: No stop signal, stepping tree");
                            tree.step(&tensor_exe_send, sw, 0, cache).await;
                        }
                        Err(TryRecvError::Disconnected) => {
                            debug_print!("Debug: Stop signal disconnected, breaking loop");
                            break;
                        }
                    },
                    None => {
                        tree.step(&tensor_exe_send, sw, 0, cache).await;
                    }
                }
            }
        }
    }
    let mut child_visits: Vec<u32> = Vec::new();

    for child in tree.nodes[0].children.clone() {
        child_visits.push(tree.nodes[child].visits);
    }

    let all_same_visits = child_visits.iter().all(|&x| x == child_visits[0]);

    let best_move_node = if !all_same_visits {
        // if visits to nodes are the same eg max_nodes=1
        tree.nodes[0]
            .children
            .clone()
            .max_by_key(|&n| tree.nodes[n].visits)
            .expect("Error")
    } else {
        tree.nodes[0]
            .children
            .clone()
            .max_by(|a, b| {
                let a_node = &tree.nodes[*a];
                let b_node = &tree.nodes[*b];
                let a_policy = a_node.policy;
                let b_policy = b_node.policy;
                a_policy.partial_cmp(&b_policy).unwrap()
            })
            .expect("Error")
    };
    let best_move = tree.nodes[best_move_node].mv;
    let mut total_visits_list = Vec::new();
    debug_print!("{}", &format!("{:#}", best_move.unwrap()));
    for child in tree.nodes[0].children.clone() {
        total_visits_list.push(tree.nodes[child].visits);
        let msg = tree.display_node(child);
        debug_print!("{}", msg);
    }

    let display_str = tree.display_node(0);
    debug_print!("{}", &format!("{}", display_str));
    let total_visits: u32 = total_visits_list.iter().sum();

    let mut pi: Vec<f32> = Vec::new();

    debug_print!("{}", &format!("{:?}", &total_visits_list));

    for &t in &total_visits_list {
        let prob = t as f32 / total_visits as f32;
        pi.push(prob);
    }

    debug_print!("{}", &format!("{:?}", &pi));
    debug_print!("{}", &format!("{}", best_move.expect("Error").to_string()));
    debug_print!(
        "{}",
        &format!("best move: {}", best_move.expect("Error").to_string())
    );

    for child in tree.nodes[0].children.clone() {
        let display_str = tree.display_node(child);
        debug_print!("{}", &format!("{}", display_str));
    }

    let mut all_tree_pol = Vec::new();

    for child in tree.nodes[0].clone().children {
        all_tree_pol.push(tree.nodes[child].policy);
    }

    let net_evaluation = ZeroEvaluationAbs {
        // network evaluation, NOT search/empirical data
        values: tree.nodes[0].net_evaluation,
        policy: all_tree_pol,
    };

    let search_data = ZeroEvaluationAbs {
        // search data
        values: tree.nodes[0].total_evaluation,
        policy: pi,
    };

    debug_print!("Debug: Tree search completed");

    (
        best_move.expect("Error"),
        net_evaluation,
        tree.nodes[0].clone().move_idx,
        search_data,
        tree.nodes[0].visits,
    )
}
