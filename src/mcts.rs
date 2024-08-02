use crate::{
    boardmanager::BoardStack,
    cache::{CacheEntryKey, CacheEntryValue},
    dataformat::{ZeroEvaluation, ZeroValuesPov},
    debug_print,
    executor::Packet,
    mcts_trainer::{Tree, Wdl},
    settings::SearchSettings,
    uci::UCIMsg,
};
use cozy_chess::Color;
use cozy_chess::Move;
use flume::{Receiver, Sender, TryRecvError};
use lru::LruCache;
use std::time::Instant;
use tokio::signal;

pub async fn get_move(
    bs: BoardStack,
    tensor_exe_send: Sender<Packet>,
    settings: SearchSettings,
    stop_signal: Option<Receiver<UCIMsg>>,
    mut cache: &mut LruCache<CacheEntryKey, CacheEntryValue>,
) -> (
    Move,
    ZeroEvaluation,
    Option<Vec<usize>>,
    ZeroEvaluation,
    u32,
) {
    let sw = Instant::now();
    debug_print!("Debug: Start of get_move function");

    let mut tree = Tree::new(bs, settings);
    debug_print!("Debug: Tree initialized");

    if tree.board.is_terminal() {
        panic!("No valid move!/Board is already game over!");
    }

    while tree.nodes[0].visits < settings.max_nodes as u32 {
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
    debug_print!("Debug: Tree search completed");

    let mut child_visits: Vec<u32> = Vec::new();
    for child in tree.nodes[0].children.clone() {
        child_visits.push(tree.nodes[child].visits);
    }

    let all_same = child_visits.iter().all(|&x| x == child_visits[0]);
    debug_print!("Debug: All child visits same: {}", all_same);

    let best_move_node = if !all_same {
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
    debug_print!("Debug: Best move node selected: {}", best_move_node);

    let best_move = tree.nodes[best_move_node].mv;
    debug_print!("Debug: Best move determined: {:?}", best_move);
    let mut total_visits_list = Vec::new();
    for child in tree.nodes[0].children.clone() {
        total_visits_list.push(tree.nodes[child].visits);
        let msg = tree.display_node(child);
        debug_print!("{}", msg);
    }
    debug_print!("Debug: Total visits list: {:?}", total_visits_list);

    let total_visits: u32 = total_visits_list.iter().sum();
    debug_print!("Debug: Total visits: {}", total_visits);

    let mut pi: Vec<f32> = Vec::new();
    for &t in &total_visits_list {
        let prob = t as f32 / total_visits as f32;
        pi.push(prob);
    }
    debug_print!("Debug: Pi vector: {:?}", pi);

    let mut all_pol = Vec::new();
    for child in tree.nodes[0].clone().children {
        all_pol.push(tree.nodes[child].policy);
    }

    debug_print!("Debug: All policies: {:?}", all_pol);
    tree.nodes[0].display_full_tree(&tree);
    let v_p_vals = ZeroValuesPov {
        value: tree.nodes[0].wdl.w - tree.nodes[0].wdl.l,
        wdl: tree.nodes[0].wdl,
        moves_left: tree.nodes[0].moves_left,
    };

    let v_p = ZeroEvaluation {
        // network evaluation, NOT search/empirical data
        values: v_p_vals,
        policy: all_pol,
    };
    debug_print!("Debug: ZeroEvaluation v_p created");

    let search_data_vals = ZeroValuesPov {
        value: match tree.board.board().side_to_move() {
            Color::White => tree.nodes[0].get_q_val(tree.settings),
            Color::Black => -tree.nodes[0].get_q_val(tree.settings),
        },
        wdl: match tree.board.board().side_to_move() {
            Color::White => Wdl {
                w: tree.nodes[0].total_wdl.w / tree.nodes[0].visits as f32,
                d: tree.nodes[0].total_wdl.d / tree.nodes[0].visits as f32,
                l: tree.nodes[0].total_wdl.l / tree.nodes[0].visits as f32,
            },
            Color::Black => Wdl {
                w: tree.nodes[0].total_wdl.l / tree.nodes[0].visits as f32,
                d: tree.nodes[0].total_wdl.d / tree.nodes[0].visits as f32,
                l: tree.nodes[0].total_wdl.w / tree.nodes[0].visits as f32,
            },
        },
        moves_left: tree.nodes[0].moves_left_total / tree.nodes[0].visits as f32,
    };

    let search_data = ZeroEvaluation {
        // search data
        values: search_data_vals,
        policy: pi,
    };
    debug_print!("Debug: ZeroEvaluation search_data created");

    debug_print!("Debug: ZeroEvaluation search_data created");

    debug_print!("Debug: End of get_move function");
    (
        best_move.expect("Error"),
        v_p,
        tree.nodes[0].clone().move_idx,
        search_data,
        tree.nodes[0].visits,
    )
}
