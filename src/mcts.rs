use std::time::Instant;

use crate::{
    boardmanager::BoardStack, dataformat::ZeroEvaluation, executor::Packet, mcts_trainer::Tree,
    settings::SearchSettings,
};
use cozy_chess::Move;

use flume::Sender;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub async fn get_move(
    bs: BoardStack,
    tensor_exe_send: Sender<Packet>,
    settings: SearchSettings,
    stop_signal: Option<Arc<AtomicBool>>,
) -> (
    Move,
    ZeroEvaluation,
    Option<Vec<usize>>,
    ZeroEvaluation,
    u32,
) {
    let sw = Instant::now();

    let mut tree = Tree::new(bs, settings);
    if tree.board.is_terminal() {
        panic!("No valid move!/Board is already game over!");
    }

    while tree.nodes[0].visits < settings.max_nodes as u32 {
        match stop_signal {
            Some(ref signal) => {
                if signal.load(Ordering::SeqCst) {
                    break;
                }
            }
            None => {}
        }

        tree.step(&tensor_exe_send, sw, 0).await;
    }

    let mut child_visits: Vec<u32> = Vec::new();
    for child in tree.nodes[0].children.clone() {
        child_visits.push(tree.nodes[child].visits);
    }

    let all_same = child_visits.iter().all(|&x| x == child_visits[0]);

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
                a_node.policy.partial_cmp(&b_node.policy).unwrap()
            })
            .expect("Error")
    };

    let best_move = tree.nodes[best_move_node].mv;
    let mut total_visits_list = Vec::new();
    for child in tree.nodes[0].children.clone() {
        total_visits_list.push(tree.nodes[child].visits);
    }

    let total_visits: u32 = total_visits_list.iter().sum();

    let mut pi: Vec<f32> = Vec::new();
    for &t in &total_visits_list {
        let prob = t as f32 / total_visits as f32;
        pi.push(prob);
    }

    let mut all_pol = Vec::new();
    for child in tree.nodes[0].clone().children {
        all_pol.push(tree.nodes[child].policy);
    }

    let v_p = ZeroEvaluation {
        values: tree.nodes[0].eval_score,
        policy: all_pol,
    };

    let search_data = ZeroEvaluation {
        values: tree.nodes[0].get_q_val(settings),
        policy: pi,
    };

    (
        best_move.expect("Error"),
        v_p,
        tree.nodes[0].clone().move_idx,
        search_data,
        tree.nodes[0].visits,
    )
}
