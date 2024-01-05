use std::time::Instant;

use crate::mcts_trainer::{Tree, MAX_NODES};
use crate::{
    boardmanager::BoardStack, dataformat::ZeroEvaluation, executor::Packet,
    mcts_trainer::TypeRequest,
};
use cozy_chess::Move;
use flume::Sender;

pub fn get_move(
    bs: BoardStack,
    tensor_exe_send: Sender<Packet>,
) -> (
    Move,
    ZeroEvaluation,
    Option<Vec<usize>>,
    ZeroEvaluation,
    u32,
) {
    // non-generator version of mcts_trainer.rs

    // most search code is located in mcts_trainer.rs

    let mut tree = Tree::new(bs);
    if tree.board.is_terminal() {
        panic!("No valid move!/Board is already game over!");
    }

    let search_type = TypeRequest::NonTrainerSearch;

    while tree.nodes[0].visits < MAX_NODES {
        let thread_name = std::thread::current()
            .name()
            .unwrap_or("unnamed")
            .to_owned();
        // println!("step {}", tree.nodes[0].visits);
        // // println!("thread {}, step {}", thread_name, tree.nodes[0].visits);
        let sw = Instant::now();
        tree.step(tensor_exe_send.clone(), search_type.clone());
        // println!("Elapsed time for step: {}ms", sw.elapsed().as_nanos() as f32 / 1e6);
    }
    let best_move_node = tree.nodes[0]
        .children
        .clone()
        .max_by_key(|&n| tree.nodes[n].visits)
        .expect("Error");
    let best_move = tree.nodes[best_move_node].mv;
    let mut total_visits_list = Vec::new();
    // println!("{:#}", best_move.unwrap());
    for child in tree.nodes[0].children.clone() {
        total_visits_list.push(tree.nodes[child].visits);
    }

    let display_str = tree.display_node(0); // print root node
                                            // // println!("{}", display_str);
    let total_visits: u32 = total_visits_list.iter().sum();

    let mut pi: Vec<f32> = Vec::new();

    // println!("{:?}", &total_visits_list);

    for &t in &total_visits_list {
        let prob = t as f32 / total_visits as f32;
        pi.push(prob);
    }

    // // println!("{:?}", &pi);
    // // println!("{}", best_move.expect("Error").to_string());
    // // println!("best move: {}", best_move.expect("Error").to_string());

    // for child in tree.nodes[0].children.clone() {
    //     let display_str = tree.display_node(child);
    //     // println!("{}", display_str);
    // }
    // tree.nodes[0].display_full_tree(&tree);

    let mut all_pol = Vec::new();

    for child in tree.nodes[0].clone().children {
        all_pol.push(tree.nodes[child].policy);
    }

    let v_p = ZeroEvaluation {
        // network evaluation, NOT search/empirical data
        values: tree.nodes[0].eval_score,
        policy: all_pol,
    };

    let search_data = ZeroEvaluation {
        // search data
        values: tree.nodes[0].get_q_val(),
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
