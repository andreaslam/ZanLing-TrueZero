use cozy_chess::*;
use tz_rust::boardmanager::BoardStack;
use tz_rust::selfplay::DataGen;
// use tch::*;
// mod decoder;
// mod mcts_trainer;
// mod boardmanager;
// mod mvs;
// mod selfplay;
// mod dirichlet;

use std::env;
// training loop code
fn main() {
    env::set_var("RUST_BACKTRACE", "0");
    let board = Board::default();
    let bs = BoardStack::new(board);

    let mut dg = DataGen {
        iterations: 10,
        bs,
    };

    let (training_data, pi_list, move_idx_list, results_list) = dg.generate_batch();

    println!(
        "{:?}, {:?}, {:?}, {:?}",
        training_data, pi_list, move_idx_list, results_list
    );
}
