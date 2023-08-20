use boardmanager::BoardStack;
use selfplay::DataGen;
use cozy_chess::*;
// use tch::*;
mod decoder;
mod mcts_trainer;
mod boardmanager;
mod mvs;
mod selfplay;
mod dirichlet;

use std::env;
// training loop code
fn main() {


    env::set_var("RUST_BACKTRACE", "0");
    let stack:Vec<u64> = Vec::new();
    let board = Board::default();
    let bs = BoardStack{
        board,
        move_stack:stack

    };
    let mut dg = DataGen{
        iterations: 10,
        stack_manager: bs
    };
    
    let (training_data,pi_list,move_idx_list,results_list) = dg.generate_batch();

    println!("{:?}, {:?}, {:?}, {:?}", training_data, pi_list, move_idx_list, results_list);

}
