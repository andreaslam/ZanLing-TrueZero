use cozy_chess::*;
// use tch::*;
mod decoder;
mod mcts_trainer;
mod boardmanager;
mod mvs;
use crate::boardmanager::BoardStack;
use crate::decoder::eval_board;

// training loop code
fn main() {
    let board = Board::default();
    let bs = BoardStack {board:board,move_stack:Vec::new()};
    eval_board(board, bs);
}
