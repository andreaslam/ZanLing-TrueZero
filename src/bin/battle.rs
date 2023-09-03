use std::env;

use cozy_chess::Board;
use tz_rust::{boardmanager::BoardStack, mcts_trainer::get_move, selfplay::DataGen};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    // let mut board =
    //     Board::from_fen("3Q4/p3b1k1/2p2rPp/2q5/4B3/P2P4/7P/6RK w - - 1 1", false).unwrap();
    let board = Board::default();
    let mut dg = DataGen { iterations: 30 };
    (_, _, _, _) = dg.generate_batch();
}
