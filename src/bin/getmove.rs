use std::{convert, env};

use cozy_chess::Board;
use tz_rust::{boardmanager::BoardStack, mcts_trainer::get_move};

fn main() {
    // test MCTS move outputs
    env::set_var("RUST_BACKTRACE", "1");
    // let mut board = Board::default();
    // board.play("e2e4".parse().unwrap());
    // let mut board = Board::from_fen(
    //     "4kb1r/p2n1ppp/4q3/4p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 1 1",
    //     false,
    // )
    // .unwrap();
    let mut board = Board::from_fen(
        "rn2kb1r/ppp1pppp/8/8/4q3/3P1N1b/PPP1BPnP/RNBQ1K1R b kq - 0 1",
        false,
    )
    .unwrap();
    let mut total_moves = 0;
    board.generate_moves(|moves| {
        // Done this way for demonstration.
        // Actual counting is best done in bulk with moves.len().
        for _mv in moves {
            total_moves += 1;
        }
        false
    });
    println!("{}", total_moves);
    let bs = BoardStack::new(board);
    let (_, _, _) = get_move(bs);
}
