// use cozy_chess::Board;
// use std::{convert, env};
// use stopwatch::Stopwatch;
// use tz_rust::mcts_trainer;
// use tz_rust::{boardmanager::BoardStack, mcts_trainer::get_move};

// fn main() {
//     // test MCTS move outputs
//     env::set_var("RUST_BACKTRACE", "1");
//     let board = Board::default();
//     // let board = Board::from_fen(
//     //     "6rk/p3p2p/1p2Pp2/2p2P2/2P1nBr1/1P6/P6P/3R1R1K b - - 0 1",
//     //     false,
//     // )
//     // .unwrap(); // black M2
//     // let board = Board::from_fen(
//     //     "r2q1k1r/3bnp2/p1n1pNp1/3pP1Qp/Pp1P4/2PB4/5PPP/R1B2RK1 w - - 1 1",
//     //     false,
//     // )
//     // .unwrap(); // white M2
//     let mut total_moves = 0;
//     board.generate_moves(|moves| {
//         // Done this way for demonstration.
//         // Actual counting is best done in bulk with moves.len().
//         for _mv in moves {
//             total_moves += 1;
//         }
//         false
//     });
//     println!("Number of legal moves: {}", total_moves);
//     let bs = BoardStack::new(board);
//     let mut sw = Stopwatch::new();
//     sw.start();
//     let (_, _, _, _, _) = get_move(bs);
//     sw.stop();
//     println!("Elapsed time: {}ms", sw.elapsed_ms());
//     let nps = mcts_trainer::MAX_NODES as f32 / (sw.elapsed_ms() as f32 / 1000.0);
//     println!("Nodes per second: {}nps", nps);
// }
