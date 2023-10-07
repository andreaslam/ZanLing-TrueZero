use std::{convert, env};
use stopwatch::Stopwatch;
use cozy_chess::Board;
use tz_rust::{boardmanager::BoardStack, decoder::convert_board, decoder::eval_board};

fn main() {
    // test board conversion
    env::set_var("RUST_BACKTRACE", "1");
    let board = Board::from_fen(
        "rn1q1k1r/pp2pp1p/8/8/1Np5/8/PPP3PP/R2QKB1R w KQ - 0 1",
        false,
    )
    .unwrap();
    let bs = BoardStack::new(board);
    let mut sw = Stopwatch::new();
    sw.start();
    let converted_tensor = convert_board(&bs);
    sw.stop();
    let print_vec: Vec<f32> = Vec::try_from(converted_tensor).expect("Error");
    println!("{:?}", print_vec);
    println!("Elapsed time: {}ms", sw.elapsed_ms());
}
