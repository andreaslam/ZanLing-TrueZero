use cozy_chess::Board;
use std::{env, time::Instant};
use tz_rust::{boardmanager::BoardStack, decoder::convert_board};

fn main() {
    // test board conversion
    env::set_var("RUST_BACKTRACE", "1");
    let board = Board::from_fen(
        "r2q1k1r/3bnp2/p1n1pNp1/3pP1Qp/Pp1P4/2PB4/5PPP/R1B2RK1 w - - 1 1",
        false,
    )
    .unwrap();
    // let board = Board::default();
    let bs = BoardStack::new(board);
    let sw = Instant::now();
    let converted_tensor = convert_board(&bs);
    println!("Elapsed time: {}ms", sw.elapsed().as_nanos() as f32 / 1e6);
    let converted_tensor = converted_tensor.reshape([-1]);
    converted_tensor.print();
}
