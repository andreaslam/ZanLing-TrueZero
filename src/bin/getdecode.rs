use cozy_chess::Board;
use std::{env, time::Instant};
use tz_rust::{
    boardmanager::BoardStack,
    decoder::convert_board,
    utils::{debug_print, TimeStampDebugger},
};

fn main() {
    // test board conversion
    env::set_var("RUST_BACKTRACE", "1");
    let board = Board::from_fen(
        "rn1q1k1r/pp2pp1p/8/8/1Np5/8/PPP3PP/R2QKB1R w KQ - 0 1",
        false,
    )
    .unwrap();
    let bs = BoardStack::new(board);
    let sw = Instant::now();
    let converted_tensor = convert_board(&bs);
    let converted_tensor = converted_tensor.reshape([21, 8, 8]);
    converted_tensor.print();
    debug_print(&format!(
        "Elapsed time: {}ms",
        sw.elapsed().as_nanos() as f32 / 1e6
    ));
}
