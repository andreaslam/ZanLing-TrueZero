use std::env;

use cozy_chess::Board;
use tz_rust::selfplay::DataGen;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let board = Board::from_fen(
        "rn1q1k1r/pp2pp1p/8/8/1Np5/8/PPP3PP/R2QKB1R w KQ - 0 1",
        false,
    )
    .unwrap();
    let mut dg = DataGen { iterations: 10 };
    (_, _, _, _) = dg.generate_batch();
}
