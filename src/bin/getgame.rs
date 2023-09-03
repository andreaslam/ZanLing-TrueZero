use std::{convert, env};

use cozy_chess::Board;
use tz_rust::selfplay::DataGen;

fn main() {
    // test game with MCTS
    env::set_var("RUST_BACKTRACE", "1");
    let mut dg = DataGen { iterations: 1 };
    let (_, _, _, _) = dg.generate_batch();
}
