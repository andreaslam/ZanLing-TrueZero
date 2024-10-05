use std::env;
use tzrust::uci::run_uci;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    // run_uci(r"chess_16x128_gen3634.pt");
    run_uci(r"nets/tz_4924.pt");
    // run_uci(r"tz_6515.pt");
}
