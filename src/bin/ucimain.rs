use std::env;
use tz_rust::uci::run_uci;

fn main() {
    env::set_var("RUST_BACKTRACE", "2");
    run_uci(r"./tz_6515.pt");
}
