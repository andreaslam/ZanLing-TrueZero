use std::env;
use tch::maybe_init_cuda;
use tzrust::uci::run_uci;

fn main() {
    // maybe_init_cuda();
    env::set_var("RUST_BACKTRACE", "1");
    // run_uci(r"chess_16x128_gen3634.pt");
    run_uci(r"tz_6515.pt");
}
