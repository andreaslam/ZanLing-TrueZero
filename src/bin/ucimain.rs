use std::env;
use tch::maybe_init_cuda;
use tz_rust::uci::run_uci;

fn main() {
    maybe_init_cuda();
    env::set_var("RUST_BACKTRACE", "1");
    run_uci(r"nets/tz_1391.pt");
}
