use std::env;
use tch::maybe_init_cuda;
use tzrust::uci::run_uci;

fn main() {
    maybe_init_cuda();
    env::set_var("RUST_BACKTRACE", "1");
    run_uci(r"nets/tz_10.pt");
}
