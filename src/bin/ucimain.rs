use std::env;

use tz_rust::uci::run_uci;

fn main() {
    env::set_var("RUST_BACKTRACE", "2");
    run_uci(r"C:\Users\andre\RemoteFolder\ZanLing-TrueZero\nets\tz_6515.pt");
}
