use std::env;
use tzrust::{data_path_str, uci::run_uci};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    run_uci(&data_path_str(r"C:\Users\andre\RemoteFolder\ZanLing-TrueZero\nets\tz_1.pt"));
}
