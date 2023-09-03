use std::env;
use tz_rust::selfplay::DataGen;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let mut dg = DataGen { iterations: 1 };
    (_, _, _, _) = dg.generate_batch();
}
