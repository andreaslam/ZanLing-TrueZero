use std::env;
use stopwatch::Stopwatch;
use tz_rust::{decoder::eval_state, mcts_trainer::Net};
use tch::Tensor;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let net = Net::new();
    let data = Tensor::from_slice(&[0.0 as f32;1344*2048]); // 8*8*21 = 1344
    let mut sw = Stopwatch::new();
    println!("{:?}", net.device);
    sw.start();
    (_, _) = eval_state(data, &net).expect("Error");   
    sw.stop();
    println!("Elapsed time: {}ms", sw.elapsed_ms());
    let eps = 1.0 / (sw.elapsed_ms() as f32 / 1000.0);
    println!("Evaluations per second: {}", eps);
}
