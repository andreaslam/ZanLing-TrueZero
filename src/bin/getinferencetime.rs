use std::env;
use stopwatch::Stopwatch;
use tz_rust::{decoder::eval_state, mcts_trainer::Net};
use tch::Tensor;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let net = Net::new();
    let mut avg = Vec::new();
    const batch_size: usize = 128;
    for _ in 0..100 {
        let mut sw = Stopwatch::new();
        let data = Tensor::from_slice(&[0.0 as f32;1344*batch_size]); // 8*8*21 = 1344
        sw.start();
        (_, _) = eval_state(data, &net).expect("Error");   
        sw.stop();
        println!("Elapsed time: {}ms", sw.elapsed_ms());
        let eps = batch_size.clone() as f32/ (sw.elapsed_ms() as f32 / 1000.0);
        println!("Evaluations per second: {}", eps);
        avg.push(eps);
    }
    println!("Average evaluations per second: {}", avg.iter().sum::<f32>() as f32 / avg.len() as f32);
    println!("{:?}", avg);
}
