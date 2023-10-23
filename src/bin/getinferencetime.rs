use std::env;
use stopwatch::Stopwatch;
use tch::Tensor;
use tz_rust::{decoder::eval_state, mcts_trainer::Net};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let net = Net::new();
    let mut avg = Vec::new();
    const BATCH_SIZE: usize = 128;

    // warmup loop

    for _ in 0..10 {
        let data = Tensor::from_slice(&[0.0 as f32; 1344 * BATCH_SIZE]); // 8*8*21 = 1344
        (_, _) = eval_state(data, &net).expect("Error");
    }

    // timed, benchmarked loop

    for _ in 0..100 {
        let mut sw = Stopwatch::new();
        let data = Tensor::from_slice(&[0.0 as f32; 1344 * BATCH_SIZE]); // 8*8*21 = 1344
        sw.start();
        (_, _) = eval_state(data, &net).expect("Error");
        sw.stop();
        println!("Elapsed time: {}ms", sw.elapsed_ms());
        let eps = BATCH_SIZE.clone() as f32 / (sw.elapsed_ms() as f32 / 1000.0);
        println!("Evaluations per second: {} evals/s", eps);
        avg.push(eps);
    }
    println!(
        "Average evaluations per second: {} evals/s (batch size {})",
        avg.iter().sum::<f32>() as f32 / avg.len() as f32,
        BATCH_SIZE
    );
    println!("{:?}", avg);
}
