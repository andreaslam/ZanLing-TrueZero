use std::{env, time::Instant};
use tch::Tensor;
use tzrust::{decoder::eval_state, mcts_trainer::Net};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let net = Net::new("tz_6515.pt");
    let mut avg = Vec::new();
    const BATCH_SIZE: usize = 1024;

    // warmup loop

    for _ in 0..10 {
        let data = Tensor::from_slice(&[0.0 as f32; 1344 * BATCH_SIZE]); // 8*8*21 = 1344
        (_, _) = eval_state(data, &net).expect("Error");
    }

    // timed, benchmarked loop

    for _ in 0..100 {
        let data = Tensor::from_slice(&[0.0 as f32; 1344 * BATCH_SIZE]); // 8*8*21 = 1344
        let sw = Instant::now();
        (_, _) = eval_state(data, &net).expect("Error");
        let elapsed_time_ms = sw.elapsed().as_nanos() as f32 / 1e6;
        println!("Elapsed time: {}ms", elapsed_time_ms);
        let eps = BATCH_SIZE.clone() as f32 / (elapsed_time_ms as f32 / 1e3);
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
