use std::{env, time::Instant};
use tch::Tensor;
use tz_rust::{decoder::eval_state, mcts_trainer::Net};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    // let net = Net::new("chess_16x128_gen3634.pt");
    let net = Net::new("nets/tz_0.pt");
    let mut avg = Vec::new();
    const BATCH_SIZE: usize = 1024;

    // warmup loop

    for _ in 0..100 {
        let data = Tensor::from_slice(&[0.1 as f32; 1344 * BATCH_SIZE]); // 8*8*21 = 1344
        (_, _) = eval_state(data, &net).expect("Error");
    }

    // timed, benchmarked loop

    for _ in 0..100 {
        let data = Tensor::from_slice(&[0.1 as f32; 1344 * BATCH_SIZE]); // 8*8*21 = 1344
        let sw = Instant::now();
        (_, _) = eval_state(data, &net).expect("Error");
        println!("Elapsed time: {}s", sw.elapsed().as_nanos() as f32 / 1e9);
        let eps = BATCH_SIZE as f32 / (sw.elapsed().as_nanos() as f32 / 1e9);
        println!("Evaluations per second: {} evals/s", eps);
        avg.push(eps);
    }
    println!(
        "Average evaluations per second: {} evals/s (batch size {})",
        avg.iter().sum::<f32>() / avg.len() as f32,
        BATCH_SIZE
    );
    println!("{:?}", avg);
}
