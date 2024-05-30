use std::{env, time::Instant};
use tch::Tensor;
use tz_rust::{decoder::eval_state, mcts_trainer::Net};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    const BATCH_SIZE: usize = 2048;
    const NUM_THREADS: usize = 1;
    const NUM_LOOPS: usize = 100;
    const NUM_WARMUPS: usize = 0;
    let entire_benchmark_timer = Instant::now();

    crossbeam::scope(|s| {
        for i in 0..NUM_THREADS {
            let _ = s.builder().name(format!("thread-{}", i + 1)).spawn(move |_| {
                let net = Net::new("tz_6515.pt");

                // Warmup loop
                for _ in 0..NUM_WARMUPS {
                    let data = Tensor::from_slice(&[0.1 as f32; 1344 * BATCH_SIZE]);
                    let _ = eval_state(data, &net).expect("Error");
                }

                // Timed, benchmarked loop
                let full_run = Instant::now();
                for _ in 0..NUM_LOOPS {
                    let data = Tensor::from_slice(&[0.1 as f32; 1344 * BATCH_SIZE]);
                    let _ = eval_state(data, &net).expect("Error");
                }

                let total_time_secs = full_run.elapsed().as_nanos() as f32 / 1e9;
                let total_evals = (BATCH_SIZE * NUM_LOOPS) as f32;
                let evals_per_sec = total_evals / total_time_secs;

                println!(
                    "Thread {}: Evaluations per second: {} evals/s (batch size {})",
                    i + 1, evals_per_sec, BATCH_SIZE
                );
                println!(
                    "Thread {}: Total time taken: {}s",
                    i + 1, total_time_secs
                );
            });
        }
    }).unwrap();

    let total_time_secs = entire_benchmark_timer.elapsed().as_nanos() as f32 / 1e9;

    println!(
        "Benchmark ran for {}s",
        total_time_secs
    );
}
