use futures::executor::ThreadPool;
use rand::Rng;
use std::{
    env, thread,
    time::{Duration, Instant},
};
use crossfire::{channel::MPMCShared, mpmc, mpmc::{TxFuture, TryRecvError}};
use tch::Tensor;
use tz_rust::{decoder::eval_state, mcts_trainer::Net};
fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    const BATCH_SIZE: usize = 512;
    const NUM_THREADS: usize = 1; // Number of threads to spawn
    const NUM_LOOPS: usize = 100;
    const NUM_WARMUPS: usize = 100;
    let num_generators = BATCH_SIZE * NUM_THREADS * 4;
    let entire_benchmark_timer = Instant::now();
    let pool = ThreadPool::builder().pool_size(6).create().unwrap();
    crossbeam::scope(|s| {
        let (tensor_sender, tensor_receiver) = mpmc::bounded_tx_future_rx_blocking::<Vec<f32>>(num_generators);
        let data_sample = random_tensor(1344); // 8*8*21 = 1344

        for _ in 0..num_generators {
            let tensor_sender_clone = tensor_sender.clone();
            let fut_generator = async move { dummy_generator(tensor_sender_clone).await };
            pool.spawn_ok(fut_generator);
        }

        for i in 0..NUM_THREADS {
            let tensor_receiver_clone = tensor_receiver.clone();
            let _ = s.builder().name(format!("thread-{}", i)).spawn(move |_| {
                let net = Net::new("tz_6515.pt"); // Create a new instance of Net within the thread

                // Timed, benchmarked loop
                let mut eval_counter = 0;
                let mut one_sec_timer = Instant::now();
                let thread_name = thread::current()
                    .name()
                    .unwrap_or("unnamed-executor")
                    .to_owned();
                let mut input_vec: Vec<f32> = Vec::new();

                loop {
                    let data_sample = tensor_receiver_clone.recv().unwrap();
                    input_vec.extend(data_sample.iter().copied());

                    while input_vec.len() < BATCH_SIZE * 1344 {

                        match tensor_receiver_clone.try_recv() {
                            Ok(data_sample) => input_vec.extend(data_sample.iter().copied()),
                            Err(TryRecvError::Empty) => break,
                            Err(TryRecvError::Disconnected) => panic!("NOOOOOOO"),
                        }
                    }
                    if input_vec.len() == BATCH_SIZE * 1344 {
                        let data = &input_vec;
                        let now = Instant::now();
                        let _ = eval_state(Tensor::from_slice(&data), &net).expect("Error");
                        let elapsed = (now.elapsed().as_nanos() as f32) / (1e9 as f32);
                        input_vec.clear();
                        if one_sec_timer.elapsed() > Duration::from_secs(1) {
                            println!(
                                "{}: {}evals/s",
                                thread_name,
                                (eval_counter * BATCH_SIZE) as f32
                                    / one_sec_timer.elapsed().as_secs_f32()
                            );
                            eval_counter = 0;
                            one_sec_timer = Instant::now();
                        }
                        eval_counter += 1;
                    }
                }
            });
        }
    })
    .unwrap();

    let total_time_secs = entire_benchmark_timer.elapsed().as_nanos() as f32 / 1e9;

    println!("Benchmark ran for {}s", total_time_secs);
}

fn random_tensor(size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen::<f32>()).collect()
}
async fn dummy_generator<S: MPMCShared>(tensor_sender: TxFuture<Vec<f32>, S>) {
    let data_sample = random_tensor(1344 * 1);
    loop {
        tensor_sender.send(data_sample.clone()).await.unwrap();
    }
}
