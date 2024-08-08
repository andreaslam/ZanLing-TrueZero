use crossfire::{
    channel::MPMCShared,
    mpmc,
    mpmc::{TryRecvError, TxFuture},
};
use futures::executor::{block_on, ThreadPool};
use futures::{FutureExt, StreamExt};
use rand::Rng;
use std::{
    env, thread,
    time::{Duration, Instant},
};
use tch::Tensor;
use tzrust::debug_print;
use tzrust::{decoder::eval_state, mcts_trainer::Net, utils::TimeStampDebugger};
fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    const BATCH_SIZE: usize = 512;
    const NUM_THREADS: usize = 2;
    const NUM_LOOPS: usize = 100;
    const NUM_WARMUPS: usize = 100;
    let num_generators = 524288;
    let entire_benchmark_timer = Instant::now();
    let pool = ThreadPool::builder().create().unwrap();

    crossbeam::scope(|s| {
        let (tensor_sender, tensor_receiver) =
            mpmc::bounded_tx_future_rx_blocking::<Vec<f32>>(num_generators);
        let data_sample = random_tensor(1344);

        // Spawn async generators
        for id in 0..num_generators {
            let tensor_sender_clone = tensor_sender.clone();
            let fut_generator = dummy_generator(tensor_sender_clone, id);
            pool.spawn_ok(fut_generator);
        }

        // Spawn processing threads
        for i in 0..NUM_THREADS {
            let tensor_receiver_clone = tensor_receiver.clone();
            let _ = s.builder().name(format!("executor-{}", i)).spawn(move |_| {
                let net = Net::new("nano.pt");

                let mut eval_counter = 0;
                let mut one_sec_timer = Instant::now();
                let thread_name = thread::current()
                    .name()
                    .unwrap_or("unnamed-executor")
                    .to_owned();
                let mut input_vec: Vec<f32> = Vec::new();
                let mut waiting_for_request = TimeStampDebugger::create_debug();
                let mut waiting_for_batch = TimeStampDebugger::create_debug();

                loop {
                    let data_sample = tensor_receiver_clone.recv().unwrap();
                    input_vec.extend(data_sample.iter().copied());

                    while input_vec.len() < BATCH_SIZE * 1344 {
                        match tensor_receiver_clone.try_recv() {
                            Ok(data_sample) => {
                                waiting_for_request.reset();
                                input_vec.extend(data_sample.iter().copied());
                            }
                            Err(TryRecvError::Empty) => break,
                            Err(TryRecvError::Disconnected) => panic!("NOOOOOOO"),
                        }
                    }
                    if input_vec.len() == BATCH_SIZE * 1344 {
                        waiting_for_batch.record("waiting_for_batch", thread_name.as_str());
                        let data = &input_vec;
                        let now = Instant::now();
                        let evaluation_time_taken = TimeStampDebugger::create_debug();
                        let _ = eval_state(Tensor::from_slice(&data), &net).expect("Error");
                        evaluation_time_taken.record("evaluation_time_taken", thread_name.as_str());
                        let elapsed = (now.elapsed().as_nanos() as f32) / (1e9 as f32);
                        input_vec.clear();
                        if one_sec_timer.elapsed() > Duration::from_secs(1) {
                            debug_print!(
                                "{}",
                                &format!(
                                    "{}: {}evals/s",
                                    thread_name,
                                    (eval_counter * BATCH_SIZE) as f32
                                        / one_sec_timer.elapsed().as_secs_f32()
                                )
                            );
                            eval_counter = 0;
                            one_sec_timer = Instant::now();
                        }
                        eval_counter += 1;
                        waiting_for_batch.reset();
                    }
                }
            });
        }
    })
    .unwrap();

    let total_time_secs = entire_benchmark_timer.elapsed().as_nanos() as f32 / 1e9;
    debug_print!("{}", &format!("Benchmark ran for {}s", total_time_secs));
}

fn random_tensor(size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen::<f32>()).collect()
}

async fn dummy_generator<S: MPMCShared>(tensor_sender: TxFuture<Vec<f32>, S>, id: usize) {
    let data_sample = random_tensor(1344 * 1);
    let mut one_sec_timer = Instant::now();
    let thread_name = format!("thread-{}", id);
    let mut loop_sender = 0;
    loop {
        tensor_sender.send(data_sample.clone()).await.unwrap();
        if id % 1 == 0 {
            if one_sec_timer.elapsed() > Duration::from_secs(1) {
                debug_print!("{}", &format!("{}: {} req", thread_name, loop_sender));
                loop_sender = 0;
                one_sec_timer = Instant::now();
            }
        }
        loop_sender += 1;
        // thread::sleep(Duration::from_nanos(600)); // simulate mcts
    }
}
