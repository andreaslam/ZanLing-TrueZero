use flume::Sender;
use futures::executor::ThreadPool;
use rand::Rng;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{collections::VecDeque, env, thread};
use tch::Tensor;
use tz_rust::{decoder::eval_state, executor::ExecutorDebugger, mcts_trainer::Net};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let pool = ThreadPool::new().expect("Failed to build pool");

    let batch_size: usize = 256;
    let num_executors: usize = 2;
    let num_generators: usize = batch_size * num_executors * 4;

    crossbeam::scope(|s| {
        let (tensor_sender, tensor_receiver) = flume::bounded::<Tensor>(num_generators);

        for _ in 0..num_generators {
            let tensor_sender_clone = tensor_sender.clone();
            let fut_generator = async move { dummy_generator(tensor_sender_clone).await };
            pool.spawn_ok(fut_generator);
        }

        for i in 0..num_executors {
            let tensor_receiver_clone = tensor_receiver.clone();
            s.builder()
                .name(format!("executor-{}", i))
                .spawn(move |_| {
                    let net = Net::new("tz_6515.pt");

                    let thread_name = thread::current()
                        .name()
                        .unwrap_or("unnamed-executor")
                        .to_owned();

                    let mut input_vec: VecDeque<Tensor> = VecDeque::new();
                    let mut one_sec_timer = Instant::now();
                    let mut eval_counter = 0;

                    let mut debugger = ExecutorDebugger::create_debug();

                    loop {
                        let data = tensor_receiver_clone.recv().unwrap();
                        input_vec.push_back(data);
                        if input_vec.len() == batch_size {
                            debugger.record("waiting_for_batch", &thread_name);

                            let input_tensors = Tensor::cat(&input_vec.make_contiguous(), 0);

                            let eval_debugger = ExecutorDebugger::create_debug();
                            let _ = eval_state(input_tensors, &net).expect("Error");
                            eval_debugger.record("evaluation_time_taken", &thread_name);

                            input_vec.clear();

                            eval_counter += 1;
                            debugger.reset();
                        }
                        if one_sec_timer.elapsed() > Duration::from_secs(1) {
                            eval_counter = 0;
                            one_sec_timer = Instant::now();
                        }
                    }
                })
                .unwrap();
        }
    })
    .unwrap();
}

fn random_tensor(size: usize) -> Tensor {
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..size).map(|_| rng.gen::<f32>()).collect();
    Tensor::from_slice(&data)
}

async fn dummy_generator(tensor_sender_clone: Sender<Tensor>) {
    loop {
        let data = random_tensor(1344 * 1);
        tensor_sender_clone.send_async(data).await.unwrap();
    }
}
