use flume::Sender;
use futures::executor::ThreadPool;
use rand::Rng;
use std::time::{Duration, Instant};
use std::{collections::VecDeque, env, thread};
use tch::Tensor;
use tz_rust::{decoder::eval_state, mcts_trainer::Net, utils::TimeStampDebugger};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let pool = ThreadPool::builder().pool_size(6).create().unwrap();

    let batch_size: usize = 2048;
    let num_executors: usize = 2;
    let num_generators: usize = batch_size * num_executors * 2;

    crossbeam::scope(|s| {
        let (tensor_sender, tensor_receiver) = flume::bounded::<Vec<f32>>(num_generators);

        for id in 0..num_generators {
            let tensor_sender_clone = tensor_sender.clone();
            let fut_generator = async move { dummy_generator(tensor_sender_clone, id).await };
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

                    let mut input_vec: VecDeque<f32> = VecDeque::new();
                    let mut one_sec_timer = Instant::now();
                    let mut eval_counter = 0;

                    let mut debugger = TimeStampDebugger::create_debug();

                    loop {
                        let data = tensor_receiver_clone.recv().unwrap();

                        input_vec.extend(data.iter().copied());
                        if input_vec.len() == batch_size * 1344 {
                            debugger.record("waiting_for_batch", &thread_name);

                            let input_tensors = Tensor::from_slice(&input_vec.make_contiguous())
                                .reshape([-1, 1344]);
                            let eval_debugger = TimeStampDebugger::create_debug();
                            let _ = eval_state(input_tensors, &net).expect("Error");
                            eval_debugger.record("evaluation_time_taken", &thread_name);

                            input_vec.clear();

                            eval_counter += 1;
                            debugger.reset();
                        }
                        if one_sec_timer.elapsed() > Duration::from_secs(1) {
                            println!(
                                "{}: {}evals/s",
                                thread_name,
                                (eval_counter * batch_size) as f32
                                    / one_sec_timer.elapsed().as_secs_f32()
                            );
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

fn random_tensor(size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen::<f32>()).collect()
}

async fn dummy_generator(tensor_sender_clone: Sender<Vec<f32>>, id: usize) {
    let data = random_tensor(1344 * 1);
    let mut generator_debug = TimeStampDebugger::create_debug();
    loop {
        tensor_sender_clone.send_async(data.clone()).await.unwrap();
        generator_debug.reset();
    }
}
