use cozy_chess::Board;
use flume::Sender;
use futures::executor::ThreadPool;
use lru::LruCache;
use rand::Rng;
use std::{
    collections::VecDeque,
    env,
    num::NonZeroUsize,
    thread,
    time::{Duration, Instant},
};
use tch::Tensor;
use tz_rust::debug_print;
use tz_rust::{
    boardmanager::BoardStack,
    decoder::eval_state,
    executor::{Packet, ReturnMessage, ReturnPacket},
    mcts_trainer::{get_move, EvalMode, Net, TypeRequest::TrainerSearch},
    settings::SearchSettings,
    utils::TimeStampDebugger,
};

fn main() {
    env::set_var("RUST_BACKTRACE", "full");
    const BATCH_SIZE: usize = 512;
    const NUM_THREADS: usize = 2;
    let num_generators = BATCH_SIZE * NUM_THREADS * 2;
    let entire_benchmark_timer = Instant::now();
    let pool = ThreadPool::builder()
        .pool_size(2048)
        .create()
        .expect("Failed to create ThreadPool");

    crossbeam::scope(|s| {
        let (tensor_sender, tensor_receiver) = flume::bounded::<Packet>(num_generators); // mcts to executor

        for id in 0..num_generators {
            let tensor_sender_clone = tensor_sender.clone();
            let fut_generator = async move { dummy_generator(tensor_sender_clone, id).await };
            pool.spawn_ok(fut_generator);
        }

        for i in 0..NUM_THREADS {
            let tensor_receiver_clone = tensor_receiver.clone();
            let _ = s.builder().name(format!("thread-{}", i)).spawn(move |_| {
                let net = Net::new("tz_6515.pt"); // Create a new instance of Net within the thread

                let mut eval_counter = 0;
                let mut one_sec_timer = Instant::now();
                let thread_name = thread::current()
                    .name()
                    .unwrap_or("unnamed-executor")
                    .to_owned();
                let mut input_vec: VecDeque<Tensor> = VecDeque::new();
                let mut sender_vec: VecDeque<Sender<ReturnMessage>> = VecDeque::new();
                let mut waiting_for_batch = TimeStampDebugger::create_debug();
                let mut id_vec: VecDeque<String> = VecDeque::new();
                let mut waiting_for_request = TimeStampDebugger::create_debug();
                loop {
                    match tensor_receiver_clone.recv() {
                        Ok(data_sample) => {
                            // waiting_for_request.record("recv_request", thread_name.as_str());
                            input_vec.push_back(data_sample.job.clone(&data_sample.job));
                            sender_vec.push_back(data_sample.resender.clone());
                            id_vec.push_back(data_sample.id.clone());
                            waiting_for_request.reset();
                        }
                        Err(err) => panic!("{}", err),
                    }

                    if input_vec.len() == BATCH_SIZE {
                        waiting_for_batch.record("waiting_for_batch", thread_name.as_str());
                        let data = &mut input_vec;
                        let evaluation_time_taken = TimeStampDebugger::create_debug();
                        let data = data.make_contiguous();
                        let data = Tensor::cat(&data, 0);
                        let (board_eval, policy) =
                            eval_state(data, &net).expect("Error evaluating state");
                        evaluation_time_taken.record("evaluation_time_taken", thread_name.as_str());

                        for i in 0..BATCH_SIZE {
                            let sender = sender_vec
                                .pop_front()
                                .expect("Failed to pop sender from queue");
                            let result = (board_eval.get(i as i64), policy.get(i as i64));
                            let packet = ReturnPacket {
                                packet: result,
                                id: id_vec.pop_front().expect("Failed to pop ID from queue"),
                            };
                            if let Err(e) = sender.send(ReturnMessage::ReturnMessage(Ok(packet))) {
                                eprintln!("Failed to send packet: {}", e);
                            } else {
                            }
                        }

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
                        input_vec.clear();
                    }
                }
            });
        }
    })
    .expect("Benchmark scope execution failed");

    let total_time_secs = entire_benchmark_timer.elapsed().as_nanos() as f32 / 1e9;
    debug_print!("{}", &format!("Benchmark ran for {}s", total_time_secs));
}

async fn dummy_generator(tensor_sender: Sender<Packet>, id: usize) {
    let board = Board::default();
    let bs = BoardStack::new(board);
    let settings: SearchSettings = SearchSettings {
        fpu: 0.0,
        wdl: EvalMode::Value,
        moves_left: None,
        c_puct: 3.0,
        max_nodes: 400,
        alpha: 0.3,
        eps: 0.3,
        search_type: TrainerSearch(None),
        pst: 1.3,
    };
    let mut cache = LruCache::new(NonZeroUsize::new(1000).unwrap());
    loop {
        let _ = get_move(
            bs.clone(),
            &tensor_sender.clone(),
            settings,
            id.clone(),
            &mut cache,
        )
        .await;
    }
}
