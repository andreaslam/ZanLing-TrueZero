use cozy_chess::{Board, Move};
use crossbeam::thread;
use flume::{Receiver, RecvError, Selector, Sender};
use futures::executor::ThreadPool;
use std::{
    cmp::min,
    collections::VecDeque,
    process,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use superluminal_perf::{begin_event_with_color, end_event};
use tch::Tensor;
use tz_rust::{
    boardmanager::BoardStack,
    dataformat::ZeroEvaluation,
    decoder::eval_state,
    executor::{handle_new_graph, Message, Packet, ReturnMessage, ReturnPacket},
    mcts_trainer::{Net, Tree, TypeRequest},
    selfplay::{CollectorMessage, DataGen},
    settings::SearchSettings,
    superluminal::{CL_BLUE, CL_ORANGE, CL_RED},
};

fn main() {
    let pool = ThreadPool::new().expect("Failed to build pool");
    let num_executors = 1;
    let batch_size = 4096;
    let num_generators = 16384;
    let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Packet>(num_generators); // mcts to executor
    let (game_sender, game_receiver) =
        flume::bounded::<CollectorMessage>(num_executors * batch_size);

    thread::scope(|s| {
        for exec_id in 0..num_executors {
            let eval_per_sec_sender = game_sender.clone();
            let tensor_exe_recv_clone = tensor_exe_recv.clone();
            s.builder()
                .name(format!("executor-{}", exec_id))
                .spawn(move |_| {
                    executor_main(tensor_exe_recv_clone, batch_size, eval_per_sec_sender)
                })
                .expect("Failed to spawn executor thread");
        }
        s.builder()
            .name("dummy-collector".to_string())
            .spawn(move |_| dummy_collector(&game_receiver, num_executors))
            .expect("Failed to spawn dummy collector thread");
        for n in 0..num_generators {
            let sender_clone = game_sender.clone();
            let selfplay_master = DataGen { iterations: 1 };
            let tensor_exe_send_clone = tensor_exe_send.clone();
            let fut_generator = async move {
                dummy_generator(sender_clone, selfplay_master, tensor_exe_send_clone, n).await;
            };
            pool.spawn_ok(fut_generator);
        }
    })
    .expect("Thread scope failed");
}

async fn dummy_generator(
    sender_collector: Sender<CollectorMessage>,
    datagen: DataGen,
    tensor_exe_send: Sender<Packet>,
    id: usize,
) {
    const BATCH_SIZE: usize = 1024;
    let thread_name = std::thread::current()
        .name()
        .unwrap_or("unnamed-generator")
        .to_owned();
    let input_tensor = Tensor::from_slice(&[0.1 as f32; 1344]); // 8*8*21 = 1344
    let (resender_send, resender_recv) = flume::unbounded::<ReturnMessage>(); // mcts to executor
    for n in 0..10 {
        // repeatedly sending the board position over and over again
        let pack = Packet {
            job: input_tensor.copy(),
            resender: resender_send.clone(),
            id: thread_name.clone(),
        };
        tensor_exe_send.send_async(pack).await.unwrap();
        // println!("YOOOO {} {}", n, id);
        resender_recv.recv_async().await.unwrap();
    }
    // FIXME: add a stop server message
    // benchmark consistent board evals
    // 1 exec vs 2 execs for cpu + gpu
}

fn dummy_collector(receiver: &Receiver<CollectorMessage>, num_executors: usize) {
    let mut counter = 0;

    let total_timer = Instant::now();

    while let Ok(_) = receiver.recv() {
        if counter == num_executors {
            let elapsed = total_timer.elapsed().as_nanos() as f32 / 1e9;
            println!("total time {}s", elapsed);
            process::exit(0);
        }
        counter += 1;
    }
}

pub fn executor_main(
    tensor_receiver: Receiver<Packet>, // receive tensors from mcts
    max_batch_size: usize,
    evals_per_sec_sender: Sender<CollectorMessage>,
) {
    let mut graph_disconnected = false;
    let mut network: Option<Net> = None;
    let thread_name = std::thread::current()
        .name()
        .unwrap_or("unnamed-executor")
        .to_owned();
    let mut input_vec: VecDeque<Tensor> = VecDeque::new();
    let mut debug_counter = 0;
    let mut output_senders = VecDeque::new(); // collect senders
    let mut id_vec = VecDeque::new(); // collect IDs

    let mut waiting_batch: Instant = Instant::now(); // time spent idling (total for each batch)
    let mut waiting_job = Instant::now(); // job timer
    let mut now_start = SystemTime::now(); // batch timer
    let since_epoch = now_start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let mut epoch_seconds_start = since_epoch.as_nanos(); // batch
    println!("{} {:?}", thread_name, tensor_receiver.capacity());
    let mut completed_loops = 0;
    let num_loops = 10;
    let mut network = Net::new("tz_6515.pt");
    let total_timer = Instant::now();
    loop {
        let mut selector = Selector::new();
        begin_event_with_color("waiting_for_job", CL_ORANGE);
        let sw = Instant::now();

        // register all tensor receivers in the selector
        selector = selector.recv(&tensor_receiver, |res| Message::JobTensor(res));

        let now_end = SystemTime::now();
        let since_epoch = now_end
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let epoch_seconds_start_job = since_epoch.as_nanos();
        let message = selector.wait();

        end_event();

        match message {
            Message::StopServer => break,
            Message::NewNetwork(Ok(graph)) => {}
            Message::JobTensor(job) => {
                let now_end = SystemTime::now();
                let since_epoch = now_end
                    .duration_since(UNIX_EPOCH)
                    .expect("Time went backwards");
                let epoch_seconds_end = since_epoch.as_nanos();

                let job = job.expect("JobTensor should be available");

                input_vec.push_back(job.job);
                output_senders.push_back(job.resender);
                id_vec.push_back(job.id);

                waiting_job = Instant::now();
                while input_vec.len() >= max_batch_size {
                    let now_end = SystemTime::now();
                    let since_epoch = now_end
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards");
                    let epoch_seconds_end = since_epoch.as_nanos();
                    println!(
                        "{} {} {} waiting_for_batch",
                        epoch_seconds_start, epoch_seconds_end, thread_name
                    );

                    let elapsed = waiting_batch.elapsed().as_nanos() as f32 / 1e6;

                    let sw_tensor_prep = Instant::now();
                    let batch_size = min(max_batch_size, input_vec.len());
                    let i_v = input_vec.make_contiguous();
                    let input_tensors = Tensor::cat(&i_v[..batch_size], 0);
                    let elapsed = sw_tensor_prep.elapsed().as_nanos() as f32 / 1e6;

                    let now_start_evals = SystemTime::now();
                    let since_epoch_evals = now_start_evals
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards");
                    let epoch_seconds_start_evals = since_epoch_evals.as_nanos();
                    begin_event_with_color("eval", CL_BLUE);
                    let start = Instant::now();
                    let (board_eval, policy) =
                        eval_state(input_tensors, &network).expect("Evaluation failed");
                    let delta = start.elapsed().as_secs_f32();
                    end_event();
                    let now_end_evals = SystemTime::now();
                    let since_epoch_evals = now_end_evals
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards");
                    let epoch_seconds_end_evals = since_epoch_evals.as_nanos();
                    println!(
                        "{} {} {} evaluation_time_taken",
                        epoch_seconds_start_evals, epoch_seconds_end_evals, thread_name
                    );

                    let sw_inference = Instant::now();
                    let elapsed = sw_inference.elapsed().as_nanos() as f32 / 1e9;

                    let packing_time = Instant::now();
                    let now_start_packing = SystemTime::now();
                    let since_epoch_packing = now_start_packing
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards");
                    let epoch_seconds_start_packing = since_epoch_packing.as_nanos();
                    begin_event_with_color("packing", CL_RED);
                    for i in 0..batch_size {
                        let sender = output_senders
                            .pop_front()
                            .expect("There should be a sender for each job");
                        let id = id_vec
                            .pop_front()
                            .expect("There should be an ID for each job");
                        let result = (board_eval.get(i as i64), policy.get(i as i64));
                        let return_pack = ReturnPacket { packet: result, id };
                        sender
                            .send(ReturnMessage::ReturnMessage(Ok(return_pack)))
                            .expect("Should be able to send the result");
                    }
                    end_event();
                    let now_end_packing = SystemTime::now();
                    let since_epoch_packing = now_end_packing
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards");
                    let epoch_seconds_end_packing = since_epoch_packing.as_nanos();

                    println!(
                        "{} {} {} packing_time",
                        epoch_seconds_start_packing, epoch_seconds_end_packing, thread_name
                    );

                    let packing_elapsed = packing_time.elapsed().as_nanos() as f32 / 1e6;
                    drop(input_vec.drain(0..batch_size));
                    waiting_batch = Instant::now();
                    now_start = SystemTime::now();
                    let since_epoch = now_start
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards");
                    epoch_seconds_start = since_epoch.as_nanos();
                    completed_loops += 1;
                }
            }
            Message::NewNetwork(Err(RecvError::Disconnected)) => {}
            Message::JobTensorExecutor(_) => {}
        }
        let elapsed = sw.elapsed().as_nanos() as f32 / 1e9;
        if completed_loops > num_loops {
            let elapsed = total_timer.elapsed().as_nanos() as f32 / 1e9;
            evals_per_sec_sender
                .send(CollectorMessage::ExecutorStatistics(1.0))
                .expect("Failed to send executor statistics");
        }
        debug_counter += 1;
    }
    process::exit(0);
}
