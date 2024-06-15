use crate::{
    decoder::eval_state,
    mcts_trainer::Net,
    selfplay::CollectorMessage,
    superluminal::{CL_BLUE, CL_ORANGE, CL_RED},
};
use crossbeam::thread;
use flume::{Receiver, RecvError, Selector, Sender};
use std::{
    cmp::min,
    collections::VecDeque,
    process,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use superluminal_perf::{begin_event_with_color, end_event};
use tch::{Cuda, Tensor};

pub struct ExecutorDebugger {
    epoch_seconds_start: u128,
}

impl ExecutorDebugger {
    pub fn create_debug() -> Self {
        let now_start = SystemTime::now();
        let since_epoch = now_start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let epoch_seconds_start = since_epoch.as_nanos();
        ExecutorDebugger {
            epoch_seconds_start,
        }
    }

    pub fn reset(&mut self) {
        let now_start = SystemTime::now();
        let since_epoch = now_start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        self.epoch_seconds_start = since_epoch.as_nanos();
    }

    pub fn record(&self, message: &str, thread_name: &str) {
        let now_end = SystemTime::now();
        let since_epoch_end = now_end
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let epoch_seconds_end = since_epoch_end.as_nanos();
        println!(
            "{} {} {} {}",
            self.epoch_seconds_start, epoch_seconds_end, thread_name, message
        );
    }
}

pub struct Packet {
    pub job: Tensor,
    pub resender: Sender<ReturnMessage>,
    pub id: String,
}

pub struct ExecutorPacket {
    pub job: VecDeque<Tensor>,
    pub resenders: VecDeque<Sender<ReturnMessage>>,
    pub id: VecDeque<String>,
}

pub struct ReturnPacket {
    pub packet: (Tensor, Tensor),
    pub id: String,
}

pub enum Message {
    NewNetwork(Result<String, RecvError>),
    JobTensor(Result<Packet, RecvError>), // (converted) tensor from mcts search that needs NN evaluation
    JobTensorExecutor(Result<ExecutorPacket, RecvError>), // (converted) tensor from mcts search that needs NN evaluation

    StopServer, // end the executor process
}

pub enum ReturnMessage {
    ReturnMessage(Result<ReturnPacket, RecvError>),
}

fn handle_new_graph(
    network: &mut Option<Net>,
    graph: Option<String>,
    thread_name: &str,
    id: usize,
) {
    if let Some(network) = network.take() {
        drop(network);
    }
    *network = graph.map(|graph| Net::new_with_device_id(&graph[..], id));
}

fn handle_requests(
    handler_send: Sender<ExecutorPacket>,
    tensor_receiver: Receiver<Packet>,
    max_batch_size: usize,
) {
    let thread_name = std::thread::current()
        .name()
        .unwrap_or("unnamed-executor")
        .to_owned();
    let mut input_vec: VecDeque<Tensor> = VecDeque::new();
    let mut output_senders: VecDeque<Sender<ReturnMessage>> = VecDeque::new();
    let mut id_vec: VecDeque<String> = VecDeque::new();
    loop {
        let job = tensor_receiver.recv().unwrap();
        input_vec.push_back(job.job);
        output_senders.push_back(job.resender);
        id_vec.push_back(job.id);

        if input_vec.len() >= max_batch_size {
            let pack = ExecutorPacket {
                job: std::mem::take(&mut input_vec),
                resenders: std::mem::take(&mut output_senders),
                id: std::mem::take(&mut id_vec),
            };
            handler_send.send(pack).unwrap();
        }
    }
}

pub fn executor_main(
    net_receiver: Receiver<String>,
    tensor_receiver: Receiver<Packet>,
    max_batch_size: usize,
    evals_per_sec_sender: Sender<CollectorMessage>,
    executor_id: usize,
) {
    let mut graph_disconnected = false;
    let mut network: Option<Net> = None;
    let thread_name = std::thread::current()
        .name()
        .unwrap_or("unnamed-executor")
        .to_owned();

    let mut waiting_for_batch_debugger = ExecutorDebugger::create_debug();
    let mut packing_time_debugger = ExecutorDebugger::create_debug();
    let mut evaluation_time_taken_debugger = ExecutorDebugger::create_debug();

    let (handler_send, handler_recv) = flume::bounded::<ExecutorPacket>(1);

    thread::scope(|s| {
        let _ = s
            .builder()
            .name(format!("executor-wait-{}", executor_id).to_string())
            .spawn(move |_| {
                handle_requests(handler_send, tensor_receiver, max_batch_size);
            })
            .unwrap();

        loop {
            let mut selector = Selector::new();
            begin_event_with_color("waiting_for_job", CL_ORANGE);
            let sw = Instant::now();
            assert!(network.is_some() || !graph_disconnected);

            if !graph_disconnected {
                selector = selector.recv(&net_receiver, |res| Message::NewNetwork(res));
                // waiting_for_batch_debugger.record("waiting_for_batch", &thread_name);
            }

             match network {
                Some(_) => {

                    selector = selector.recv(&handler_recv, |res| Message::JobTensorExecutor(res));
                }
                None => (),
            }

            let message = selector.wait();
            end_event();

            match message {
                Message::StopServer => break,
                Message::NewNetwork(Ok(graph)) => {
                    handle_new_graph(&mut network, Some(graph), &thread_name, executor_id);
                }
                Message::JobTensor(_) => {}
                Message::JobTensorExecutor(job) => {
                    let job = job.expect("JobTensor should be available");
                    let mut input_vec = job.job;
                    let mut id_vec = job.id;
                    let mut output_senders = job.resenders;

                    let network = network.as_mut().expect("Network should be available");
                    let batch_size = input_vec.len();
                    let i_v = input_vec.make_contiguous();
                    let input_tensors = Tensor::cat(&i_v, 0);

                    let now_start_evals = SystemTime::now();
                    let since_epoch_evals = now_start_evals
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards");
                    let epoch_seconds_start_evals = since_epoch_evals.as_nanos();
                    begin_event_with_color("eval", CL_BLUE);
                    evaluation_time_taken_debugger.reset();
                    let (board_eval, policy) =
                        eval_state(input_tensors, network).expect("Evaluation failed");
                    end_event();
                    // evaluation_time_taken_debugger
                    //     .record("evaluation_time_taken", &thread_name);
                    evaluation_time_taken_debugger.reset();
                    let sw_inference = Instant::now();
                    let elapsed = sw_inference.elapsed().as_nanos() as f32 / 1e9;
                    let evals_per_sec = batch_size as f32 / elapsed;
                    let _ = evals_per_sec_sender
                        .send(CollectorMessage::ExecutorStatistics(batch_size));

                    begin_event_with_color("packing", CL_RED);
                    packing_time_debugger.reset();
                    for i in 0..batch_size {
                        let sender: Sender<ReturnMessage> = output_senders
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
                    // packing_time_debugger.record("packing_time", &thread_name);

                    drop(input_vec.drain(0..batch_size));
                    waiting_for_batch_debugger.reset();
                    packing_time_debugger.reset();
                }
                Message::NewNetwork(Err(RecvError::Disconnected)) => {
                    graph_disconnected = true;
                    if network.is_none() {
                        break;
                    }
                }
            }

            let elapsed = sw.elapsed().as_nanos() as f32 / 1e9;
        }

        process::exit(0)
    })
    .unwrap();
}

pub fn executor_static(
    net_path: String,
    tensor_receiver: Receiver<Packet>, // receive tensors from mcts
    ctrl_receiver: Receiver<Message>,  // receive control messages
    num_threads: usize,
) {
    let max_batch_size = min(1024, num_threads);
    let mut network: Option<Net> = None;
    let thread_name = std::thread::current()
        .name()
        .unwrap_or("unnamed-executor")
        .to_owned();
    let mut input_vec: VecDeque<Tensor> = VecDeque::new();
    let mut debug_counter = 0;
    let mut output_senders: VecDeque<Sender<ReturnMessage>> = VecDeque::new();
    let mut id_vec: VecDeque<String> = VecDeque::new();

    handle_new_graph(&mut network, Some(net_path), thread_name.as_str(), 0); // TODO maybe pass an option to use either/both GPUs? and not hardcode static GPU to 1 GPU only?

    loop {
        let sw = Instant::now();

        let mut selector = Selector::new();

        // Register all receivers in the selector
        selector = selector.recv(&tensor_receiver, |res| Message::JobTensor(res));
        selector = selector.recv(&ctrl_receiver, |_| Message::StopServer);
        let message = selector.wait();

        match message {
            Message::StopServer => {
                break;
            }
            Message::NewNetwork(_) => {
                unreachable!(); // Handle new network message if needed
            }
            Message::JobTensor(job) => {
                let job = job.expect("JobTensor should be available");
                let network = network.as_mut().expect("Network should be available");

                input_vec.push_back(job.job);
                output_senders.push_back(job.resender);
                id_vec.push_back(job.id);

                while input_vec.len() >= max_batch_size {
                    let batch_size = min(max_batch_size, input_vec.len());
                    let i_v = input_vec.make_contiguous();
                    let input_tensors = Tensor::cat(&i_v[..batch_size], 0);

                    let sw_inference = Instant::now();
                    let (board_eval, policy) =
                        eval_state(input_tensors, network).expect("Evaluation failed");
                    let elapsed = sw_inference.elapsed().as_nanos() as f32 / 1e9;
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
                    drop(input_vec.drain(0..batch_size));
                }
            }
            Message::JobTensorExecutor(_) => {}
        }
        let elapsed = sw.elapsed().as_nanos() as f32 / 1e9;
        debug_counter += 1;
    }
    // Return the senders to avoid them being dropped and disconnected
}
