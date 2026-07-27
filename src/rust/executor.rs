use crate::{
    debug_print, decoder::eval_state, mcts_trainer::Net, selfplay::CollectorMessage,
    utils::TimeStampDebugger,
};

use crossbeam::thread;

use flume::{Receiver, RecvError, RecvTimeoutError, Selector, Sender};

use std::{
    cmp::min,
    collections::VecDeque,
    process,
    time::{Duration, Instant},
};
use tch::Tensor;

const PARTIAL_BATCH_TIMEOUT: Duration = Duration::from_micros(100);

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
    _thread_name: &str,
    id: usize,
) {
    debug_print!("{}: loading new net", _thread_name);

    if let Some(network) = network.take() {
        drop(network)
    }

    *network = graph.map(|graph| Net::new_with_device_id(&graph[..], id));
}

fn handle_requests(
    handler_send: Sender<ExecutorPacket>,
    tensor_receiver: Receiver<Packet>,
    max_batch_size: usize,
) {
    let mut input_vec: VecDeque<Tensor> = VecDeque::new();
    let mut output_senders: VecDeque<Sender<ReturnMessage>> = VecDeque::new();
    let mut id_vec: VecDeque<String> = VecDeque::new();
    // maximum time a partial batch may wait before being flushed to the GPU. Self-play
    // datagen fills its large batch well within this window, so this only triggers for
    // finite/low-concurrency workloads (e.g. the SPRT match tail), preventing the
    // partial-batch deadlock where the executor waits forever for a batch that never fills.
    let flush_batch = |input_vec: &mut VecDeque<Tensor>,
                       output_senders: &mut VecDeque<Sender<ReturnMessage>>,
                       id_vec: &mut VecDeque<String>| {
        if input_vec.is_empty() {
            return true;
        }
        let pack = ExecutorPacket {
            job: std::mem::take(input_vec),
            resenders: std::mem::take(output_senders),
            id: std::mem::take(id_vec),
        };
        handler_send.send(pack).is_ok()
    };

    loop {
        match tensor_receiver.recv_timeout(PARTIAL_BATCH_TIMEOUT) {
            Ok(job) => {
                input_vec.push_back(job.job);
                output_senders.push_back(job.resender);
                id_vec.push_back(job.id);
                debug_print!("Executor: received new request!");
                if input_vec.len() >= max_batch_size
                    && !flush_batch(&mut input_vec, &mut output_senders, &mut id_vec)
                {
                    break;
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if !flush_batch(&mut input_vec, &mut output_senders, &mut id_vec) {
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                let _ = flush_batch(&mut input_vec, &mut output_senders, &mut id_vec);
                break;
            }
        }
    }
}

pub fn executor_main(
    net_receiver: Receiver<String>,
    tensor_receiver: Receiver<Packet>,
    max_batch_size: usize,
    evals_per_sec_sender: Option<Sender<CollectorMessage>>,
    executor_ready_sender: Option<Sender<bool>>,
    executor_id: usize,
) {
    let mut graph_disconnected = false;
    let mut network: Option<Net> = None;
    let thread_name = std::thread::current()
        .name()
        .unwrap_or("unnamed-executor")
        .to_owned();

    let mut waiting_for_batch_debugger = TimeStampDebugger::create_debug();
    let mut packing_time_debugger = TimeStampDebugger::create_debug();
    let mut evaluation_time_taken_debugger = TimeStampDebugger::create_debug();

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
            // begin_event_with_color("waiting_for_job", CL_ORANGE);
            let _sw = Instant::now();
            assert!(network.is_some() || !graph_disconnected);

            if !graph_disconnected {
                selector = selector.recv(&net_receiver, Message::NewNetwork);
                waiting_for_batch_debugger.record("waiting_for_batch", &thread_name);
            }

            if let Some(_) = network {
                selector = selector.recv(&handler_recv, Message::JobTensorExecutor);
            }

            let message = selector.wait();
            // end_event();

            match message {
                Message::StopServer => break,
                Message::NewNetwork(Ok(graph)) => {
                    handle_new_graph(&mut network, Some(graph), &thread_name, executor_id);
                    match executor_ready_sender {
                        Some(ref sender) => sender.send(true).unwrap(),
                        None => {}
                    }
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
                    let input_tensors = Tensor::cat(i_v, 0);

                    // begin_event_with_color("eval", CL_BLUE);
                    evaluation_time_taken_debugger.reset();
                    let (board_eval, policy) =
                        eval_state(input_tensors, network).expect("Evaluation failed");
                    // end_event();
                    evaluation_time_taken_debugger.record("evaluation_time_taken", &thread_name);
                    evaluation_time_taken_debugger.reset();
                    match evals_per_sec_sender {
                        Some(ref sender) => {
                            sender
                                .send(CollectorMessage::ExecutorStatistics(batch_size))
                                .unwrap();
                        }
                        None => {}
                    }

                    // begin_event_with_color("packing", CL_RED);
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
                    // end_event();

                    packing_time_debugger.record("packing_time", &thread_name);

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

    let (handler_send, handler_recv) = flume::bounded::<ExecutorPacket>(1);

    handle_new_graph(&mut network, Some(net_path), thread_name.as_str(), 0); // TODO maybe pass an option to use either/both GPUs? and not hardcode static GPU to 1 GPU only?

    let _ = thread::scope(|s| {
        s.builder()
            .name(format!("executor-wait-{}", 0).to_string()) // TODO placeholder
            .spawn(move |_| {
                handle_requests(handler_send, tensor_receiver, max_batch_size);
            })
            .unwrap();

        loop {
            let mut selector = Selector::new();

            // register all receivers in the selector
            selector = selector.recv(&handler_recv, Message::JobTensorExecutor);
            selector = selector.recv(&ctrl_receiver, |_| Message::StopServer);
            let message = selector.wait();
            match message {
                Message::StopServer => {
                    break;
                }
                Message::NewNetwork(_) => {
                    unreachable!(); // handle new network message if needed
                }
                Message::JobTensor(_) => {}
                Message::JobTensorExecutor(job) => {
                    let job = job.expect("JobTensor should be available");
                    let network = network.as_mut().expect("Network should be available");
                    let mut input_vec = job.job;
                    let mut id_vec = job.id;
                    let mut output_senders = job.resenders;

                    let batch_size = min(max_batch_size, input_vec.len());
                    let i_v = input_vec.make_contiguous();
                    let input_tensors = Tensor::cat(&i_v[..batch_size], 0);
                    let (board_eval, policy) =
                        eval_state(input_tensors, network).expect("Evaluation failed");
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
        }
        // return the senders to avoid them being dropped and disconnected
    });
}
