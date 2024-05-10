use crate::{
    decoder::eval_state,
    mcts_trainer::Net,
    selfplay::CollectorMessage,
    superluminal::{CL_BLUE, CL_ORANGE, CL_RED},
};
use flume::{Receiver, RecvError, Selector, Sender};
use std::{
    cmp::min, collections::VecDeque, process, thread, time::{Duration, Instant, SystemTime, UNIX_EPOCH}
};
use superluminal_perf::{begin_event_with_color, end_event};
use tch::Tensor;

pub struct Packet {
    pub job: Tensor,
    pub resender: Sender<ReturnMessage>,
    pub id: String,
}

pub struct ReturnPacket {
    pub packet: (Tensor, Tensor),
    pub id: String,
}

pub enum Message {
    NewNetwork(Result<String, RecvError>),
    JobTensor(Result<Packet, RecvError>), // (converted) tensor from mcts search that needs NN evaluation

    StopServer, // end the executor process
}

pub enum ReturnMessage {
    ReturnMessage(Result<ReturnPacket, RecvError>),
}

fn handle_new_graph(network: &mut Option<Net>, graph: Option<String>, thread_name: &str) {
    // drop previous network if any to save GPU memory
    if let Some(network) = network.take() {
        // // // println!("{} dropping network", thread_name);
        drop(network);
    }

    // load the new network if any
    *network = graph.map(|graph| Net::new(&graph[..]));
}

pub fn executor_main(
    net_receiver: Receiver<String>,
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
    let mut id_vec = VecDeque::new(); // collect senders
                                      // // println!("num_threads (generator): {}", num_threads);

    let mut waiting_batch: Instant = Instant::now(); // time spent idling (total for each batch)
    let mut waiting_job = Instant::now(); // job timer
    let mut now_start = SystemTime::now(); // batch timer
    let since_epoch = now_start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let mut epoch_seconds_start = since_epoch.as_nanos(); // batch
    loop {
        begin_event_with_color("waiting_for_job", CL_ORANGE);
        let sw = Instant::now();
        // // println!("thread {} loop {}:", thread_name, debug_counter);
        // // println!("    thread {} number of requests made from mcts: {}, {}", thread_name, output_senders.len(), num_threads);
        // // println!("    thread {} graph_disconnected: {}", thread_name, graph_disconnected);
        assert!(network.is_some() || !graph_disconnected);

        let mut selector = Selector::new();

        if !graph_disconnected {
            selector = selector.recv(&net_receiver, |res| Message::NewNetwork(res));
        }

        // register all tensor receivers in the selector
        // println!("{}, {} {}, {}", input_vec.len(), thread_name, max_batch_size, has_started);
        // println!("{}",((input_vec.len() == max_batch_size) || (has_started == false)) || ((input_vec.len() < max_batch_size )));
        if input_vec.len() < max_batch_size {

            match network {
                Some(_) => {
                    selector = selector.recv(&tensor_receiver, |res| Message::JobTensor(res));
                }
                None => (),
            }
        } else {
        }

        let now_end = SystemTime::now();
        let since_epoch = now_end
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let epoch_seconds_start_job = since_epoch.as_nanos();
        let message = selector.wait();

        end_event();

        // println!("RECV SIZE {} NUM SENDERS {} RECV {}", tensor_receiver.len(), tensor_receiver.sender_count(), tensor_receiver.receiver_count());
        match message {
            Message::StopServer => break,
            Message::NewNetwork(Ok(graph)) => {
                // // println!("    NEW NET!");
                handle_new_graph(&mut network, Some(graph), &thread_name);
            }
            Message::JobTensor(job) => {
                // println!("EXEC ID {} CHANNEL_LEN {}", thread_name, tensor_receiver.len());
                let now_end = SystemTime::now();
                let since_epoch = now_end
                    .duration_since(UNIX_EPOCH)
                    .expect("Time went backwards");
                let epoch_seconds_end = since_epoch.as_nanos();
                // println!("{} {} {} waiting_for_job", epoch_seconds_start_job, epoch_seconds_end, thread_name);

                let job = job.expect("JobTensor should be available");

                input_vec.push_back(job.job);
                output_senders.push_back(job.resender);
                id_vec.push_back(job.id);
                // evaluate batches
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
                    let network = network.as_mut().expect("Network should be available");
                    let elapsed = waiting_batch.elapsed().as_nanos() as f32 / 1e6;

                    // println!("loop {} time taken for buffer to fill: {}ms", debug_counter, elapsed);
                    let sw_tensor_prep = Instant::now();
                    let batch_size = min(max_batch_size, input_vec.len());
                    let i_v = input_vec.make_contiguous();
                    let input_tensors = Tensor::cat(&i_v[..batch_size], 0);
                    let elapsed = sw_tensor_prep.elapsed().as_nanos() as f32 / 1e6;

                    // println!("loop {} prepping tensors: {}ms", debug_counter, elapsed);
                    // // println!("        thread {}: preparing tensors", thread_name);
                    // // println!("            thread {}: eval input tensors: {:?}", thread_name, input_tensors);
                    // // println!("        thread {}: NN evaluation:", thread_name);

                    let now_start_evals = SystemTime::now();
                    let since_epoch_evals = now_start_evals
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards");
                    let epoch_seconds_start_evals = since_epoch_evals.as_nanos();

                    begin_event_with_color("eval", CL_BLUE);
                    let start = Instant::now();
                    let (board_eval, policy) =
                        eval_state(input_tensors, network).expect("Evaluation failed");
                    let delta = start.elapsed().as_secs_f32();
                    // println!("Eval took {}, tp {}, batch_size {}, max_batch_size {}",delta, batch_size as f32 / delta, batch_size, max_batch_size);
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
                    // let evals_per_sec = batch_size as f32 / elapsed;

                    let _ = evals_per_sec_sender
                        .send(CollectorMessage::ExecutorStatistics(batch_size as f32));
                    // println!("inference_time {}s", elapsed);
                    // // println!("        thread {}: processing outputs:",thread_name);
                    // // println!("            thread {}: output tensors: {:?}, {:?}", thread_name, board_eval, policy);
                    // distribute results to the output senders
                    // // println!("        thread {}: sending tensors back to mcts:", thread_name);
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
                        // // println!("            thread {}, SENT! {:?}", i, &result);
                        let return_pack = ReturnPacket { packet: result, id };
                        sender
                            .send(ReturnMessage::ReturnMessage(Ok(return_pack)))
                            .expect("Should be able to send the result");
                        // let _ = sender.send(ReturnMessage::ReturnMessage(Ok(return_pack)));
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
                    // println!("loop {} packing time {}ms", debug_counter, packing_elapsed);
                    drop(input_vec.drain(0..batch_size));
                    waiting_batch = Instant::now();
                    now_start = SystemTime::now();
                    let since_epoch = now_start
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards");
                    epoch_seconds_start = since_epoch.as_nanos();
                    // begin_event_with_color("waiting_for_batch", CL_ORANGE);
                }
            }
            Message::NewNetwork(Err(RecvError::Disconnected)) => {
                // // println!("DISCONNECTED NET!");
                graph_disconnected = true;
                if network.is_none() && input_vec.is_empty() {
                    break; // exit if no network and no ongoing jobs
                }
            }
        }
        let elapsed = sw.elapsed().as_nanos() as f32 / 1e9;
        // println!("loop {}, elapsed time: {}s", debug_counter, elapsed);
        debug_counter += 1;
    }
    // Return the senders to avoid them being dropped and disconnected
    process::exit(0)
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

    handle_new_graph(&mut network, Some(net_path), thread_name.as_str());

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
        }
        let elapsed = sw.elapsed().as_nanos() as f32 / 1e9;
        debug_counter += 1;
    }
    // Return the senders to avoid them being dropped and disconnected
    process::exit(0)
}
