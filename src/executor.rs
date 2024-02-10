use crate::{decoder::eval_state, mcts_trainer::Net, selfplay::CollectorMessage};
use flume::{Receiver, RecvError, Selector, Sender};
use std::{
    cmp::{max, min},
    collections::VecDeque,
    time::Instant,
};
use tch::Tensor;

pub struct Packet {
    pub job: Tensor,
    pub resender: Sender<ReturnMessage>,
    pub id: String,
}

struct CacheEntry {
    position: Tensor,
    output: (Tensor, Tensor),
}

impl CacheEntry {
    fn new(position: Tensor, output: (Tensor, Tensor)) -> Self {
        Self { position, output }
    }
}

pub struct ReturnPacket {
    pub packet: (Tensor, Tensor),
    pub id: String,
}

pub enum Message {
    NewNetwork(Result<String, RecvError>),
    JobTensor(Result<Packet, RecvError>), // (converted) tensor from mcts search that needs NN evaluation

    StopServer(), // end the executor process
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
    batch_size: usize,
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
    let mut cache: Vec<CacheEntry> = Vec::new();
    let mut cache_hits = 0;
    let mut requests = 0;
    loop {
        let sw = Instant::now();
        // println!("thread {} loop {}:", thread_name, debug_counter);

        // // println!(
        //     "    thread {} number of requests made from mcts: {}, {}",
        //     thread_name,
        //     output_senders.len(),
        //     batch_size
        // );
        // // println!(
        //     "    thread {} waiting graph_disconnected: {}",
        //     thread_name, graph_disconnected
        // );
        assert!(network.is_some() || !graph_disconnected);

        let mut selector = Selector::new();

        if !graph_disconnected {
            selector = selector.recv(&net_receiver, |res| Message::NewNetwork(res));
        }

        // register all tensor receivers in the selector
        match network {
            Some(_) => {
                selector = selector.recv(&tensor_receiver, |res| Message::JobTensor(res));
            }
            None => (),
        }

        let message = selector.wait();
        match message {
            Message::StopServer() => break,
            Message::NewNetwork(Ok(graph)) => {
                // println!("    NEW NET!");
                handle_new_graph(&mut network, Some(graph), &thread_name);
                cache = Vec::new();
            }
            Message::JobTensor(job) => {
                requests += 1;
                // // println!("    received job");
                let job = job.expect("JobTensor should be available");
                let network = network.as_mut().expect("Network should be available");
                let is_contained = cache
                    .iter()
                    .any(|cache_entry| cache_entry.position == job.job);
                let cache_idx = cache
                    .iter()
                    .position(|cache_entry| cache_entry.position == job.job);
                input_vec.push_back(job.job);
                output_senders.push_back(job.resender);
                id_vec.push_back(job.id);
                // // println!(
                //     "    received job, {}, batch_size {}",
                //     input_vec.len(),
                //     batch_size
                // );
                // evaluate batches

                if is_contained {
                    let sender = output_senders
                        .pop_back()
                        .expect("There should be a sender for each job");
                    let id = id_vec
                        .pop_back()
                        .expect("There should be an ID for each job");
                    let value = &cache[cache_idx.unwrap()].output.0;
                    let policy = &cache[cache_idx.unwrap()].output.1;
                    // // println!("            thread {}, SENT! {:?}", i, &result);
                    // // println!("result {:?}", (value, policy));
                    let return_pack = ReturnPacket {
                        packet: (value.clone(value), policy.clone(policy)),
                        id,
                    };
                    sender
                        .send(ReturnMessage::ReturnMessage(Ok(return_pack)))
                        .expect("Should be able to send the result");
                    input_vec.pop_back();
                    cache_hits += 1;
                    println!(
                        "cache size {} cache hits {}, total requests {}, cache satisfaction {:.3}%",
                        cache.len(),
                        cache_hits,
                        requests,
                        ((cache_hits as f32) / (requests as f32) * 100.0)
                    );
                } else {
                    while input_vec.len() >= batch_size {
                        let waiting_time = sw.elapsed().as_nanos() as f32 / 1e9; // waiting time in seconds
                        println!(
                            "        thread name {}, waiting time :{}s",
                            thread_name, waiting_time
                        );
                        let batch_size = min(batch_size, input_vec.len());
                        let i_v = input_vec.make_contiguous();
                        let input_tensors = Tensor::cat(&i_v[..batch_size], 0);
                        // // println!("        thread {}: preparing tensors", thread_name);
                        // // println!(
                        // "            thread {}: eval input tensors: {:?}",
                        // thread_name, input_tensors
                        // );
                        println!("        thread {}: NN evaluation:", thread_name);
                        let sw_inference = Instant::now();
                        let (board_eval, policy) =
                            eval_state(input_tensors.clone(&input_tensors), network)
                                .expect("Evaluation failed");

                        let elapsed = sw_inference.elapsed().as_nanos() as f32 / 1e9;
                        println!("        inference time: {}s", elapsed);
                        // let evals_per_sec = batch_size as f32 / elapsed;
                        evals_per_sec_sender
                            .send(CollectorMessage::ExecutorStatistics(batch_size as f32))
                            .unwrap();
                        // println!("sent to server");
                        // // println!(
                        //     "            thread {}: NN evaluation done! {}s",
                        //     thread_name, elapsed
                        // );
                        // // println!("        thread {}: processing outputs:", thread_name);
                        // // println!(
                        //     "            thread {}: output tensors: {:?}, {:?}",
                        //     thread_name, board_eval, policy
                        // );
                        // distribute results to the output senders
                        // // println!(
                        //     "        thread {}: sending tensors back to mcts:",
                        //     thread_name
                        // );

                        let sending_outputs_sw = Instant::now();

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
                            let (board_eval, policy) =
                                (board_eval.get(i as i64), policy.get(i as i64));
                            if (cache.iter().any(|cache_entry| {
                                cache_entry.output
                                    == (board_eval.clone(&board_eval), policy.clone(&policy))
                            })) == false
                            {
                                // println!("added to cache {:?}", input_tensors.reshape([-1,1344]).get(i as i64));
                                cache.push(CacheEntry::new(
                                    input_tensors.reshape([-1, 1344]).get(i as i64),
                                    (board_eval.clone(&board_eval), policy.clone(&policy)),
                                ));
                                // println!("progress {}, cache len {}", i, cache.len());
                            }
                        }
                        let elapsed = sending_outputs_sw.elapsed().as_nanos() as f32 / 1e9;
                        println!("sent all outputs back to mcts {}", elapsed);
                        drop(input_vec.drain(0..batch_size));
                    }
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
        println!("thread {}, elapsed time: {}s", thread_name, elapsed);
        debug_counter += 1;
    }
    // Return the senders to avoid them being dropped and disconnected
}

pub fn executor_static(
    net_path: String,
    tensor_receiver: Receiver<Packet>, // receive tensors from mcts
    ctrl_receiver: Receiver<Message>,  // receive control messages
    num_threads: usize,
) {
    let max_batch_size = min(256, num_threads);
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
        selector = selector.recv(&ctrl_receiver, |_| Message::StopServer());
        let message = selector.wait();

        match message {
            Message::StopServer() => break,
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
                    let waiting_time = sw.elapsed().as_nanos() as f32 / 1e9; // waiting time in seconds
                                                                             // // println!(
                                                                             //     "thread name {}, waiting time :{}s",
                                                                             //     thread_name, waiting_time
                                                                             // );
                    let batch_size = min(max_batch_size, input_vec.len());
                    let i_v = input_vec.make_contiguous();
                    let input_tensors = Tensor::cat(&i_v[..batch_size], 0);

                    let sw_inference = Instant::now();
                    let (board_eval, policy) =
                        eval_state(input_tensors, network).expect("Evaluation failed");
                    let elapsed = sw_inference.elapsed().as_nanos() as f32 / 1e9;
                    // println!("elapsed: {}", elapsed);
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
}
