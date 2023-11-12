use crate::{decoder::eval_state, mcts_trainer::Net, selfplay::CollectorMessage};
use flume::{Receiver, RecvError, Selector, Sender};
use std::{cmp::min, collections::VecDeque, time::Instant};
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
}

pub enum ReturnMessage {
    ReturnMessage(Result<ReturnPacket, RecvError>),
}

fn handle_new_graph(network: &mut Option<Net>, graph: Option<String>, thread_name: &str) {
    // drop previous network if any to save GPU memory
    if let Some(network) = network.take() {
        // // println!("{} dropping network", thread_name);
        drop(network);
    }

    // load the new network if any
    *network = graph.map(|graph| Net::new(&graph[..]));
}

pub fn executor_main(
    net_receiver: Receiver<String>,
    tensor_receiver: Receiver<Packet>, // receive tensors from mcts
    num_threads: usize,
    evals_per_sec_sender: Sender<CollectorMessage>,
) {
    let max_batch_size = min(256, num_threads);
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
                                      // println!("num_threads (generator): {}", num_threads);

    loop {
        let sw = Instant::now();
        // println!("thread {} loop {}:", thread_name, debug_counter);
        // println!("    thread {} number of requests made from mcts: {}, {}", thread_name, output_senders.len(), num_threads);
        // println!("    thread {} graph_disconnected: {}", thread_name, graph_disconnected);
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
            Message::NewNetwork(Ok(graph)) => {
                // println!("    NEW NET!");
                handle_new_graph(&mut network, Some(graph), &thread_name);
            }
            Message::JobTensor(job) => {
                let job = job.expect("JobTensor should be available");
                let network = network.as_mut().expect("Network should be available");

                input_vec.push_back(job.job);
                output_senders.push_back(job.resender);
                id_vec.push_back(job.id);
                // evaluate batches
                while input_vec.len() >= max_batch_size {
                    let batch_size = min(max_batch_size, input_vec.len());
                    let i_v = input_vec.make_contiguous();
                    let input_tensors = Tensor::cat(&i_v[..batch_size], 0);
                    // println!("        thread {}: preparing tensors", thread_name);
                    // println!("            thread {}: eval input tensors: {:?}", thread_name, input_tensors);
                    // println!("        thread {}: NN evaluation:", thread_name);
                    let sw_inference = Instant::now();
                    let (board_eval, policy) =
                        eval_state(input_tensors, network).expect("Evaluation failed");
                    let elapsed = sw_inference.elapsed().as_nanos() as f32 / 1e9;
                    // let evals_per_sec = batch_size as f32 / elapsed;
                    let _ = evals_per_sec_sender
                        .send(CollectorMessage::ExecutorStatistics(batch_size as f32));
                    // println!("            thread {}: NN evaluation done! {}s", thread_name, elapsed);
                    // println!("        thread {}: processing outputs:",thread_name);
                    // println!("            thread {}: output tensors: {:?}, {:?}", thread_name, board_eval, policy);
                    // distribute results to the output senders
                    // println!("        thread {}: sending tensors back to mcts:", thread_name);
                    for i in 0..batch_size {
                        let sender = output_senders
                            .pop_front()
                            .expect("There should be a sender for each job");
                        let id = id_vec
                            .pop_front()
                            .expect("There should be an ID for each job");
                        let result = (board_eval.get(i as i64), policy.get(i as i64));
                        // println!("            thread {}, SENT! {:?}", i, &result);
                        let return_pack = ReturnPacket { packet: result, id };
                        sender
                            .send(ReturnMessage::ReturnMessage(Ok(return_pack)))
                            .expect("Should be able to send the result");
                    }
                    drop(input_vec.drain(0..batch_size));
                }
            }
            Message::NewNetwork(Err(RecvError::Disconnected)) => {
                // println!("DISCONNECTED NET!");
                graph_disconnected = true;
                if network.is_none() && input_vec.is_empty() {
                    break; // exit if no network and no ongoing jobs
                }
            }
        }
        let elapsed = sw.elapsed().as_nanos() as f32 / 1e9;
        // println!("thread {}, elapsed time: {}s", thread_name, elapsed);
        debug_counter += 1;
    }
    // Return the senders to avoid them being dropped and disconnected
}
