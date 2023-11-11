use crate::{decoder::eval_state, mcts_trainer::Net};
use flume::{Receiver, RecvError, Selector, Sender};
use std::{cmp::min, collections::VecDeque};
use tch::Tensor;

pub struct Packet {
    pub job: Tensor,
    pub resender: Sender<Message>,
}

pub enum Message {
    NewNetwork(Result<String, RecvError>),
    JobTensor(Result<Packet, RecvError>), // (converted) tensor from mcts search that needs NN evaluation
    ReturnMessage(Result<(Tensor, Tensor), RecvError>),
}

fn handle_new_graph(network: &mut Option<Net>, graph: Option<String>, thread_name: &str) {
    // drop previous network if any to save GPU memory
    if let Some(network) = network.take() {
        println!("{} dropping network", thread_name);
        drop(network);
    }

    // load the new network if any
    *network = graph.map(|graph| Net::new(&graph[..]));
}

pub fn executor_main(
    net_receiver: Receiver<String>,
    tensor_receiver: Receiver<Packet>, // receive tensors from mcts
    num_threads: usize,
) {
    let max_batch_size = min(256, num_threads);
    let mut graph_disconnected = false;
    let mut network: Option<Net> = None;
    let thread_name = std::thread::current()
        .name()
        .unwrap_or("unnamed")
        .to_owned();
    let mut input_vec: Vec<Tensor> = Vec::new();
    let mut output_senders = VecDeque::new(); // collect senders
    let mut debug_counter = 0;

    println!("num_threads: {}", num_threads);

    loop {
        println!("thread {} loop {}:", thread_name, debug_counter);
        println!("    number of output senders: {}", output_senders.len());
        println!("    graph_disconnected: {}", graph_disconnected);
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
                println!("    NEW NET!");
                handle_new_graph(&mut network, Some(graph), &thread_name);
            }
            Message::JobTensor(job) => {
                let job = job.expect("JobTensor should be available");
                let network = network.as_mut().expect("Network should be available");

                input_vec.push(job.job);
                output_senders.push_back(job.resender);

                // evaluate batches
                while input_vec.len() >= max_batch_size {
                    let batch_size = min(max_batch_size, input_vec.len());
                    let input_tensors = Tensor::cat(&input_vec[..batch_size], 0);
                    println!("        preparing tensors");
                    println!("            eval input tensors: {:?}", input_tensors);
                    println!("        NN evaluation:");
                    let (board_eval, policy) =
                        eval_state(input_tensors, network).expect("Evaluation failed");
                    println!("            NN evaluation done!");
                    println!("        processing outputs:");
                    println!("            output tensors: {:?}, {:?}", board_eval, policy);
                    // distribute results to the output senders
                    println!("        sending tensors back to mcts:");
                    for i in 0..batch_size {
                        let sender = output_senders
                            .pop_front()
                            .expect("There should be a sender for each job");
                        let result = (board_eval.get(i as i64), policy.get(i as i64));
                        println!("            thread {}, SENT! {:?}", i, &result);
                        sender
                            .send(Message::ReturnMessage(Ok(result)))
                            .expect("Should be able to send the result");
                    }
                    input_vec.drain(0..batch_size);
                }
            }
            Message::NewNetwork(Err(RecvError::Disconnected)) => {
                println!("DISCONNECTED NET!");
                graph_disconnected = true;
                if network.is_none() && input_vec.is_empty() {
                    break; // exit if no network and no ongoing jobs
                }
            }
            Message::ReturnMessage(_) => unreachable!(),
        }
        debug_counter += 1;
    }
    // Return the senders to avoid them being dropped and disconnected
}
