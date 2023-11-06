use crate::{decoder::eval_state, mcts_trainer::Net};
use flume::{Receiver, RecvError, Selector, Sender, TryRecvError};
use std::{
    cmp::min,
    collections::VecDeque,
    fmt::{Debug, Formatter},
};
use tch::{Device, Tensor};

pub enum Message {
    NewNetwork(Result<String, RecvError>),
    JobTensor(Result<Tensor, RecvError>), // (converted) tensor from mcts search that needs NN evaluation
    ReturnMessage(Result<(Tensor, Tensor), RecvError>),
}

fn handle_new_graph(network: &mut Option<Net>, graph: Option<String>, thread_name: &str) {
    // drop previous network if any to save GPU memory
    if let Some(network) = network.take() {
        // println!("{} dropping network", thread_name);
        drop(network);
    }

    // load the new network if any
    *network = graph.map(|graph| {
        let network = Net::new(&graph[..]);
        network
    });
}

pub fn executor_main(
    net_receiver: Receiver<String>,
    tensor_receivers: Vec<Receiver<Tensor>>, // receive tensors from mcts
    num_threads: usize,
    mut sender_vec: Vec<Sender<Message>>, // send tensors back to mcts
) {
    let max_batch_size = min(256, num_threads);
    let mut graph_disconnected = false;
    let mut network: Option<Net> = None;
    let thread_name = std::thread::current()
        .name()
        .unwrap_or("unnamed")
        .to_owned();
    let mut input_vec: Vec<Tensor> = Vec::new();
    let mut output_senders = VecDeque::from(sender_vec.clone()); // clone the sender vector to avoid disconnection
    let mut debug_counter = 0;

    // println!("num_threads: {}", num_threads);

    loop {
        // println!("loop {}:", debug_counter);
        // println!("    graph_disconnected: {}", graph_disconnected);
        assert!(network.is_some() || !graph_disconnected);

        let mut selector = Selector::new();

        if !graph_disconnected {
            selector = selector.recv(&net_receiver, |res| Message::NewNetwork(res));
        }

        // register all tensor receivers in the selector
        for tensor_receiver in &tensor_receivers {
            selector = selector.recv(tensor_receiver, |res| Message::JobTensor(res));
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

                input_vec.push(job);

                // evaluate a batch
                if input_vec.len() >= max_batch_size {
                    let input_tensors = Tensor::cat(&input_vec, 0);
                    // println!("        preparing tensors");
                    // println!("            eval input tensors: {:?}", input_tensors);
                    // println!("        NN evaluation:");
                    let (board_eval, policy) =
                        eval_state(input_tensors, network).expect("Evaluation failed");
                    // println!("            NN evaluation done!");
                    // println!("        processing outputs:");
                    // println!("            output tensors: {:?}, {:?}", board_eval, policy);
                    // distribute results to the output senders
                    // println!("        sending tensors back to mcts:");
                    for i in 0..input_vec.len() {
                        let sender = output_senders
                            .pop_front()
                            .expect("There should be a sender for each job");
                        let result = (board_eval.get(i as i64), policy.get(i as i64));
                        // println!("            number {}, SENT! {:?}", i, &result);
                        sender
                            .send(Message::ReturnMessage(Ok(result)))
                            .expect("Should be able to send the result");
                        output_senders.push_back(sender); // re-insert the sender back into the deque
                    }
                    input_vec.clear();
                }
            }
            Message::NewNetwork(Err(RecvError::Disconnected)) => {
                // println!("DISCONNECTED NET!");
                graph_disconnected = true;
                if network.is_none() && input_vec.is_empty() {
                    break; // exit if no network and no ongoing jobs
                }
            }
            _ => unreachable!(),
        }
        debug_counter += 1;
    }
    // Return the senders to avoid them being dropped and disconnected
}
