use std::{
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use tz_rust::message_types::{
    Entity, MessageServer, MessageType, ServerMessageRecv, ServerMessageSend,
};

fn handle_client(
    stream: TcpStream,
    clients: Arc<Mutex<Vec<TcpStream>>>,
    messages: Arc<Mutex<Vec<MessageServer>>>,
    stats_counters: Arc<Mutex<(f32, f32)>>,
    start_time: Arc<Mutex<Instant>>,
) {
    let mut cloned_handle = stream.try_clone().unwrap();
    let mut reader = BufReader::new(&stream);

    loop {
        let mut recv_msg = String::new();
        if let Err(_) = reader.read_line(&mut recv_msg) {
            recv_msg.clear();
            continue;
        }
        // println!("{:?}", recv_msg);
        let message: MessageServer = match serde_json::from_str(&recv_msg) {
            Ok(msg) => msg,
            Err(err) => {
                eprintln!(
                    "[Error] Error deserialising message! {:?} {}",
                    recv_msg, err
                );
                recv_msg.clear();
                continue;
            }
        };
        println!("[Message] {:?}", message);
        let mut all_messages = messages.lock().unwrap();
        all_messages.push(message.clone());
        let purpose = message.purpose;

        match purpose {
            MessageType::Initialise(entity) => {
                let message_send: MessageServer;
                match entity {
                    Entity::RustDataGen => {
                        let id = all_messages
                            .iter()
                            .filter(|&n| {
                                *n == MessageServer {
                                    purpose: MessageType::Initialise(Entity::RustDataGen),
                                }
                            })
                            .count()
                            - 1;
                        println!("id {}", id);
                        message_send = MessageServer {
                            purpose: MessageType::IdentityConfirmation((entity, id)),
                        };
                    }
                    Entity::PythonTraining => {
                        let id = all_messages
                            .iter()
                            .filter(|&n| {
                                *n == MessageServer {
                                    purpose: MessageType::Initialise(Entity::PythonTraining),
                                }
                            })
                            .count()
                            - 1;
                        message_send = MessageServer {
                            purpose: MessageType::IdentityConfirmation((entity, id)),
                        };
                    }
                    Entity::GUIMonitor => {
                        let id = all_messages
                            .iter()
                            .filter(|&n| {
                                *n == MessageServer {
                                    purpose: MessageType::Initialise(Entity::GUIMonitor),
                                }
                            })
                            .count()
                            - 1;
                        message_send = MessageServer {
                            purpose: MessageType::IdentityConfirmation((entity, id)),
                        };
                    }
                }
                let mut serialised =
                    serde_json::to_string(&message_send).expect("serialization failed");
                serialised += "\n";
                if let Err(msg) = cloned_handle.write_all(serialised.as_bytes()) {
                    eprintln!("Error sending identification! {}", msg);
                    break;
                } else {
                    println!("[Sent Identification] {:?}", message_send);
                }
                recv_msg.clear();
                continue;
            }
            MessageType::JobSendPath(job_path) => {
                // send to all python training instances
                let target_receivers = all_messages
                    .iter()
                    .filter(|&n| {
                        *n == MessageServer {
                            purpose: MessageType::Initialise(Entity::GUIMonitor),
                        }
                    })
                    .count()
                    - 1;
            }
            MessageType::StatisticsSend(statistics) => {
                let mut stats = stats_counters.lock().unwrap_or_else(|e| e.into_inner());
                let mut start_time = start_time.lock().unwrap_or_else(|e| e.into_inner());
                let elapsed = start_time.elapsed().as_secs_f32();
                match statistics {
                        tz_rust::message_types::Statistics::NodesPerSecond(nps) => {
                            stats.0 += nps;
                        }
                        tz_rust::message_types::Statistics::EvalsPerSecond(evals_per_sec) => {
                            stats.1 += evals_per_sec;
                        }
                    }
                if elapsed >= 1.0 {
                    println!("[Statistics-nps] {}", stats.0);
                    println!("[Statistics-evals] {}", stats.1);
                    *stats = (0.0, 0.0);
                    *start_time = Instant::now();
                }
                recv_msg.clear();
                continue;
            }
            MessageType::RequestingNet => {}
            MessageType::NewNetworkPath(_) => {}
            MessageType::IdentityConfirmation(_) => {
                println!("[Warning] IdentityConfirmation Message type is not possible")
            }
            // let received = message.initialise_identity.unwrap();
        }
        let all_clients = clients.lock().unwrap();
        for mut client in all_clients.iter() {
            if let Err(msg) = client.write_all(recv_msg.as_bytes()) {
                eprintln!("Error sending message to client! {}", msg);
                continue;
            }
            let mut disp_msg = recv_msg.clone();
            disp_msg.retain(|c| c != '\n'); // remove newline
            println!("[Sent to {}]: {}", client.peer_addr().unwrap(), disp_msg);
        }
        recv_msg.clear();
        continue;

        // if message.has_net {
        //     let all_clients = clients.lock().unwrap();
        //     for mut client in all_clients.iter() {
        //         if let Err(msg) = client.write_all(recv_msg.as_bytes()) {
        //             // eprintln!("Error sending message to client! {}", msg);
        //             continue;
        //         }

        //         let mut disp_msg = recv_msg.clone();
        //         disp_msg.retain(|c| c != '\n'); // remove newline
        //         println!("[Sent to {}]: {}", client.peer_addr().unwrap(), disp_msg);
        //     }
        // }
        // recv_msg.clear();
    }

    println!(
        "[Server] Client disconnected: {:?}",
        stream.peer_addr().unwrap()
    );
}

fn main() {
    let listener = TcpListener::bind("127.0.0.1:38475").expect("Failed to bind address");
    let clients: Arc<Mutex<Vec<TcpStream>>> = Arc::new(Mutex::new(Vec::new()));
    let messages: Arc<Mutex<Vec<MessageServer>>> = Arc::new(Mutex::new(Vec::new()));
    let stats_counters: Arc<Mutex<(f32, f32)>> = Arc::new(Mutex::new((0.0, 0.0)));
    let start_time: Arc<Mutex<Instant>> = Arc::new(Mutex::new(Instant::now()));

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let cloned_clients = Arc::clone(&clients);
                let cloned_messages = Arc::clone(&messages);
                let cloned_stats_counters = Arc::clone(&stats_counters);
                let cloned_start_time = Arc::clone(&start_time);
                let addr = stream.peer_addr().expect("Failed to get peer address");
                println!("[Server] New connection: {}", addr);

                {
                    let mut all_clients = cloned_clients.lock().unwrap();
                    all_clients.push(stream.try_clone().expect("Failed to clone stream"));
                }

                let cloned_clients = Arc::clone(&clients);
                thread::spawn(move || {
                    handle_client(
                        stream,
                        cloned_clients,
                        cloned_messages,
                        cloned_stats_counters,
                        cloned_start_time,
                    );
                });
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }
}
