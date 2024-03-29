use std::{
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use tz_rust::message_types::{Entity, MessageServer, MessageType};

fn handle_client(
    stream: TcpStream,
    clients: Arc<Mutex<Vec<TcpStream>>>,
    messages: Arc<Mutex<Vec<MessageServer>>>,
    stats_counters: Arc<Mutex<(f32, f32)>>,
    start_time: Arc<Mutex<Instant>>,
    net_path: Arc<Mutex<Option<String>>>,
    net_data: Arc<Mutex<Option<Vec<u8>>>>,
) {
    let mut cloned_handle = stream.try_clone().unwrap();
    let mut reader = BufReader::new(&stream);
    let mut has_net = false;
    loop {
        
        let mut recv_msg = String::new();
        if let Err(_) = reader.read_line(&mut recv_msg) {
            recv_msg.clear();
            break;
        }

        if !recv_msg.is_empty() {
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

            let mut all_messages = messages.lock().unwrap();
            let mut net_path = net_path.lock().unwrap();
            let mut net_data = net_data.lock().unwrap();

            // TODO: find more msg types to NOT save to all_messages
            let saved_msg: MessageServer = message.clone();
            let purpose = message.purpose;
            match purpose {
                MessageType::Initialise(entity) => {
                    let message_send: MessageServer;
                    match entity {
                        Entity::RustDataGen => {
                            all_messages.push(saved_msg.clone());
                            let id = all_messages
                                .iter()
                                .filter(|&n| {
                                    *n == MessageServer {
                                        purpose: MessageType::Initialise(Entity::RustDataGen),
                                    }
                                })
                                .count()
                                - 1;
                            // println!("id {}", id);
                            message_send = MessageServer {
                                purpose: MessageType::IdentityConfirmation((entity, id)),
                            };
                        }
                        Entity::PythonTraining => {
                            all_messages.push(saved_msg.clone());
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
                            if !has_net {
                                let extra_request = MessageServer {
                                    purpose: MessageType::RequestingNet,
                                };
                                let mut serialised = serde_json::to_string(&extra_request)
                                    .expect("serialization failed");
                                serialised += "\n";
                                if let Err(msg) = cloned_handle.write_all(serialised.as_bytes()) {
                                    eprintln!("Error sending identification! {}", msg);
                                    break;
                                } else {
                                    println!("[Server] Requesting net");
                                }
                            }
                        }
                        Entity::GUIMonitor => {
                            all_messages.push(saved_msg.clone());
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
                        println!(
                            "[Sent Identification to {}] {:?}",
                            cloned_handle.peer_addr().unwrap(),
                            message_send
                        );
                    }
                    recv_msg.clear();
                    continue;
                }
                MessageType::JobSendPath(_) => {}
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
                MessageType::RequestingNet => {
                    if !has_net {
                        match *net_data {
                            Some(ref path) => {
                                let extra_request = MessageServer {
                                    // purpose: MessageType::NewNetworkPath(path.clone()),
                                    purpose: MessageType::NewNetworkData(path.clone()),
                                };
                                let mut serialised = serde_json::to_string(&extra_request)
                                    .expect("serialisation failed");
                                serialised += "\n";
                                if let Err(msg) = cloned_handle.write_all(serialised.as_bytes()) {
                                    eprintln!("Error sending identification! {}", msg);
                                    break;
                                } else {
                                    // println!("[Server] Sending net path {}", path.clone());
                                }
                            }
                            None => {}
                        }
                        recv_msg.clear();
                        continue;
                    } else {
                        has_net = false;
                    }
                }
                MessageType::NewNetworkPath(path) => {
                    *net_path = Some(path);
                }
                MessageType::IdentityConfirmation(_) => {
                    println!("[Warning] Identity Confirmation Message type is not possible")
                }
                MessageType::JobSendData(_) => {}
                MessageType::NewNetworkData(data) => {
                    *net_data = Some(data);
                }
            }
            // println!("[Message] {:?}", message);
            let all_clients = clients.lock().unwrap();
            for mut client in all_clients.iter() {
                if let Err(msg) = client.write_all(recv_msg.as_bytes()) {
                    // eprintln!("Error sending message to client! {}", msg);
                    continue;
                }
                let mut disp_msg = recv_msg.clone();
                disp_msg.retain(|c| c != '\n'); // remove newline
                                                // println!("[Sent to {}]: {}", client.peer_addr().unwrap(), disp_msg);
            }
            recv_msg.clear();
            continue;
        } else {
            break
        }
    }
    println!("[Server] client disconnected {:#}", stream.peer_addr().unwrap());
    drop(cloned_handle);
    drop(reader);
}

fn main() {
    let listener = TcpListener::bind("127.0.0.1:38475").expect("Failed to bind address");
    let clients: Arc<Mutex<Vec<TcpStream>>> = Arc::new(Mutex::new(Vec::new()));
    let messages: Arc<Mutex<Vec<MessageServer>>> = Arc::new(Mutex::new(Vec::new()));
    let net_path: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let net_data: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
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
                let cloned_net_path = Arc::clone(&net_path);
                let cloned_net_data = Arc::clone(&net_data);
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
                        cloned_net_path,
                        cloned_net_data,
                    );
                });
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }
}
