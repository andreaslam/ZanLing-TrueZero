use std::{
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use sha2::{Digest, Sha256};
use tzrust::{
    debug_print,
    message_types::{Entity, MessageServer, MessageType, SPRTResult},
};

fn net_checksum(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

// serialise a message with the trailing newline delimiter the wire protocol expects.
fn to_wire(message: &MessageServer) -> String {
    let mut serialised = serde_json::to_string(message).expect("serialisation failed");
    serialised += "\n";
    serialised
}

fn write_to_live_streams(streams: &mut Vec<TcpStream>, wire: &str) -> usize {
    streams.retain_mut(|stream| stream.write_all(wire.as_bytes()).is_ok());
    streams.len()
}

fn rejection_result() -> SPRTResult {
    SPRTResult {
        elo: (0.0, 0.0, 0.0),
        accept_new_net: false,
    }
}

fn notify_trainers_of_rejection(training_streams: &Arc<Mutex<Vec<TcpStream>>>, reason: &str) {
    println!("[Server] Resetting pending candidate: {}", reason);
    let result_wire = to_wire(&MessageServer {
        purpose: MessageType::TestResult(rejection_result()),
    });
    let mut trainers = training_streams.lock().unwrap();
    let n_trainers = write_to_live_streams(&mut trainers, &result_wire);
    println!(
        "[Server] Sent candidate reset to {} Python trainer(s)",
        n_trainers
    );
}

fn clear_pending_candidate(
    pending_candidate: &Arc<Mutex<Option<Vec<u8>>>>,
    pending_candidate_checksum: &Arc<Mutex<Option<String>>>,
) -> bool {
    let mut pending = pending_candidate.lock().unwrap();
    let mut pending_checksum = pending_candidate_checksum.lock().unwrap();
    let had_pending = pending.is_some() || pending_checksum.is_some();
    *pending = None;
    *pending_checksum = None;
    had_pending
}

fn handle_client(
    stream: TcpStream,
    clients: Arc<Mutex<Vec<TcpStream>>>,
    messages: Arc<Mutex<Vec<MessageServer>>>,
    stats_counters: Arc<Mutex<(usize, usize)>>,
    start_time: Arc<Mutex<Instant>>,
    net_path: Arc<Mutex<Option<String>>>,
    net_data: Arc<Mutex<Option<Vec<u8>>>>,
    tb_link: Arc<Mutex<Option<(String, String)>>>,
    transfer_in_progress: Arc<Mutex<bool>>,
    latest_net: Arc<Mutex<Option<Vec<u8>>>>,
    latest_net_checksum: Arc<Mutex<Option<String>>>,
    pending_candidate: Arc<Mutex<Option<Vec<u8>>>>,
    pending_candidate_checksum: Arc<Mutex<Option<String>>>,
    sprt_streams: Arc<Mutex<Vec<TcpStream>>>,
    training_streams: Arc<Mutex<Vec<TcpStream>>>,
    datagen_streams: Arc<Mutex<Vec<TcpStream>>>,
    h0_sent_to_datagen: Arc<Mutex<bool>>,
) {
    let mut cloned_handle = stream.try_clone().unwrap();
    let mut reader = BufReader::new(&stream);
    let mut has_net = false;
    let mut needs_tb_link = false;
    let mut client_entity: Option<Entity> = None;
    let peer_addr = stream
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    loop {
        let mut recv_msg = String::new();
        if reader.read_line(&mut recv_msg).is_err() {
            recv_msg.clear();
            break;
        }

        if !recv_msg.is_empty() {
            let message: MessageServer = match serde_json::from_str(&recv_msg) {
                Ok(msg) => msg,
                Err(_) => {
                    recv_msg.clear();
                    continue;
                }
            };

            debug_print!("Received message: {:?}", message);

            let mut all_messages = messages.lock().unwrap();
            let mut net_path = net_path.lock().unwrap();
            let mut net_data = net_data.lock().unwrap();
            let mut tb_link = tb_link.lock().unwrap();
            let mut transfer_in_progress = transfer_in_progress.lock().unwrap();
            let saved_msg: MessageServer = message.clone();
            let purpose = message.purpose;

            match purpose {
                MessageType::TestResult(result) => {
                    println!("[Test-result] {:?}", result);
                    let mut pending = pending_candidate.lock().unwrap();
                    let mut pending_checksum = pending_candidate_checksum.lock().unwrap();
                    if result.accept_new_net {
                        println!("[Test-result]: New network accepted, releasing to generators");
                        if let Some(candidate) = pending.take() {
                            // promote the candidate to the latest net and release it to every data generator.
                            *latest_net.lock().unwrap() = Some(candidate.clone());
                            *latest_net_checksum.lock().unwrap() = pending_checksum.take();
                            let release = to_wire(&MessageServer {
                                purpose: MessageType::NewNetworkData(candidate),
                            });
                            let mut datagens = datagen_streams.lock().unwrap();
                            let n_datagens = write_to_live_streams(&mut datagens, &release);
                            println!(
                                "[Server] Released accepted candidate to {} data generator(s)",
                                n_datagens
                            );
                        }
                    } else {
                        // rejected: drop the candidate, generators keep the current net
                        *pending = None;
                        *pending_checksum = None;
                        println!("[Test-result]: Candidate rejected, keeping old network");
                    }
                    let result_wire = to_wire(&MessageServer {
                        purpose: MessageType::TestResult(result),
                    });
                    let mut trainers = training_streams.lock().unwrap();
                    write_to_live_streams(&mut trainers, &result_wire);
                    let mut sprts = sprt_streams.lock().unwrap();
                    write_to_live_streams(&mut sprts, &result_wire);
                    recv_msg.clear();
                    continue;
                }
                MessageType::Initialise(entity) => {
                    debug_print!("Initialise message received for entity: {:?}", entity);
                    client_entity = Some(entity.clone());
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
                                .count();
                            message_send = MessageServer {
                                purpose: MessageType::IdentityConfirmation((entity, id)),
                            };
                            // register as a data generator so accepted candidate nets can be released to it
                            if let Ok(clone) = stream.try_clone() {
                                let mut datagens = datagen_streams.lock().unwrap();
                                datagens.retain(|c| {
                                    c.peer_addr()
                                        .map(|a| a.to_string() != peer_addr)
                                        .unwrap_or(false)
                                });
                                datagens.push(clone);
                            }
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
                                .count();
                            message_send = MessageServer {
                                purpose: MessageType::IdentityConfirmation((entity, id)),
                            };
                            if !has_net {
                                let extra_request = to_wire(&MessageServer {
                                    purpose: MessageType::RequestingNet(),
                                });
                                if let Err(msg) = cloned_handle.write_all(extra_request.as_bytes())
                                {
                                    eprintln!("Error sending identification! {}", msg);
                                    break;
                                } else {
                                    println!("[Server] Requesting net");
                                }
                            }
                            if let Ok(clone) = stream.try_clone() {
                                let mut trainers = training_streams.lock().unwrap();
                                trainers.retain(|c| {
                                    c.peer_addr()
                                        .map(|a| a.to_string() != peer_addr)
                                        .unwrap_or(false)
                                });
                                trainers.push(clone);
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
                                .count();
                            message_send = MessageServer {
                                purpose: MessageType::IdentityConfirmation((entity, id)),
                            };
                        }
                        Entity::TBHost => {
                            all_messages.push(saved_msg.clone());
                            let id = all_messages
                                .iter()
                                .filter(|&n| {
                                    *n == MessageServer {
                                        purpose: MessageType::Initialise(Entity::GUIMonitor),
                                    }
                                })
                                .count();
                            message_send = MessageServer {
                                purpose: MessageType::IdentityConfirmation((entity, id)),
                            };
                            let tb_link_request = MessageServer {
                                purpose: MessageType::TBLinkRequest(),
                            };

                            println!("[Server] Requested TensorBoard link");

                            let mut serialised = serde_json::to_string(&tb_link_request)
                                .expect("serialisation failed");
                            serialised += "\n";
                            if let Err(msg) = cloned_handle.write_all(serialised.as_bytes()) {
                                eprintln!("Error sending identification! {}", msg);
                                break;
                            } else {
                                println!("[Server] Requesting net");
                            }
                        }
                        Entity::SPRTRunner => {
                            // register this connection as the SPRT runner so new candidate nets can be routed to it exclusively
                            message_send = MessageServer {
                                purpose: MessageType::IdentityConfirmation((entity, 1)),
                            };
                            match stream.try_clone() {
                                Ok(clone) => {
                                    let mut sprts = sprt_streams.lock().unwrap();
                                    sprts.retain(|c| {
                                        c.peer_addr()
                                            .map(|a| a.to_string() != peer_addr)
                                            .unwrap_or(false)
                                    });
                                    sprts.push(clone);
                                }
                                Err(e) => {
                                    eprintln!("[Server] Failed to register SPRT stream: {}", e)
                                }
                            }
                        }
                    }
                    let serialised = to_wire(&message_send);
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
                    if matches!(client_entity, Some(Entity::SPRTRunner)) {
                        let latest = latest_net.lock().unwrap().clone();
                        let pending = pending_candidate.lock().unwrap().clone();
                        if let Some(net) = latest {
                            let h0_wire = to_wire(&MessageServer {
                                purpose: MessageType::NewNetworkData(net),
                            });
                            if let Err(msg) = cloned_handle.write_all(h0_wire.as_bytes()) {
                                eprintln!("[Server] Failed to send H0 to SPRT runner: {}", msg);
                                break;
                            }
                            println!("[Server] Sent accepted H0 to SPRT runner");
                        }
                        if let Some(candidate) = pending {
                            let h1_wire = to_wire(&MessageServer {
                                purpose: MessageType::NewNetworkData(candidate),
                            });
                            if let Err(msg) = cloned_handle.write_all(h1_wire.as_bytes()) {
                                eprintln!(
                                    "[Server] Failed to send pending H1 to SPRT runner: {}",
                                    msg
                                );
                                break;
                            }
                            println!("[Server] Sent pending candidate H1 to SPRT runner");
                        }
                    }
                    recv_msg.clear();
                    continue;
                }
                MessageType::JobSendPath(_) => {
                    debug_print!("JobSendPath message received");
                    let refresh_msg = MessageServer {
                        purpose: MessageType::TBLinkRequest(),
                    };
                    let mut serialised =
                        serde_json::to_string(&refresh_msg).expect("serialisation failed");
                    serialised += "\n";
                    if let Err(msg) = cloned_handle.write_all(serialised.as_bytes()) {
                        eprintln!("Error sending identification! {}", msg);
                        break;
                    }
                }
                MessageType::StatisticsSend(statistics) => {
                    debug_print!("StatisticsSend message received: {:?}", statistics);
                    let mut stats = stats_counters.lock().unwrap_or_else(|e| e.into_inner());
                    let mut start_time = start_time.lock().unwrap_or_else(|e| e.into_inner());
                    let elapsed = start_time.elapsed().as_secs_f32() as usize;
                    match statistics {
                        tzrust::message_types::Statistics::NodesPerSecond(nps) => {
                            stats.0 += nps;
                        }
                        tzrust::message_types::Statistics::EvalsPerSecond(evals_per_sec) => {
                            stats.1 += evals_per_sec;
                        }
                    }
                    if elapsed >= 1 {
                        println!("[Statistics-nps] {}", stats.0);
                        println!("[Statistics-evals] {}", stats.1);
                        *stats = (0, 0);
                        *start_time = Instant::now();
                    }
                    recv_msg.clear();
                    continue;
                }
                MessageType::RequestingNet() => {
                    debug_print!("RequestingNet message received");
                    if !has_net {
                        let requested_net = if matches!(client_entity, Some(Entity::SPRTRunner)) {
                            latest_net.lock().unwrap().clone()
                        } else {
                            net_data.clone()
                        };
                        if let Some(path) = requested_net {
                            let extra_request = MessageServer {
                                purpose: MessageType::NewNetworkData(path),
                            };
                            let mut serialised = serde_json::to_string(&extra_request)
                                .expect("serialisation failed");
                            serialised += "\n";
                            if let Err(msg) = cloned_handle.write_all(serialised.as_bytes()) {
                                eprintln!("Error sending identification! {}", msg);
                                break;
                            }
                        }
                        recv_msg.clear();
                        continue;
                    } else {
                        has_net = false;
                    }
                }
                MessageType::NewNetworkPath(path) => {
                    debug_print!("NewNetworkPath message received: {}", path);
                    *net_path = Some(path);
                    recv_msg.clear();
                    continue;
                }
                MessageType::IdentityConfirmation(_) => {
                    println!("[Warning] Identity Confirmation Message type is not possible")
                }
                MessageType::JobSendData(_) => {
                    debug_print!("JobSendData message received");
                    let refresh_msg = MessageServer {
                        purpose: MessageType::TBLinkRequest(),
                    };
                    let mut serialised =
                        serde_json::to_string(&refresh_msg).expect("serialisation failed");
                    serialised += "\n";
                    if let Err(msg) = cloned_handle.write_all(serialised.as_bytes()) {
                        eprintln!("Error sending identification! {}", msg);
                        break;
                    }
                }
                MessageType::NewNetworkData(data) => {
                    debug_print!("NewNetworkData message received: {:?}", data);
                    let checksum = net_checksum(&data);
                    if *transfer_in_progress {
                        // Another transfer is mid-flight; forward as a follow-up chunk.
                        let follow_up_msg = to_wire(&MessageServer {
                            purpose: MessageType::NewNetworkData(data.clone()),
                        });
                        thread::sleep(Duration::from_millis(1));
                        if let Err(msg) = cloned_handle.write_all(follow_up_msg.as_bytes()) {
                            eprintln!("Error sending follow-up net data! {}", msg);
                        }
                    } else {
                        *transfer_in_progress = true;
                        *net_data = Some(data.clone());

                        let mut latest = latest_net.lock().unwrap();
                        let mut latest_checksum = latest_net_checksum.lock().unwrap();
                        let mut pending = pending_candidate.lock().unwrap();
                        let mut pending_checksum = pending_candidate_checksum.lock().unwrap();
                        let mut h0_sent = h0_sent_to_datagen.lock().unwrap();

                        if latest_checksum.as_deref() == Some(checksum.as_str()) {
                            println!("[Server] Ignoring duplicate latest net {}", checksum);
                            *transfer_in_progress = false;
                            recv_msg.clear();
                            continue;
                        }

                        if pending_checksum.as_deref() == Some(checksum.as_str()) {
                            println!("[Server] Ignoring duplicate pending candidate {}", checksum);
                            *transfer_in_progress = false;
                            recv_msg.clear();
                            continue;
                        }

                        if pending.is_some() {
                            println!(
                                "[Server][Warning] Candidate {} arrived while another candidate is pending; ignoring until the current SPRT finishes",
                                checksum
                            );
                            *transfer_in_progress = false;
                            recv_msg.clear();
                            continue;
                        }

                        if latest.is_none() && pending.is_none() && !*h0_sent {
                            // this is the very first net. Everyone gets it so generators can start producing data and the SPRT runner has its initial benchmark
                            *latest = Some(data.clone());
                            *latest_checksum = Some(checksum.clone());
                            *h0_sent = true;
                            let h0_wire = to_wire(&MessageServer {
                                purpose: MessageType::NewNetworkData(data.clone()),
                            });
                            {
                                let mut sprts = sprt_streams.lock().unwrap();
                                let n_sprt = write_to_live_streams(&mut sprts, &h0_wire);
                                println!("[Server] H0 sent to {} SPRT runner(s)", n_sprt);
                            }
                            {
                                let mut datagens = datagen_streams.lock().unwrap();
                                let n_datagens = write_to_live_streams(&mut datagens, &h0_wire);
                                println!(
                                    "[Server] H0 {} sent to {} data generator(s)",
                                    checksum, n_datagens
                                );
                            }
                        } else {
                            // hold candidate as pending and route it ONLY to the SPRT runner for gating. Generators do NOT get it until the test passes
                            let cand_wire = to_wire(&MessageServer {
                                purpose: MessageType::NewNetworkData(data.clone()),
                            });
                            let mut sprts = sprt_streams.lock().unwrap();
                            let n_sprt = write_to_live_streams(&mut sprts, &cand_wire);
                            *pending = Some(data.clone());
                            *pending_checksum = Some(checksum.clone());
                            if n_sprt == 0 {
                                println!(
                                    "[Server] Candidate net {} held pending until an SPRT runner connects",
                                    checksum
                                );
                            } else {
                                println!(
                                    "[Server] Candidate net {} routed to {} SPRT runner(s) for testing",
                                    checksum, n_sprt
                                );
                            }
                        }
                        *transfer_in_progress = false;
                    }
                    recv_msg.clear();
                    continue;
                }
                MessageType::TBLink(ref msg) => {
                    debug_print!("TBLink message received: {:?}", msg);
                    *tb_link = Some(msg.clone());
                    if needs_tb_link {
                        let tb_link_msg = MessageServer {
                            purpose: MessageType::TBLink(msg.clone()),
                        };
                        let mut serialised =
                            serde_json::to_string(&tb_link_msg).expect("serialisation failed");
                        serialised += "\n";
                        if let Err(msg) = cloned_handle.write_all(serialised.as_bytes()) {
                            eprintln!("Error sending TensorBoard link! {}", msg);
                            break;
                        }
                        needs_tb_link = false;
                    }
                }
                MessageType::CreateTB() => {
                    debug_print!("CreateTB message received");
                    needs_tb_link = true;
                }
                MessageType::TBLinkRequest() => match *tb_link {
                    Some(ref link) => {
                        let tb_link_msg = MessageServer {
                            purpose: MessageType::TBLink(link.clone()),
                        };
                        println!("[Server] TensorBoard Link: {:?}", tb_link_msg);
                        let mut serialised =
                            serde_json::to_string(&tb_link_msg).expect("serialisation failed");
                        serialised += "\n";
                        if let Err(msg) = cloned_handle.write_all(serialised.as_bytes()) {
                            eprintln!("Error sending TensorBoard link! {}", msg);
                            break;
                        }
                        needs_tb_link = false;
                    }
                    None => {
                        needs_tb_link = true;
                    }
                },
                MessageType::EvaluationRequest(_input_data) => {
                    debug_print!("EvaluationRequest message received: {:?}", _input_data);
                }
            }

            let all_clients = clients.lock().unwrap();
            for mut client in all_clients.iter() {
                if client.write_all(recv_msg.as_bytes()).is_err() {
                    continue;
                }
                let mut disp_msg = recv_msg.clone();
                disp_msg.retain(|c| c != '\n');
                debug_print!("Sent message to client: {:?}", disp_msg);
            }
            recv_msg.clear();
            continue;
        } else {
            break;
        }
    }

    if matches!(client_entity, Some(Entity::SPRTRunner)) {
        {
            let mut sprts = sprt_streams.lock().unwrap();
            sprts.retain(|c| {
                c.peer_addr()
                    .map(|a| a.to_string() != peer_addr)
                    .unwrap_or(false)
            });
        }

        let no_sprt_runners = sprt_streams.lock().unwrap().is_empty();
        if no_sprt_runners
            && clear_pending_candidate(&pending_candidate, &pending_candidate_checksum)
        {
            notify_trainers_of_rejection(
                &training_streams,
                "SPRT runner disconnected before completing the test",
            );
        }
    }
}

fn main() {
    let listener = TcpListener::bind("0.0.0.0:38475").expect("Failed to bind address");
    let clients: Arc<Mutex<Vec<TcpStream>>> = Arc::new(Mutex::new(Vec::new()));
    let messages: Arc<Mutex<Vec<MessageServer>>> = Arc::new(Mutex::new(Vec::new()));
    let net_path: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let net_data: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let tb_link: Arc<Mutex<Option<(String, String)>>> = Arc::new(Mutex::new(None));
    let stats_counters: Arc<Mutex<(usize, usize)>> = Arc::new(Mutex::new((0, 0)));
    let start_time: Arc<Mutex<Instant>> = Arc::new(Mutex::new(Instant::now()));
    let transfer_in_progress: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
    let latest_net: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let latest_net_checksum: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let pending_candidate: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let pending_candidate_checksum: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let sprt_streams: Arc<Mutex<Vec<TcpStream>>> = Arc::new(Mutex::new(Vec::new()));
    let training_streams: Arc<Mutex<Vec<TcpStream>>> = Arc::new(Mutex::new(Vec::new()));
    let datagen_streams: Arc<Mutex<Vec<TcpStream>>> = Arc::new(Mutex::new(Vec::new()));
    let h0_sent_to_datagen: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));

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
                let cloned_tb_link = Arc::clone(&tb_link);
                let cloned_transfer_in_progress = Arc::clone(&transfer_in_progress);
                let cloned_latest_net = Arc::clone(&latest_net);
                let cloned_latest_net_checksum = Arc::clone(&latest_net_checksum);
                let cloned_pending_candidate = Arc::clone(&pending_candidate);
                let cloned_pending_candidate_checksum = Arc::clone(&pending_candidate_checksum);
                let cloned_sprt_streams = Arc::clone(&sprt_streams);
                let cloned_training_streams = Arc::clone(&training_streams);
                let cloned_datagen_streams = Arc::clone(&datagen_streams);
                let cloned_h0_sent = Arc::clone(&h0_sent_to_datagen);
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
                        cloned_tb_link,
                        cloned_transfer_in_progress,
                        cloned_latest_net,
                        cloned_latest_net_checksum,
                        cloned_pending_candidate,
                        cloned_pending_candidate_checksum,
                        cloned_sprt_streams,
                        cloned_training_streams,
                        cloned_datagen_streams,
                        cloned_h0_sent,
                    );
                });
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }
}
