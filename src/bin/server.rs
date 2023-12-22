use std::{
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use tz_rust::message_types::{ServerMessageRecv, ServerMessageSend};

fn handle_client(
    stream: TcpStream,
    clients: Arc<Mutex<Vec<TcpStream>>>,
    messages: Arc<Mutex<Vec<String>>>,
    stats_counters: Arc<Mutex<(f32, f32)>>,
    start_time: Arc<Mutex<Instant>>,
) {
    let mut cloned_handle = stream.try_clone().unwrap();
    let mut reader = BufReader::new(&stream);

    loop {
        let mut recv_msg = String::new();
        if let Err(_) = reader.read_line(&mut recv_msg) {
            return;
        }

        let message: ServerMessageSend = match serde_json::from_str(&recv_msg) {
            Ok(msg) => msg,
            Err(err) => {
                // eprintln!(
                //     "[Error] Error deserialising message! {:?} {}",
                //     recv_msg, err
                // );
                recv_msg.clear();
                continue;
            }
        };

        println!("[Message] {:?}", message);
        let mut all_messages = messages.lock().unwrap();
        let purpose = message.purpose;

        if purpose == "initialise" {
            let received = message.initialise_identity.unwrap();
            if received.starts_with("python-training") || received.starts_with("rust-datagen") {
                all_messages.push(received.clone());
                let id = match received.as_str() {
                    "python-training" => {
                        all_messages
                            .iter()
                            .filter(|&n| *n == "python-training")
                            .count()
                            - 1
                    }
                    "rust-datagen" => {
                        all_messages
                            .iter()
                            .filter(|&n| *n == "rust-datagen")
                            .count()
                            - 1
                    }
                    _ => {
                        eprintln!("Impossible!");
                        break;
                    }
                };

                let message = ServerMessageRecv {
                    verification: format!("{}: {}", received, id).into(),
                    net_path: None,
                };

                let mut serialised = serde_json::to_string(&message).expect("serialization failed");
                serialised += "\n";
                if let Err(msg) = cloned_handle.write_all(serialised.as_bytes()) {
                    // eprintln!("Error sending identification! {}", msg);
                    break;
                }
                println!("[Sent Identification] {:?}", message);
            }
        } else if purpose.starts_with("statistics") {
            let mut stats = stats_counters.lock().unwrap();
            let mut start_time = start_time.lock().unwrap();
            let elapsed = start_time.elapsed().as_secs_f32();
            if elapsed >= 1.0 {
                println!("[Statistics-nps] {}", stats.0);
                println!("[Statistics-evals] {}", stats.1);
                *stats = (0.0, 0.0);
                *start_time = Instant::now();
            } else {
                if let Some(nps_value) = message.nps {
                    stats.0 += nps_value;
                }
                if let Some(evals_value) = message.evals_per_second {
                    stats.1 += evals_value;
                }
            }
            recv_msg.clear();
            continue;
        } else {
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
        }

        if message.has_net {
            let all_clients = clients.lock().unwrap();
            for mut client in all_clients.iter() {
                if let Err(msg) = client.write_all(recv_msg.as_bytes()) {
                    // eprintln!("Error sending message to client! {}", msg);
                    continue;
                }

                let mut disp_msg = recv_msg.clone();
                disp_msg.retain(|c| c != '\n'); // remove newline
                println!("[Sent to {}]: {}", client.peer_addr().unwrap(), disp_msg);
            }
        }
        recv_msg.clear();
    }

    println!("[Server] Client disconnected: {:?}", stream.peer_addr().unwrap());
}

fn main() {
    let listener = TcpListener::bind("127.0.0.1:38475").expect("Failed to bind address");
    let clients: Arc<Mutex<Vec<TcpStream>>> = Arc::new(Mutex::new(Vec::new()));
    let messages: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
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
