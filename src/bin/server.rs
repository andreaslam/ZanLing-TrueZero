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
) {
    let mut cloned_handle = stream.try_clone().unwrap();
    let mut reader = BufReader::new(&stream);
    let mut nps_start_time = Instant::now();
    let mut nps_vec: Vec<f32> = Vec::new();
    let mut evals_start_time = Instant::now();
    let mut evals_vec: Vec<f32> = Vec::new();

    loop {
        let mut recv_msg = String::new();
        if let Err(_) = reader.read_line(&mut recv_msg) {
            return;
        }

        let message: ServerMessageSend = match serde_json::from_str(&recv_msg) {
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
            if purpose == "statistics-nps" {
                if nps_start_time.elapsed() >= Duration::from_secs(1) {
                    let nps: f32 = nps_vec.iter().sum();
                    nps_start_time = Instant::now();
                    nps_vec = Vec::new();
                    println!("[Statistics-nps] {}", nps);
                } else {
                    nps_vec.push(message.nps.unwrap());
                }
            } else if purpose == "statistics-evals" {
                if evals_start_time.elapsed() >= Duration::from_secs(1) {
                    let evals_per_second: f32 = evals_vec.iter().sum();
                    evals_start_time = Instant::now();
                    evals_vec = Vec::new();
                    println!("[Statistics-evals] {}", evals_per_second);
                } else {
                    evals_vec.push(message.evals_per_second.unwrap());
                }
            }
        }
        else {
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

    println!("[Server] Client disconnected!");
}
fn main() {
    let listener = TcpListener::bind("127.0.0.1:8080").expect("Failed to bind address");
    let clients: Arc<Mutex<Vec<TcpStream>>> = Arc::new(Mutex::new(Vec::new()));
    let messages: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let cloned_clients = Arc::clone(&clients);
                let cloned_messages = Arc::clone(&messages);
                let addr = stream.peer_addr().expect("Failed to get peer address");
                println!("[Server] New connection: {}", addr);

                {
                    let mut all_clients = cloned_clients.lock().unwrap();
                    all_clients.push(stream.try_clone().expect("Failed to clone stream"));
                }

                let cloned_clients = Arc::clone(&clients);
                thread::spawn(move || {
                    handle_client(stream, cloned_clients, cloned_messages);
                });
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }
}
