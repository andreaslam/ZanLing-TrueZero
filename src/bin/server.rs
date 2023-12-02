use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

fn handle_client(mut stream: TcpStream, clients: Arc<Mutex<Vec<TcpStream>>>) {
    let mut buffer = [0; 512];
    loop {
        let bytes_read = match stream.read(&mut buffer) {
            Ok(n) if n == 0 => {
                break; // connection closed
            }
            Ok(n) => n,
            Err(_) => {
                break; // error while reading
            }
        };

        let received = String::from_utf8_lossy(&buffer[..bytes_read]);
        println!("Received: {}", received);

        // Broadcast message to all clients except the sender
        let all_clients = clients.lock().unwrap();
        for mut client in all_clients.iter() {
            if let Err(_) = client.write_all(received.as_bytes()) {
            }
        }
    }
}

fn main() {
    let listener = TcpListener::bind("127.0.0.1:8080").expect("Failed to bind address");
    let clients: Arc<Mutex<Vec<TcpStream>>> = Arc::new(Mutex::new(Vec::new()));

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let cloned_clients = Arc::clone(&clients);
                let addr = stream.peer_addr().expect("Failed to get peer address");
                println!("New connection: {}", addr);

                {
                    let mut all_clients = cloned_clients.lock().unwrap();
                    all_clients.push(stream.try_clone().expect("Failed to clone stream"));
                }

                let cloned_clients = Arc::clone(&clients);
                thread::spawn(move || {
                    handle_client(stream, cloned_clients);
                });
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }
}
