use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

fn handle_client(mut stream: TcpStream) {
    let mut buffer = [0; 16384];
    loop {
        match stream.read(&mut buffer) {
            Ok(bytes_read) => {
                if bytes_read == 0 {
                    break;
                }
                let received = &buffer[..bytes_read];
                let message = String::from_utf8_lossy(&buffer[..bytes_read]).to_string(); // convert received bytes to String
                println!("Received: {:?}", message);

                stream.write_all(received).unwrap();
            }
            Err(e) => {
                eprintln!("Error reading from socket: {}", e);
                break;
            }
        }
        buffer = [0; 1024];
    }
}

fn main() {
    let listener = TcpListener::bind("127.0.0.1:8080").expect("Could not bind to address");
    println!("Server listening on port 8080...");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                println!("New connection: {:?}", stream.peer_addr().unwrap());
                thread::spawn(move || {
                    handle_client(stream);
                });
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }
}
