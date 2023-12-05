use crossbeam::thread;
use flume::{Receiver, Sender};
use std::{
    env, fs,
    io::{Read, Write},
    net::TcpStream,
    panic,
    time::{Duration, Instant},
};
use tz_rust::{
    executor::{executor_main, Packet},
    fileformat::BinaryOutput,
    selfplay::{CollectorMessage, DataGen},
};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    panic::set_hook(Box::new(|panic_info| {
        // Print panic information
        eprintln!("Panic occurred: {:?}", panic_info);
        // Exit the program immediately
        std::process::exit(1);
    }));
    // connect to python-rust server
    let mut stream = loop {
        match TcpStream::connect("127.0.0.1:8080") {
            Ok(s) => break s,
            Err(_) => continue,
        };
    };
    // identification - this is rust data generation
    stream
        .write_all("rust-datagen".as_bytes())
        .expect("Failed to send data");
    println!("Connected to server!");
    let (game_sender, game_receiver) = flume::bounded::<CollectorMessage>(1);
    let num_threads = 32;
    let num_executors = 2;
    thread::scope(|s| {
        let mut selfplay_masters: Vec<DataGen> = Vec::new();
        // commander

        let mut vec_communicate_exe_send: Vec<Sender<String>> = Vec::new();
        let mut vec_communicate_exe_recv: Vec<Receiver<String>> = Vec::new();

        for _ in 0..num_executors {
            let (communicate_exe_send, communicate_exe_recv) = flume::bounded::<String>(1);
            vec_communicate_exe_send.push(communicate_exe_send);
            vec_communicate_exe_recv.push(communicate_exe_recv);
        }

        // send-recv pair between commander and collector

        let (id_send, id_recv) = flume::bounded::<String>(1);

        s.builder()
            .name("commander".to_string())
            .spawn(|_| {
                commander_main(
                    vec_communicate_exe_send,
                    &mut stream.try_clone().expect("clone failed"),
                    id_send,
                )
            })
            .unwrap();
        // selfplay threads
        let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Packet>(1); // mcts to executor
        for n in 0..num_threads {
            // // executor
            // sender-receiver pair to communicate for each thread instance to the executor
            let sender_clone = game_sender.clone();
            let mut selfplay_master = DataGen { iterations: 1 };
            let tensor_exe_send_clone = tensor_exe_send.clone();
            let nps_sender = game_sender.clone();
            s.builder()
                .name(format!("generator_{}", n.to_string()))
                .spawn(move |_| {
                    generator_main(
                        &sender_clone,
                        &mut selfplay_master,
                        tensor_exe_send_clone,
                        nps_sender,
                    )
                })
                .unwrap();

            selfplay_masters.push(selfplay_master.clone());
        }
        // collector

        s.builder()
            .name("collector".to_string())
            .spawn(|_| {
                collector_main(
                    &game_receiver,
                    &mut stream.try_clone().expect("clone failed"),
                    id_recv,
                )
            })
            .unwrap();
        // executor
        let mut n = 0;
        for communicate_exe_recv in vec_communicate_exe_recv {
            // send/recv pair between executor and commander
            let eval_per_sec_sender = game_sender.clone();
            let tensor_exe_recv_clone = tensor_exe_recv.clone();
            s.builder()
                .name(format!("executor_{}", n.to_string()))
                .spawn(move |_| {
                    executor_main(
                        communicate_exe_recv,
                        tensor_exe_recv_clone,
                        num_threads / num_executors,
                        eval_per_sec_sender,
                    )
                })
                .unwrap();
            n += 1;
        }
    })
    .unwrap();
}

fn generator_main(
    sender_collector: &Sender<CollectorMessage>,
    datagen: &mut DataGen,
    tensor_exe_send: Sender<Packet>,
    nps_sender: Sender<CollectorMessage>,
) {
    loop {
        let sim = datagen.play_game(tensor_exe_send.clone(), nps_sender.clone());
        sender_collector
            .send(CollectorMessage::FinishedGame(sim))
            .unwrap();
    }
}

fn collector_main(
    receiver: &Receiver<CollectorMessage>,
    server_handle: &mut TcpStream,
    id_recv: Receiver<String>,
) {
    let thread_name = std::thread::current()
        .name()
        .unwrap_or("unnamed")
        .to_owned();
    let folder_name = "games"; // Change this to your desired folder name

    if let Err(e) = fs::create_dir(folder_name) {
        match e.kind() {
            std::io::ErrorKind::AlreadyExists => {}
            _ => {
                eprintln!("Error creating folder: {}", e);
            }
        }
    } else {
        println!("created {}", folder_name);
    }
    let mut counter = 0;
    let id = id_recv.recv().unwrap();

    let mut path = format!("games/generator_{}_games_{}", id, counter);
    println!("collector path: {}", path);
    let mut bin_output = BinaryOutput::new(path.clone(), "chess").unwrap();
    let mut nps_start_time = Instant::now();
    let mut nps_vec: Vec<f32> = Vec::new();
    let mut evals_start_time = Instant::now();
    let mut evals_vec: Vec<f32> = Vec::new();

    loop {
        let msg = receiver.recv().unwrap();
        match msg {
            CollectorMessage::FinishedGame(sim) => {
                let _ = bin_output.append(&sim).unwrap();
                if bin_output.game_count() >= 5 {
                    let _ = bin_output.finish().unwrap();
                    let message = format!("new-training-data: {}.json", path.clone());
                    // TODO: keep a vec of all data training files sent, reset every time when python completes a training loop
                    // also filter for whether the files still exist in the meantime?
                    // maybe better to do checking files still exist in python better, if done here, it may affect nps/eval scores
                    server_handle.write_all(message.as_bytes()).unwrap();
                    println!("{}, {}", thread_name, counter);
                    counter += 1;
                    path = format!("games/generator_{}_games_{}", id, counter);
                    bin_output = BinaryOutput::new(path.clone(), "chess").unwrap();
                }
            }
            CollectorMessage::GeneratorStatistics(nps) => {
                if nps_start_time.elapsed() >= Duration::from_secs(1) {
                    let nps: f32 = nps_vec.iter().sum();
                    // println!("{} nps", nps);
                    nps_start_time = Instant::now();
                    nps_vec = Vec::new();
                } else {
                    nps_vec.push(nps);
                }
            }
            CollectorMessage::ExecutorStatistics(evals_per_sec) => {
                if evals_start_time.elapsed() >= Duration::from_secs(1) {
                    let nps: f32 = evals_vec.iter().sum();
                    // println!("{} evals/s", nps);
                    evals_start_time = Instant::now();
                    evals_vec = Vec::new();
                } else {
                    evals_vec.push(evals_per_sec);
                }
            }
        }
    }
}
fn commander_main(
    vec_exe_sender: Vec<Sender<String>>,
    server_handle: &mut TcpStream,
    id_sender: Sender<String>,
) {
    let mut curr_net = String::new();
    let mut buffer = [0; 16384];
    let mut is_initialised = false;
    let mut net_path = String::new(); // initialize net_path with an empty string
    loop {
        println!("is_initialised: {}", is_initialised);
        let recv_msg = match server_handle.read(&mut buffer) {
            Ok(msg) if msg == 0 => {
                // no more data, connection closed by server
                println!("Server closed the connection");
                // force quit the program when the server closes the connection
                std::process::exit(0);
            }
            Ok(id) if !is_initialised => {
                let msg = String::from_utf8(buffer[..].to_vec()).unwrap(); // convert received bytes to String
                if msg.starts_with("rust-datagen") {
                    id // Explicitly return the value
                } else {
                    continue;
                }
            }

            Ok(recv_net) => {
                let msg = String::from_utf8(buffer[..].to_vec()).unwrap(); // convert received bytes to String
                if msg.starts_with("newnet") {
                    // println!("found net path: {}", msg);
                    recv_net // Explicitly return the value
                } else {
                    continue;
                }
            }

            Err(err) => {
                eprintln!("Error reading from server: {}", err);
                break;
            }
        };

        if is_initialised {
            net_path = String::from_utf8(buffer[..recv_msg].to_vec()).unwrap();
            let segment: Vec<String> = net_path.split_whitespace().map(String::from).collect();
            net_path = segment[1].clone();
            net_path.retain(|c| c != '\0'); // remove invalid chars
        } else {
            let msg = String::from_utf8(buffer[..recv_msg].to_vec()).unwrap();
            if msg.starts_with("rust-datagen") {
                let segment: Vec<String> = msg.split_whitespace().map(String::from).collect();
                let mut id = segment[1].clone();
                id.retain(|c| c != '\0'); // remove invalid chars
                                          // send to collector

                id_sender.send(id).unwrap();

                is_initialised = true;
            } else {
                // force quit, there is no ID, hence potentially overwriting existing game files should this process continue
                std::process::exit(0);
            }
        }

        if curr_net != net_path {
            if !net_path.is_empty() {
                println!("updating net to: {}", net_path.clone());
                for exe_sender in &vec_exe_sender {
                    exe_sender.send(net_path.clone()).unwrap();
                    // println!("SENT!");
                }

                curr_net = net_path.clone();
            }
        } 
        if net_path.is_empty() {
            println!("no net yet");
            // actively request for net path
            server_handle
                .write_all("requesting-net".as_bytes())
                .unwrap();
        }

        buffer = [0; 16384];
    }

    // If the loop exits unexpectedly, force quit the program
    println!("Exiting program due to unexpected server disconnection");
    std::process::exit(0);
}
