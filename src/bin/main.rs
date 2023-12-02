use crossbeam::thread;
use flume::{Receiver, Sender};
use std::{
    env,
    io::{Read, Write},
    net::TcpStream,
    time::{Duration, Instant},
};
use tz_rust::{
    executor::{executor_main, Packet},
    fileformat::BinaryOutput,
    selfplay::{CollectorMessage, DataGen},
};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
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
    let mut buffer = [0; 16384];
    let id: String = loop {
        match stream.read(&mut buffer) {
            Ok(recv_net) if recv_net == 0 => {
                // no more data, connection closed by server
                println!("Server closed the connection");
                // Force quit the program when the server closes the connection
                std::process::exit(0);
            }
            Ok(_) => {
                let msg = String::from_utf8_lossy(&buffer[..]); // convert received bytes to String
                if msg.starts_with("rust-datagen") {
                    let segment: Vec<String> = msg.split_whitespace().map(String::from).collect();
                    break segment[1].clone();
                }
            }
            Err(err) => {
                panic!("Error reading from server: {}", err);
            }
        };
    };

    println!("MY ID {}", id);

    let (game_sender, game_receiver) = flume::bounded::<CollectorMessage>(1);
    let num_threads = 5;
    let num_executors = 1;
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

        s.builder()
            .name("commander".to_string())
            .spawn(|_| {
                commander_main(
                    vec_communicate_exe_send,
                    &mut stream.try_clone().expect("clone failed"),
                )
            })
            .unwrap();

        // s.spawn(move |_| {
        //     commander_main(communicate_exe_send);
        // });
        // selfplay threads
        // let mut exe_resenders: Vec<Sender<Message>> = Vec::new(); // sent FROM executor to mcts
        // let mut tensor_senders: Vec<Receiver<Tensor>> = Vec::new(); // sent FROM mcts to executor
        let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Packet>(1); // mcts to executor
                                                                              // let (eval_exe_send, eval_exe_recv) = flume::bounded::<Message>(1); // executor to mcts
                                                                              // let mut exe_count = 0;
        for n in 0..num_threads {
            // // executor
            // if n % num_executors == 0 {
            //     let communicate_exe_recv_clone = communicate_exe_recv.clone();
            //     let tensor_exe_recv_clone = tensor_exe_recv.clone();
            //     s.builder()
            //     .name(format!("executor_{}", exe_count.to_string()))
            //     .spawn(move |_| executor_main(communicate_exe_recv_clone, tensor_exe_recv_clone, num_threads))
            //     .unwrap();
            //     exe_count += 1;
            // }
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
            // exe_resenders.push(eval_exe_send);
            // tensor_senders.push(tensor_exe_recv);
            // s.spawn(move |_| {
            //     generator_main(&sender_clone, &mut selfplay_master);
            // });

            selfplay_masters.push(selfplay_master.clone());
        }
        // collector

        s.builder()
            .name("collector".to_string())
            .spawn(|_| {
                collector_main(
                    &game_receiver,
                    &mut stream.try_clone().expect("clone failed"),
                )
            })
            .unwrap();

        // s.spawn(move |_| {
        //     collector_main(&game_receiver);
        // });
        // executor
        let mut n = 0;
        for communicate_exe_recv in vec_communicate_exe_recv {
            // send/recv pair between executor and commander
            let eval_per_sec_sender = game_sender.clone();
            // let communicate_exe_recv_clone = communicate_exe_recv.clone();
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

        // s.spawn(move |_| {
        //     executor_main(communicate_exe_recv);
        // });
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

fn collector_main(receiver: &Receiver<CollectorMessage>, server_handle: &mut TcpStream) {
    let mut counter = 0;
    let mut path = format!("games_{}", counter);
    let mut bin_output = BinaryOutput::new(path.clone(), "chess").unwrap();
    let mut nps_start_time = Instant::now();
    let mut evals_start_time = Instant::now();
    let mut nps_vec: Vec<f32> = Vec::new();

    loop {
        let msg = receiver.recv().unwrap();
        match msg {
            CollectorMessage::FinishedGame(sim) => {
                let _ = bin_output.append(&sim).unwrap();
                println!("{}", bin_output.game_count());
                if bin_output.game_count() >= 5 {
                    let _ = bin_output.finish().unwrap();
                    let message = format!("new-training-data: {}.json", path.clone());
                    server_handle.write_all(message.as_bytes()).unwrap();
                    counter += 1;
                    path = format!("games_{}", counter);
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
                    let nps: f32 = nps_vec.iter().sum();
                    // println!("{} evals/s", nps);
                    evals_start_time = Instant::now();
                    nps_vec = Vec::new();
                } else {
                    nps_vec.push(evals_per_sec);
                }
            }
        }
    }
}
fn commander_main(vec_exe_sender: Vec<Sender<String>>, server_handle: &mut TcpStream) {
    let mut curr_net = String::new(); // changed to String type
    let mut buffer = [0; 16384];
    loop {
        let recv_net = match server_handle.read(&mut buffer) {
            Ok(recv_net) if recv_net == 0 => {
                // no more data, connection closed by server
                println!("Server closed the connection");
                // Force quit the program when the server closes the connection
                std::process::exit(0);
            }
            Ok(recv_net) => recv_net,
            Err(err) => {
                eprintln!("Error reading from server: {}", err);
                break;
            }
        };

        let net_path = String::from_utf8_lossy(&buffer[..recv_net]).to_string(); // convert received bytes to String

        if curr_net != net_path {
            for exe_sender in &vec_exe_sender {
                exe_sender
                    .send("chess_16x128_gen3634.pt".to_string())
                    .unwrap();
                // println!("SENT!");
            }

            curr_net = net_path;
        }

        buffer = [0; 16384];
    }

    // If the loop exits unexpectedly, force quit the program
    println!("Exiting program due to unexpected server disconnection");
    std::process::exit(0);
}
