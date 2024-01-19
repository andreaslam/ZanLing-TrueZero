use crossbeam::thread;
use flume::{Receiver, Sender};
use rand::prelude::*;
use rand_distr::WeightedIndex;
use std::{
    env, fs,
    io::{BufRead, BufReader, Write},
    net::TcpStream,
    panic,
    time::{Duration, Instant},
};
use tz_rust::{
    executor::{executor_main, Packet},
    fileformat::BinaryOutput,
    mcts_trainer::TypeRequest::TrainerSearch,
    message_types::{ServerMessageRecv, ServerMessageSend},
    selfplay::{synthetic_expansion, CollectorMessage, DataGen},
    settings::SearchSettings,
};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    panic::set_hook(Box::new(|panic_info| {
        // print panic information
        eprintln!("Panic occurred: {:?}", panic_info);
        // exit the program immediately
        std::process::exit(1);
    }));
    // connect to python-rust server
    let mut stream = loop {
        match TcpStream::connect("127.0.0.1:38475") {
            Ok(s) => break s,
            Err(_) => continue,
        };
    };
    // identification - this is rust data generation

    let message = ServerMessageSend {
        is_continue: true,
        initialise_identity: Some("rust-datagen".to_string()),
        nps: None,
        evals_per_second: None,
        job_path: None,
        net_path: None,
        has_net: false,
        purpose: "initialise".to_string(),
    };
    let mut serialised = serde_json::to_string(&message).expect("serialisation failed");
    serialised += "\n";
    stream
        .write_all(serialised.as_bytes())
        .expect("Failed to send data");
    println!("Connected to server!");
    let (game_sender, game_receiver) = flume::bounded::<CollectorMessage>(1);
    let num_threads = 512;
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
    let settings: SearchSettings = SearchSettings {
        fpu: 0.0,
        wdl: None,
        moves_left: None,
        c_puct: 2.0,
        max_nodes: 400,
        alpha: 0.3,
        eps: 0.3,
        search_type: TrainerSearch(None),
    };
    loop {
        let sim = datagen.play_game(
            tensor_exe_send.clone(),
            nps_sender.clone(),
            settings.clone(),
        );

        match settings.search_type {
            TrainerSearch(expansiontype) => match expansiontype {
                Some(_) => {
                    let num_positions_sample =
                        rand::thread_rng().gen_range(1..=sim.positions.len());
                    let mut rng = rand::thread_rng();
                    let random_idx: Vec<usize> = (0..num_positions_sample)
                        .map(|_| rng.gen_range(0..=sim.positions.len() - 1))
                        .collect();

                    for position in random_idx {
                        let sim = synthetic_expansion(
                            sim.clone(),
                            position,
                            tensor_exe_send.clone(),
                            settings.clone(),
                        );
                        sender_collector
                            .send(CollectorMessage::FinishedGame(sim))
                            .unwrap();
                    }
                }
                None => {}
            },
            tz_rust::mcts_trainer::TypeRequest::SyntheticSearch => {}
            tz_rust::mcts_trainer::TypeRequest::NonTrainerSearch => {}
            tz_rust::mcts_trainer::TypeRequest::UCISearch => {}
        }

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
    let folder_name = "games";

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
                if bin_output.game_count() >= 100 {
                    let _ = bin_output.finish().unwrap();
                    let message = ServerMessageSend {
                        is_continue: true,
                        initialise_identity: None,
                        nps: None,
                        evals_per_second: None,
                        job_path: Some(path.clone().to_string()),
                        net_path: None,
                        has_net: true,
                        purpose: "jobsend".to_string(),
                    };
                    let mut serialised =
                        serde_json::to_string(&message).expect("serialisation failed");
                    serialised += "\n";
                    server_handle.write_all(serialised.as_bytes()).unwrap();
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
                    // let message = format!("statistics-nps: {}\n", nps.clone());
                    let message = ServerMessageSend {
                        is_continue: true,
                        initialise_identity: None,
                        nps: Some(nps),
                        job_path: None,
                        evals_per_second: None,
                        net_path: None,
                        has_net: true,
                        purpose: "statistics-nps".to_string(),
                    };
                    let mut serialised =
                        serde_json::to_string(&message).expect("serialisation failed");
                    serialised += "\n";
                    server_handle.write_all(serialised.as_bytes()).unwrap();
                } else {
                    nps_vec.push(nps);
                }
            }
            CollectorMessage::ExecutorStatistics(evals_per_sec) => {
                if evals_start_time.elapsed() >= Duration::from_secs(1) {
                    let evals_per_second: f32 = evals_vec.iter().sum();
                    // println!("{} evals/s", evals_per_second);
                    evals_start_time = Instant::now();
                    evals_vec = Vec::new();
                    let message = ServerMessageSend {
                        is_continue: true,
                        initialise_identity: None,
                        nps: None,
                        evals_per_second: Some(evals_per_second),
                        job_path: None,
                        net_path: None,
                        has_net: true,
                        purpose: "statistics-evals".to_string(),
                    };
                    let mut serialised =
                        serde_json::to_string(&message).expect("serialisation failed");
                    serialised += "\n";
                    server_handle.write_all(serialised.as_bytes()).unwrap();
                } else {
                    evals_vec.push(evals_per_sec);
                }
            }
            CollectorMessage::GameResult(_) => {}
        }
    }
}
fn commander_main(
    vec_exe_sender: Vec<Sender<String>>,
    server_handle: &mut TcpStream,
    id_sender: Sender<String>,
) {
    let mut curr_net = String::new();
    let mut is_initialised = false;
    let mut net_path = String::new(); // initialize net_path with an empty string
    let mut cloned_handle = server_handle.try_clone().unwrap();
    let mut reader = BufReader::new(server_handle);
    loop {
        let mut recv_msg = String::new();
        if let Err(_) = reader.read_line(&mut recv_msg) {
            return;
        }
        // deserialise data
        // println!("RAW message: {:?}", recv_msg);
        let message = match serde_json::from_str::<ServerMessageRecv>(&recv_msg) {
            Ok(message) => {
                message
                // process the received JSON data
            }
            Err(err) => {
                // println!("error deserialising message! {}, {}", recv_msg, err);
                continue;
            }
        };

        if is_initialised {
            let np = message.net_path;
            match np {
                Some(net_p) => {
                    net_path = net_p;
                    net_path.retain(|c| c != '\0'); // remove invalid chars
                }
                None => {
                    continue;
                }
            }
        } else {
            let msg = message.verification.unwrap();
            if msg.starts_with("rust-datagen") {
                let segment: Vec<String> = msg.split_whitespace().map(String::from).collect();
                let mut id = segment[1].clone();
                id.retain(|c| c != '\n'); // remove invalid chars
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

            let message = ServerMessageSend {
                is_continue: true,
                initialise_identity: None,
                nps: None,
                evals_per_second: None,
                job_path: None,
                net_path: None,
                has_net: false,
                purpose: "requesting-net".to_string(),
            };
            let mut serialised = serde_json::to_string(&message).expect("serialisation failed");
            serialised += "\n";
            println!("serialised {:?}", serialised);
            cloned_handle.write_all(serialised.as_bytes()).unwrap();
        }
        recv_msg.clear();
    }
}
