use crossbeam::thread;
use flume::{Receiver, Sender};
use futures::executor::ThreadPool;
use rand::prelude::*;
use std::{
    cmp::{max, min},
    env,
    fs::{self, File},
    io::{self, BufRead, BufReader, Read, Write},
    net::TcpStream,
    panic,
    path::Path,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tz_rust::{
    dummyreq::{send_request, send_request_async},
    executor::{executor_main, Packet},
    fileformat::BinaryOutput,
    mcts_trainer::TypeRequest::TrainerSearch,
    message_types::{DataFileType, Entity, MessageServer, MessageType, Statistics},
    selfplay::{CollectorMessage, DataGen},
    settings::SearchSettings,
};
fn main() {
    let pool = ThreadPool::new().expect("Failed to build pool");
    env::set_var("RUST_BACKTRACE", "2");

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
        }
    };
    // identification - this is rust data generation
    let message = MessageServer {
        purpose: MessageType::Initialise(Entity::RustDataGen),
    };
    let serialised = serde_json::to_string(&message).expect("serialization failed");
    let mut serialised = serialised + "\n";
    stream
        .write_all(serialised.as_bytes())
        .expect("Failed to send data");
    println!("Connected to server!");

    let mut num_executors = 2;
    let batch_size = 1024; // executor batch size
    let num_generators = num_executors * batch_size * 2;
    num_executors = max(min(tch::Cuda::device_count() as usize, num_executors), 1);
    let (game_sender, game_receiver) =
        flume::bounded::<CollectorMessage>(num_executors * batch_size);

    thread::scope(|s| {
        // commander
        let mut vec_communicate_exe_send: Vec<Sender<String>> = Vec::new(); // commander to executor, send net
        let mut vec_communicate_exe_recv: Vec<Receiver<String>> = Vec::new(); // commander to executor, send net

        for _ in 0..num_executors {
            let (communicate_exe_send, communicate_exe_recv) =
                flume::bounded::<String>(num_generators);
            vec_communicate_exe_send.push(communicate_exe_send);
            vec_communicate_exe_recv.push(communicate_exe_recv);
        }

        // send-recv pair between commander and collector
        let (id_send, id_recv) = flume::bounded::<usize>(1);

        let _ = s
            .builder()
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
        let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Packet>(num_generators); // mcts to executor

        // executor
        let mut exec_id = 0;
        for communicate_exe_recv in vec_communicate_exe_recv {
            let eval_per_sec_sender = game_sender.clone();
            let tensor_exe_recv_clone = tensor_exe_recv.clone();
            let _ = s
                .builder()
                .name(format!("executor-{}", exec_id).to_string())
                .spawn(move |_| {
                    executor_main(
                        communicate_exe_recv,
                        tensor_exe_recv_clone,
                        batch_size,
                        eval_per_sec_sender,
                        exec_id,
                    )
                })
                .unwrap();
            exec_id += 1;
        }

        for n in 0..num_generators {
            let sender_clone = game_sender.clone();
            let mut selfplay_master = DataGen { iterations: 1 };
            let tensor_exe_send_clone = tensor_exe_send.clone();
            let fut_generator = async move {
                generator_main(sender_clone, selfplay_master, tensor_exe_send_clone, n).await;
            };
            pool.spawn_ok(fut_generator);
        }

        // collector
        let _ = s
            .builder()
            .name("collector".to_string())
            .spawn(|_| {
                collector_main(
                    &game_receiver,
                    &mut stream.try_clone().expect("clone failed"),
                    id_recv,
                )
            })
            .unwrap();
    })
    .unwrap();
}

async fn generator_main(
    sender_collector: Sender<CollectorMessage>,
    datagen: DataGen,
    tensor_exe_send: Sender<Packet>,
    id: usize,
) {
    let settings: SearchSettings = SearchSettings {
        fpu: 0.0,
        wdl: None,
        moves_left: None,
        c_puct: 3.0,
        max_nodes: 800,
        alpha: 0.3,
        eps: 0.3,
        search_type: TrainerSearch(None),
        pst: 1.25,
    };
    let nps_sender = sender_collector.clone();
    loop {
        let sim = datagen
            // .fast_data(&tensor_exe_send, &nps_sender, &settings, id)
            .play_game(&tensor_exe_send, &nps_sender, &settings, id)
            .await;

        // match settings.search_type {
        //     TrainerSearch(expansiontype) => match expansiontype {
        //         Some(_) => {
        //             let num_positions_sample =
        //                 rand::thread_rng().gen_range(1..=sim.positions.len());
        //             let mut rng = rand::thread_rng();
        //             let random_idx: Vec<usize> = (0..num_positions_sample)
        //                 .map(|_| rng.gen_range(0..=sim.positions.len() - 1))
        //                 .collect();

        //             for position in random_idx {
        //                 let sim = synthetic_expansion(
        //                     sim.clone(),
        //                     position,
        //                     tensor_exe_send.clone(),
        //                     settings.clone(),
        //                 );
        //                 sender_collector
        //                     .send(CollectorMessage::FinishedGame(sim))
        //                     .unwrap();
        //             }
        //         }
        //         None => {}
        //     },
        //     tz_rust::mcts_trainer::TypeRequest::SyntheticSearch => {}
        //     tz_rust::mcts_trainer::TypeRequest::NonTrainerSearch => {}
        //     tz_rust::mcts_trainer::TypeRequest::UCISearch => {}
        // }

        sender_collector
            .send_async(CollectorMessage::FinishedGame(sim))
            .await
            .unwrap();
    }
}

fn serialise_file_to_bytes(file_path: &str) -> io::Result<Vec<u8>> {
    let mut file = File::open(file_path)?;

    let metadata = file.metadata()?;
    let file_size = metadata.len() as usize;

    let mut buffer = Vec::with_capacity(file_size);

    file.read_to_end(&mut buffer)?;

    Ok(buffer)
}

fn collector_main(
    receiver: &Receiver<CollectorMessage>,
    server_handle: &mut TcpStream,
    id_recv: Receiver<usize>,
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
                println!("Error creating folder: {}", e);
            }
        }
    } else {
        println!("created {}", folder_name);
    }
    let id = id_recv.recv().unwrap();
    let file_save_time = SystemTime::now();
    let file_save_time_duration = file_save_time
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let file_save_time_num = file_save_time_duration.as_nanos();
    let mut path = format!("games/gen_{}_games_{}", id, file_save_time_num);
    let mut bin_output = BinaryOutput::new(path.clone(), "chess").unwrap();
    let mut nps_start_time = Instant::now();
    let mut nps_vec: Vec<f32> = Vec::new();
    let mut evals_start_time = Instant::now();
    let mut evals_vec: Vec<f32> = Vec::new();
    let files = [".bin", ".off", ".json"];
    loop {
        let msg = receiver.recv().unwrap();
        match msg {
            CollectorMessage::FinishedGame(sim) => {
                bin_output.append(&sim).unwrap();
                if bin_output.game_count() >= 100 {
                    bin_output.finish().unwrap();
                    let mut file_data: Vec<Vec<u8>> = Vec::new(); // Clear file_data vector
                    for file in files {
                        let file_path = format!("{}{}", path, file);
                        let data = serialise_file_to_bytes(&file_path.to_owned())
                            .unwrap()
                            .clone();
                        file_data.push(data);
                    }

                    let (bin_file, off_file, metadata) = (
                        file_data[0].clone(),
                        file_data[1].clone(),
                        file_data[2].clone(),
                    );

                    let message = MessageServer {
                        // purpose: MessageType::JobSendPath(path.clone().to_string()),
                        purpose: MessageType::JobSendData(vec![
                            DataFileType::BinFile(bin_file),
                            DataFileType::OffFile(off_file),
                            DataFileType::MetaDataFile(metadata),
                        ]),
                    };
                    let mut serialised =
                        serde_json::to_string(&message).expect("serialisation failed");
                    serialised += "\n";
                    server_handle.write_all(serialised.as_bytes()).unwrap();
                    // for file_path in files {
                    //     let exists_file = Path::new(file_path).is_file();
                    //     if exists_file {
                    //         // delete sent files
                    //         // Attempt to delete the file
                    //         match fs::remove_file(format!("{}{}", path, file_path)) {
                    //             Ok(_) => {
                    //                 println!("Deleted file {}", format!("{}{}", path, file_path));
                    //             }
                    //             Err(e) => println!("Error deleting the file: {}", e),
                    //         }
                    //     } else {
                    //     }
                    // }
                    // println!("{}, {}", thread_name, counter);
                    let file_save_time = SystemTime::now();
                    let file_save_time_duration = file_save_time
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards");
                    let file_save_time_num = file_save_time_duration.as_nanos();
                    path = format!("games/gen_{}_games_{}", id, file_save_time_num);
                    bin_output = BinaryOutput::new(path.clone(), "chess").unwrap();
                }
            }
            CollectorMessage::GeneratorStatistics(nps) => {
                if nps_start_time.elapsed() >= Duration::from_secs(1) {
                    let nps: f32 = nps_vec.iter().sum();
                    // println!("{} nps", nps);
                    nps_start_time = Instant::now();
                    nps_vec = Vec::new();

                    let message = MessageServer {
                        purpose: MessageType::StatisticsSend(Statistics::NodesPerSecond(nps)),
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
                    let message = MessageServer {
                        purpose: MessageType::StatisticsSend(Statistics::EvalsPerSecond(
                            evals_per_second,
                        )),
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
            CollectorMessage::TestingResult(_) => {}
        }
    }
}

fn commander_main(
    vec_exe_sender: Vec<Sender<String>>,
    server_handle: &mut TcpStream,
    id_sender: Sender<usize>,
) {
    let mut curr_net = String::new();
    let mut is_initialised = false;
    let mut net_path = String::new(); // initialize net_path with an empty string
    let mut cloned_handle = server_handle.try_clone().unwrap();
    let mut reader = BufReader::new(server_handle);
    let mut net_path_counter = 0;
    let mut generator_id: usize = 0;
    loop {
        let mut recv_msg = String::new();
        if let Err(_) = reader.read_line(&mut recv_msg) {
            return;
        }
        // deserialise data
        // println!("RAW message: {:?}", recv_msg);
        let message = match serde_json::from_str::<MessageServer>(&recv_msg) {
            Ok(message) => {
                message
                // process the received JSON data
            }
            Err(err) => {
                // println!("error deserialising message! {}, {}", recv_msg, err);
                recv_msg.clear();
                continue;
            }
        };

        if is_initialised {
            match message.purpose {
                MessageType::Initialise(_) => {}
                MessageType::JobSendPath(_) => {}
                MessageType::StatisticsSend(_) => {}
                MessageType::RequestingNet => {}
                MessageType::NewNetworkPath(path) => {}
                MessageType::IdentityConfirmation(_) => {}
                MessageType::JobSendData(_) => {}
                MessageType::NewNetworkData(data) => {
                    println!("new net path");
                    net_path = format!("nets/tz_temp_net_{}_{}.pt", generator_id, net_path_counter);
                    let mut file = File::create(net_path.clone()).expect("Unable to create file");
                    file.write_all(&data).expect("Unable to write data");
                    net_path_counter += 1;
                }
                MessageType::TBLink(_) => {}
                MessageType::CreateTB => {}
                MessageType::RequestingTBLink => {}
            }
        } else {
            match message.purpose {
                MessageType::Initialise(_) => {}
                MessageType::JobSendPath(_) => {}
                MessageType::StatisticsSend(_) => {}
                MessageType::RequestingNet => {}
                MessageType::NewNetworkPath(_) => {}
                MessageType::IdentityConfirmation((entity, id)) => match entity {
                    Entity::RustDataGen => {
                        generator_id = id;
                        id_sender.send(id).unwrap();
                        is_initialised = true;
                    }
                    Entity::PythonTraining => {
                        println!("[Warning] Wrong entity, got {:?}", entity)
                    }
                    Entity::GUIMonitor => {
                        println!("[Warning] Wrong entity, got {:?}", entity)
                    }
                    Entity::TBHost => {
                        println!("[Warning] Wrong entity, got {:?}", entity)
                    }
                },
                MessageType::JobSendData(_) => {}
                MessageType::NewNetworkData(_) => {}
                MessageType::TBLink(_) => {}
                MessageType::CreateTB => {}
                MessageType::RequestingTBLink => {}
            }
        }

        if curr_net != net_path {
            if !net_path.is_empty() {
                println!("updating net to: {}", net_path.clone());
                let exists_file = Path::new(&curr_net).is_file();
                if exists_file {
                    match fs::remove_file(curr_net.clone()) {
                        Ok(_) => {
                            println!("Deleted net {}", curr_net);
                        }
                        Err(e) => eprintln!("Error deleting the file: {}", e),
                    }
                }
                for exe_sender in &vec_exe_sender {
                    exe_sender.send(net_path.clone()).unwrap();
                    // println!("SENT!");
                }
                curr_net = net_path.clone();
            }
        }
        if net_path.is_empty() {
            // println!("no net yet");
            // actively request for net path

            let message = MessageServer {
                purpose: MessageType::RequestingNet,
            };
            let mut serialised = serde_json::to_string(&message).expect("serialisation failed");
            serialised += "\n";
            // println!("serialised {:?}", serialised);
            cloned_handle.write_all(serialised.as_bytes()).unwrap();
        }
        recv_msg.clear();
    }
}
