use crossbeam::thread;
use flume::{Receiver, Sender};
use futures::executor::ThreadPool;
use lru::LruCache;
use std::{
    cmp::{max, min},
    env,
    fs::{self, File},
    io::{self, BufRead, BufReader, Read, Write},
    net::TcpStream,
    num::NonZeroUsize,
    panic,
    path::Path,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tch::Tensor;
use tzrust::{
    cache::CacheEntryKey,
    executor::{executor_main, Packet},
    fileformat::BinaryOutput,
    mcts_trainer::{EvalMode, TypeRequest::TrainerSearch},
    message_types::{DataFileType, Entity, MessageServer, MessageType, Statistics},
    selfplay::{CollectorMessage, DataGen},
    settings::SearchSettings,
};

fn main() {
    let pool = ThreadPool::new().expect("Failed to build pool");
    env::set_var("RUST_BACKTRACE", "2");

    panic::set_hook(Box::new(|panic_info| {
        eprintln!("Panic occurred: {:?}", panic_info);
        std::process::exit(1);
    }));

    let mut stream = loop {
        match TcpStream::connect("127.0.0.1:38475") {
            Ok(s) => break s,
            Err(_) => continue,
        }
    };

    let message = MessageServer {
        purpose: MessageType::Initialise(Entity::RustDataGen),
    };
    let serialised = serde_json::to_string(&message).expect("serialisation failed");
    let mut serialised = serialised + "\n";
    stream
        .write_all(serialised.as_bytes())
        .expect("Failed to send data");
    println!("Connected to server!");

    let mut num_executors = 2;
    // num_executors = max(min(tch::Cuda::device_count() as usize, num_executors), 1);
    let batch_size = 64;
    let num_generators = num_executors * batch_size * 2;

    let (game_sender, game_receiver) = flume::bounded::<CollectorMessage>(num_generators);

    thread::scope(|s| {
        let mut vec_communicate_exe_send: Vec<Sender<String>> = Vec::new();
        let mut vec_communicate_exe_recv: Vec<Receiver<String>> = Vec::new();

        for _ in 0..num_executors {
            let (communicate_exe_send, communicate_exe_recv) =
                flume::bounded::<String>(num_generators);
            vec_communicate_exe_send.push(communicate_exe_send);
            vec_communicate_exe_recv.push(communicate_exe_recv);
        }

        let (id_send, id_recv) = flume::bounded::<usize>(1);
        // selfplay send/recv pair
        let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Packet>(num_generators); // mcts to executor

        // spawn executor threads

        for (exec_id, communicate_exe_recv) in vec_communicate_exe_recv.into_iter().enumerate() {
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
        }

        // commander will handle evaluation requests as well

        let _ = s
            .builder()
            .name("commander".to_string())
            .spawn(|_| {
                commander_executor(
                    vec_communicate_exe_send,
                    &mut stream.try_clone().expect("clone failed"),
                    id_send,
                    tensor_exe_send,
                )
            })
            .unwrap();
    })
    .unwrap();
}

fn serialise_file_to_bytes(file_path: &str) -> io::Result<Vec<u8>> {
    let mut file = File::open(file_path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len() as usize;
    let mut buffer = Vec::with_capacity(file_size);
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

fn directory_exists<P: AsRef<Path>>(path: P) -> bool {
    match fs::metadata(path) {
        Ok(metadata) => metadata.is_dir(),
        Err(_) => false,
    }
}

fn commander_executor(
    vec_exe_sender: Vec<Sender<String>>,
    server_handle: &mut TcpStream,
    id_sender: Sender<usize>,
    tensor_sender: Sender<Packet>,
) {
    let mut curr_net = String::new();
    let mut is_initialised = false;
    let mut net_path = String::new(); // initialize net_path with an empty string
    let mut cloned_handle = server_handle.try_clone().unwrap();
    let mut reader = BufReader::new(server_handle);
    let mut net_path_counter = 0;
    let mut generator_id: usize = 0;
    loop {
        if !directory_exists("nets") {
            fs::create_dir("nets").unwrap();
        }
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
                MessageType::RequestingNet() => {}
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
                MessageType::CreateTB() => {}
                MessageType::RequestingTBLink() => {}
                MessageType::EvaluationRequest(input_data) => {
                    let tensor_packet = Packet {
                        job: Tensor::from_slice(&input_data.data),
                        resender: todo!(),
                        id: todo!(),
                    };
                }
            }
        } else {
            match message.purpose {
                MessageType::Initialise(_) => {}
                MessageType::JobSendPath(_) => {}
                MessageType::StatisticsSend(_) => {}
                MessageType::RequestingNet() => {}
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
                MessageType::CreateTB() => {}
                MessageType::RequestingTBLink() => {}
                MessageType::EvaluationRequest(_) => {}
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
                purpose: MessageType::RequestingNet(),
            };
            let mut serialised = serde_json::to_string(&message).expect("serialisation failed");
            serialised += "\n";
            // println!("serialised {:?}", serialised);
            cloned_handle.write_all(serialised.as_bytes()).unwrap();
        }
        recv_msg.clear();
    }
}
