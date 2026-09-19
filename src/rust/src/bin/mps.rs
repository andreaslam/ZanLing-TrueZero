// use crossbeam::thread;
// use flume::{Receiver, Sender};
// use futures::executor::ThreadPool;
// use lru::LruCache;
// use sha2::{Digest, Sha256};
// use std::{
//     env,
//     fs::{self, File},
//     io::{self, BufRead, BufReader, Read, Write},
//     net::TcpStream,
//     num::NonZeroUsize,
//     panic,
//     path::Path,
//     thread as std_thread,
//     time::{Duration, Instant, SystemTime, UNIX_EPOCH},
// };
// use tch::Tensor;
// use tzrust::{
//     cache::CacheEntryKey,
//     data_path, data_path_str,
//     dataformat::ZeroEvaluationAbs,
//     debug_print,
//     executor::{executor_main, Packet, ReturnMessage, TEMP_NETWORK_PREFIX},
//     fileformat::BinaryOutput,
//     mcts_trainer::{EvalMode, TypeRequest::TrainerSearch},
//     message_types::{DataFileType, Entity, MessageServer, MessageType, Statistics},
//     selfplay::{CollectorMessage, DataGen},
//     settings::{CPUCTSettings, FPUSettings, MovesLeftSettings, PSTSettings, SearchSettings},
//     utils::directory_exists,
// };

// fn net_checksum(data: &[u8]) -> String {
//     format!("{:x}", Sha256::digest(data))
// }

// fn net_file_checksum(file_path: &str) -> io::Result<String> {
//     let mut file = File::open(file_path)?;
//     let mut checksum = Sha256::new();
//     let mut buffer = [0u8; 64 * 1024];
//     loop {
//         let bytes_read = file.read(&mut buffer)?;
//         if bytes_read == 0 {
//             break;
//         }
//         checksum.update(&buffer[..bytes_read]);
//     }
//     Ok(format!("{:x}", checksum.finalize()))
// }

// fn main() {
//     let logical_cpus = num_cpus::get();
//     let physical_cpus = num_cpus::get_physical();
//     if logical_cpus > physical_cpus {
//         println!(
//             "We have simultaneous multithreading with about {:.2} \
//               logical cores to 1 physical core.",
//             (logical_cpus as f64) / (physical_cpus as f64)
//         );
//     } else if logical_cpus == physical_cpus {
//         println!(
//             "Either we don't have simultaneous multithreading, or our \
//               system doesn't support getting the number of physical CPUs."
//         );
//     } else {
//         println!(
//             "We have less logical CPUs than physical CPUs, maybe we only have access to \
//               some of the CPUs on our system."
//         );
//     }
//     println!("{physical_cpus}");
//     let pool = ThreadPool::builder()
//         .pool_size(physical_cpus)
//         // .pool_size(physical_cpus)
//         .create()
//         .unwrap();
//     env::set_var("RUST_BACKTRACE", "2");

//     panic::set_hook(Box::new(|panic_info| {
//         eprintln!("Panic occurred: {:?}", panic_info);
//         std::process::exit(1);
//     }));

//     let mut stream = loop {
//         match TcpStream::connect("127.0.0.1:38475") {
//             Ok(s) => break s,
//             Err(_) => continue,
//         }
//     };

//     let message = MessageServer {
//         purpose: MessageType::Initialise(Entity::RustDataGen),
//     };
//     let serialised = serde_json::to_string(&message).expect("serialisation failed");
//     let serialised = serialised + "\n";
//     stream
//         .write_all(serialised.as_bytes())
//         .expect("Failed to send data");
//     println!("Connected to server!");

//     let num_executors = 1;
//     // num_executors = max(min(tch::Cuda::device_count() as usize, num_executors), 1);
//     let batch_size = 2048;
//     let num_generators = num_executors * batch_size * 2;

//     let (game_sender, game_receiver) = flume::bounded::<CollectorMessage>(num_generators);
//     thread::scope(|s| {
//         let mut vec_communicate_exe_send: Vec<Sender<String>> = Vec::new();
//         let mut vec_communicate_exe_recv: Vec<Receiver<String>> = Vec::new();

//         for _ in 0..num_executors {
//             let (communicate_exe_send, communicate_exe_recv) =
//                 flume::bounded::<String>(num_generators);
//             vec_communicate_exe_send.push(communicate_exe_send);
//             vec_communicate_exe_recv.push(communicate_exe_recv);
//         }

//         let (id_send, id_recv) = flume::bounded::<usize>(1);
//         let (executor_ready_send, executor_ready_recv) = flume::unbounded::<bool>();

//         let _ = s
//             .builder()
//             .name("commander".to_string())
//             .spawn(|_| {
//                 commander_main(
//                     vec_communicate_exe_send,
//                     &mut stream.try_clone().expect("clone failed"),
//                     id_send,
//                     executor_ready_recv,
//                     num_executors,
//                 )
//             })
//             .unwrap();

//         // selfplay threads
//         let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Packet>(num_generators); // mcts to executor

//         for (exec_id, communicate_exe_recv) in vec_communicate_exe_recv.into_iter().enumerate() {
//             let eval_per_sec_sender = game_sender.clone();
//             let tensor_exe_recv_clone = tensor_exe_recv.clone();
//             let executor_ready = executor_ready_send.clone();
//             let _ = s
//                 .builder()
//                 .name(format!("executor-{}", exec_id).to_string())
//                 .spawn(move |_| {
//                     executor_main(
//                         communicate_exe_recv,
//                         tensor_exe_recv_clone,
//                         batch_size,
//                         Some(eval_per_sec_sender),
//                         Some(executor_ready),
//                         exec_id,
//                     )
//                 })
//                 .unwrap();
//         }

//         for n in 0..num_generators {
//             let sender_clone = game_sender.clone();
//             let selfplay_master = DataGen { iterations: 1 };
//             let tensor_exe_send_clone = tensor_exe_send.clone();
//             let fut_generator = async move {
//                 generator_main(sender_clone, selfplay_master, tensor_exe_send_clone, n).await;
//             };
//             pool.spawn_ok(fut_generator);
//         }

//         let _ = s
//             .builder()
//             .name("collector".to_string())
//             .spawn(|_| {
//                 collector_main(
//                     &game_receiver,
//                     &mut stream.try_clone().expect("clone failed"),
//                     id_recv,
//                 )
//             })
//             .unwrap();
//     })
//     .unwrap();
// }

// async fn generator_main(
//     sender_collector: Sender<CollectorMessage>,
//     datagen: DataGen,
//     tensor_exe_send: Sender<Packet>,
//     id: usize,
// ) {
//     let m_settings = MovesLeftSettings {
//         moves_left_weight: 0.05,
//         moves_left_clip: 20.0,
//         moves_left_sharpness: 0.5,
//     };

//     let settings: SearchSettings = SearchSettings {
//         fpu: FPUSettings {
//             root_fpu: 1.0,
//             children_fpu: 0.5,
//         },
//         wdl: EvalMode::Wdl,
//         moves_left: Some(m_settings),
//         c_puct: CPUCTSettings {
//             root_c_puct: 2.0,
//             children_c_puct: 2.0,
//         },
//         max_nodes: Some(1600),
//         alpha: 0.03,
//         eps: 0.25,
//         search_type: TrainerSearch(None),
//         pst: PSTSettings {
//             root_pst: 1.5,
//             children_pst: 1.5,
//         },
//         batch_size: 1,
//     };

//     let nps_sender = sender_collector.clone();

//     // implement caching

//     // let mut cache: LruCache<CacheEntryKey, ZeroEvaluationAbs> =
//     //     LruCache::new(NonZeroUsize::new(settings.max_nodes.unwrap() as usize).unwrap());
//     loop {
//         let input_data: Vec<f32> = (0..1344)
//             .map(|i| ((i * 17 + id) % 101) as f32 / 100.0 - 0.5)
//             .collect();

//         let input_tensor = Tensor::from_slice(&input_data);
//         let (resender_send, resender_recv) = flume::bounded::<ReturnMessage>(1);

//         let thread_name = std::thread::current()
//             .name()
//             .unwrap_or("unnamed-generator")
//             .to_owned();
//         let pack = Packet {
//             job: input_tensor,
//             resender: resender_send,
//             id: thread_name.clone(),
//         };
//         tensor_exe_send.send_async(pack).await.unwrap();
//         let _ = resender_recv.recv_async().await.unwrap();
//     }
// }

// fn serialise_file_to_bytes(file_path: &str) -> io::Result<Vec<u8>> {
//     let mut file = File::open(file_path)?;
//     let metadata = file.metadata()?;
//     let file_size = metadata.len() as usize;
//     let mut buffer = Vec::with_capacity(file_size);
//     file.read_to_end(&mut buffer)?;
//     Ok(buffer)
// }

// fn collector_main(
//     receiver: &Receiver<CollectorMessage>,
//     server_handle: &mut TcpStream,
//     id_recv: Receiver<usize>,
// ) {
//     let _thread_name = std::thread::current()
//         .name()
//         .unwrap_or("unnamed")
//         .to_owned();
//     debug_print!("Initialised {}", _thread_name);
//     let folder_name = data_path("games");
//     if let Err(e) = fs::create_dir(&folder_name) {
//         if e.kind() != std::io::ErrorKind::AlreadyExists {
//             println!("Error creating folder: {}", e);
//         }
//     } else {
//         println!("created {}", folder_name.display());
//     }
//     let id = id_recv.recv().unwrap();
//     let file_save_time = SystemTime::now();
//     let file_save_time_duration = file_save_time
//         .duration_since(UNIX_EPOCH)
//         .expect("Time went backwards");
//     let file_save_time_num = file_save_time_duration.as_nanos();
//     let mut path = data_path_str(&format!("games/gen_{}_games_{}", id, file_save_time_num));
//     let mut bin_output = BinaryOutput::new(path.clone(), "chess").unwrap();
//     let mut nps_start_time = Instant::now();
//     let mut nps_vec: Vec<usize> = Vec::new();
//     let mut evals_start_time = Instant::now();
//     let mut evals_vec: Vec<usize> = Vec::new();
//     let files = [".bin", ".off", ".json"];
//     loop {
//         let msg = receiver.recv().unwrap();
//         match msg {
//             CollectorMessage::FinishedGame(sim) => {
//                 bin_output.append(&sim).unwrap();
//                 if bin_output.game_count() >= 100 && bin_output.position_count() >= 25000 {
//                     bin_output.finish().unwrap();
//                     let mut file_data: Vec<Vec<u8>> = Vec::new(); // Clear file_data vector
//                     for file in files {
//                         let file_path = format!("{}{}", path, file);
//                         let data = serialise_file_to_bytes(&file_path).unwrap();
//                         file_data.push(data);
//                     }

//                     let (bin_file, off_file, metadata) = (
//                         file_data[0].clone(),
//                         file_data[1].clone(),
//                         file_data[2].clone(),
//                     );

//                     let message = MessageServer {
//                         purpose: MessageType::JobSendData(vec![
//                             DataFileType::BinFile(bin_file),
//                             DataFileType::OffFile(off_file),
//                             DataFileType::MetaDataFile(metadata),
//                         ]),
//                     };
//                     let mut serialised =
//                         serde_json::to_string(&message).expect("serialisation failed");
//                     serialised += "\n";
//                     server_handle.write_all(serialised.as_bytes()).unwrap();
//                     for file in files {
//                         let file_path = format!("{}{}", path, file);
//                         fs::remove_file(&file_path).expect("failed to delete sent game data file");
//                     }

//                     let file_save_time = SystemTime::now();
//                     let file_save_time_duration = file_save_time
//                         .duration_since(UNIX_EPOCH)
//                         .expect("Time went backwards");
//                     let file_save_time_num = file_save_time_duration.as_nanos();
//                     path = data_path_str(&format!("games/gen_{}_games_{}", id, file_save_time_num));
//                     bin_output = BinaryOutput::new(path.clone(), "chess").unwrap();
//                 }
//             }
//             CollectorMessage::GeneratorStatistics(nps) => {
//                 if nps_start_time.elapsed() >= Duration::from_secs(1) {
//                     let nps: usize = nps_vec.iter().sum();
//                     nps_start_time = Instant::now();
//                     nps_vec = Vec::new();
//                     let message = MessageServer {
//                         purpose: MessageType::StatisticsSend(Statistics::NodesPerSecond(nps)),
//                     };
//                     let mut serialised =
//                         serde_json::to_string(&message).expect("serialisation failed");
//                     serialised += "\n";
//                     server_handle.write_all(serialised.as_bytes()).unwrap();
//                 } else {
//                     nps_vec.push(nps);
//                 }
//             }
//             CollectorMessage::ExecutorStatistics(evals_per_sec) => {
//                 if evals_start_time.elapsed() >= Duration::from_secs(1) {
//                     let evals_per_second: usize = evals_vec.iter().sum();
//                     evals_start_time = Instant::now();
//                     evals_vec = Vec::new();
//                     let message = MessageServer {
//                         purpose: MessageType::StatisticsSend(Statistics::EvalsPerSecond(
//                             evals_per_second,
//                         )),
//                     };
//                     let mut serialised =
//                         serde_json::to_string(&message).expect("serialisation failed");
//                     serialised += "\n";
//                     server_handle.write_all(serialised.as_bytes()).unwrap();
//                 } else {
//                     evals_vec.push(evals_per_sec);
//                 }
//             }
//             CollectorMessage::GameResult(_) => {}
//             CollectorMessage::TestingResult(_) => {}
//             CollectorMessage::TestingControl(_) => {}
//         }
//     }
// }

// fn commander_main(
//     vec_exe_sender: Vec<Sender<String>>,
//     server_handle: &mut TcpStream,
//     id_sender: Sender<usize>,
//     executor_ready_receiver: Receiver<bool>,
//     num_executors: usize,
// ) {
//     let mut curr_net = String::new();
//     let mut is_initialised = false;
//     let mut net_path = String::new(); // initialize net_path with an empty string
//     let mut cloned_handle = server_handle.try_clone().unwrap();
//     let mut reader = BufReader::new(server_handle);
//     let mut net_path_counter = 0;
//     let mut generator_id: usize = 0;
//     let mut net_timestamp = SystemTime::now();
//     let net_save_time_duration = net_timestamp
//         .duration_since(UNIX_EPOCH)
//         .expect("Time went backwards");
//     let mut net_save_timestamp = net_save_time_duration.as_nanos();
//     let mut curr_net_checksum: Option<String> = None;
//     loop {
//         if !directory_exists(&data_path_str("nets")) {
//             fs::create_dir(data_path("nets")).unwrap();
//         }
//         let mut recv_msg = String::new();
//         if let Err(_) = reader.read_line(&mut recv_msg) {
//             return;
//         }
//         let message = match serde_json::from_str::<MessageServer>(&recv_msg) {
//             Ok(message) => message,
//             Err(_) => {
//                 recv_msg.clear();
//                 continue;
//             }
//         };

//         if is_initialised {
//             match message.purpose {
//                 MessageType::Initialise(_) => {}
//                 MessageType::JobSendPath(_) => {}
//                 MessageType::StatisticsSend(_) => {}
//                 MessageType::RequestingNet() => {}
//                 MessageType::NewNetworkPath(_) => {}
//                 MessageType::IdentityConfirmation(_) => {}
//                 MessageType::JobSendData(_) => {}
//                 MessageType::NewNetworkData(data) => {
//                     let checksum = net_checksum(&data);
//                     let loaded_checksum = curr_net_checksum.clone().or_else(|| {
//                         if curr_net.is_empty() {
//                             None
//                         } else {
//                             net_file_checksum(&curr_net).ok()
//                         }
//                     });
//                     if loaded_checksum.as_deref() == Some(checksum.as_str()) {
//                         println!("[Datagen] Ignoring duplicate network {}", checksum);
//                         curr_net_checksum = loaded_checksum;
//                         recv_msg.clear();
//                         continue;
//                     }
//                     println!("[Datagen] new net data {}", checksum);
//                     net_path = data_path_str(&format!(
//                         "nets/{}{}_{}_{}.pt",
//                         TEMP_NETWORK_PREFIX, generator_id, net_path_counter, net_save_timestamp
//                     ));
//                     let mut file = File::create(net_path.clone()).expect("Unable to create file");
//                     file.write_all(&data).expect("Unable to write data");
//                     net_timestamp = SystemTime::now();
//                     let net_save_time_duration = net_timestamp
//                         .duration_since(UNIX_EPOCH)
//                         .expect("Time went backwards");
//                     net_save_timestamp = net_save_time_duration.as_nanos();
//                     net_path_counter += 1;
//                     curr_net_checksum = Some(checksum);
//                 }
//                 MessageType::TBLink(_) => {}
//                 MessageType::CreateTB() => {}
//                 MessageType::TBLinkRequest() => {}
//                 MessageType::EvaluationRequest(_) => {}
//                 MessageType::TestResult(_) => {}
//             }
//         } else {
//             match message.purpose {
//                 MessageType::Initialise(_) => {}
//                 MessageType::JobSendPath(_) => {}
//                 MessageType::StatisticsSend(_) => {}
//                 MessageType::RequestingNet() => {}
//                 MessageType::NewNetworkPath(_) => {}
//                 MessageType::IdentityConfirmation((entity, id)) => match entity {
//                     Entity::RustDataGen => {
//                         generator_id = id;
//                         id_sender.send(id).unwrap();
//                         is_initialised = true;
//                     }
//                     _ => {
//                         println!("[Warning] Wrong entity, got {:?}", entity)
//                     }
//                 },
//                 MessageType::JobSendData(_) => {}
//                 MessageType::NewNetworkData(_) => {}
//                 MessageType::TBLink(_) => {}
//                 MessageType::CreateTB() => {}
//                 MessageType::TBLinkRequest() => {}
//                 MessageType::EvaluationRequest(_) => {}
//                 MessageType::TestResult(_) => {}
//             }
//         }

//         if curr_net != net_path && !net_path.is_empty() {
//             println!("updating net to: {}", net_path.clone());
//             let exists_file = Path::new(&curr_net).is_file();
//             if exists_file {
//                 match fs::remove_file(curr_net.clone()) {
//                     Ok(_) => {
//                         println!("Deleted net {}", curr_net);
//                     }
//                     Err(e) => eprintln!("Error deleting the file: {}", e),
//                 }
//             }
//             for exe_sender in &vec_exe_sender {
//                 exe_sender.send(net_path.clone()).unwrap();
//                 debug_print!("sent net!");
//             }
//             for _ in 0..num_executors {
//                 executor_ready_receiver
//                     .recv()
//                     .expect("executor disconnected before loading network");
//             }
//             if Path::new(&net_path)
//                 .file_name()
//                 .and_then(|name| name.to_str())
//                 .is_some_and(|name| name.starts_with(TEMP_NETWORK_PREFIX))
//             {
//                 fs::remove_file(&net_path).expect("failed to delete loaded temporary network");
//             }

//             curr_net = net_path.clone();
//         }
//         if net_path.is_empty() {
//             // actively request for net path

//             let message = MessageServer {
//                 purpose: MessageType::RequestingNet(),
//             };
//             let mut serialised = serde_json::to_string(&message).expect("serialisation failed");
//             serialised += "\n";
//             cloned_handle.write_all(serialised.as_bytes()).unwrap();
//         }
//         recv_msg.clear();
//     }
// }

// use std::time::Instant;
// use tch::{Device, IValue, Tensor};
// use tzrust::mcts_trainer::Net; // change module path if your Net lives elsewhere

// const BATCH_SIZE: i64 = 2048;
// const RUNS: usize = 100;

// fn main() -> anyhow::Result<()> {
//     let model_path = "/Users/andreas/Desktop/Code/RemoteFolder/TrueZero/nets/tz_0.pt";

//     let net = Net::new(model_path);

//     println!("device: {:?}", net.device);
//     assert_eq!(net.device, Device::Mps);

//     let input = Tensor::randn([BATCH_SIZE, 1344], (tch::Kind::Float, Device::Cpu));

//     // Warmup.
//     for _ in 0..10 {
//         let b = input.reshape([-1, 21, 8, 8]);
//         let b = b.to(net.device);

//         let output = net.net.forward_is(&[IValue::Tensor(b)])?;

//         let output_tensor = match output {
//             IValue::Tuple(v) => v,
//             _ => anyhow::bail!("unexpected output"),
//         };

//         let board_eval = match &output_tensor[0] {
//             IValue::Tensor(t) => t,
//             _ => anyhow::bail!("unexpected board output"),
//         };

//         let policy = match &output_tensor[1] {
//             IValue::Tensor(t) => t,
//             _ => anyhow::bail!("unexpected policy output"),
//         };

//         let _ = board_eval.to(Device::Cpu);
//         let _ = policy.to(Device::Cpu);
//     }

//     println!("warmup complete");

//     let mut times = Vec::with_capacity(RUNS);

//     for i in 0..RUNS {
//         let start = Instant::now();

//         let b = input.reshape([-1, 21, 8, 8]);
//         let b = b.to(net.device);

//         let output = net.net.forward_is(&[IValue::Tensor(b)])?;

//         let output_tensor = match output {
//             IValue::Tuple(v) => v,
//             _ => anyhow::bail!("unexpected output"),
//         };

//         let board_eval = match &output_tensor[0] {
//             IValue::Tensor(t) => t,
//             _ => anyhow::bail!("unexpected board output"),
//         };

//         let policy = match &output_tensor[1] {
//             IValue::Tensor(t) => t,
//             _ => anyhow::bail!("unexpected policy output"),
//         };

//         let _board_cpu = board_eval.to(Device::Cpu);
//         let _policy_cpu = policy.to(Device::Cpu);

//         let elapsed = start.elapsed();
//         times.push(elapsed);

//         println!("run {:>3}: {:.3} ms", i, elapsed.as_secs_f64() * 1000.0);
//     }

//     let first_n = RUNS / 10;

//     let first: f64 = times[..first_n]
//         .iter()
//         .map(|x| x.as_secs_f64())
//         .sum::<f64>()
//         / first_n as f64;

//     let last: f64 = times[RUNS - first_n..]
//         .iter()
//         .map(|x| x.as_secs_f64())
//         .sum::<f64>()
//         / first_n as f64;

//     println!();
//     println!("first 10%: {:.3} ms", first * 1000.0);
//     println!("last  10%: {:.3} ms", last * 1000.0);
//     println!("ratio:     {:.3}x", last / first);

//     Ok(())
// }


fn main() {
    let n = 2048usize * 1880;

    let mut v = vec![1.0f32; n];

    let start = std::time::Instant::now();

    v.fill(0.0);

    println!(
        "fill_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
}