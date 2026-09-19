use crossbeam::thread;
use flume::{Receiver, Sender};
use sha2::{Digest, Sha256};
use std::{
    env,
    fs::{self, File},
    io::{self, BufRead, BufReader, Read, Write},
    net::TcpStream,
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread as std_thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tch::{Device, Kind, Tensor};

use tzrust::{
    data_path, data_path_str,
    executor::{executor_main, Packet, TEMP_NETWORK_PREFIX},
    message_types::{Entity, MessageServer, MessageType},
    utils::directory_exists,
};

const NUM_EXECUTORS: usize = 2;
const MAX_BATCH_SIZE: usize = 2048;
const WORKER_THREADS: usize = 60;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    println!("==============================================");
    println!(" IN-FLIGHT REQUEST BENCHMARK");
    println!("==============================================");
    println!("Worker threads: {}", WORKER_THREADS);
    println!("Executors:      {}", NUM_EXECUTORS);
    println!("Max batch:      {}", MAX_BATCH_SIZE);
    println!("Tensor device:  CPU");
    println!("==============================================");

    let mut stream = loop {
        match TcpStream::connect("127.0.0.1:38475") {
            Ok(s) => break s,
            Err(_) => std_thread::sleep(Duration::from_millis(100)),
        }
    };

    let message = MessageServer {
        purpose: MessageType::Initialise(Entity::RustDataGen),
    };

    let mut serialised = serde_json::to_string(&message).expect("serialisation failed");

    serialised.push('\n');

    stream
        .write_all(serialised.as_bytes())
        .expect("Failed to send data");

    println!("Connected to server!");

    let (id_send, _id_recv) = flume::bounded::<usize>(1);

    let (executor_ready_send, executor_ready_recv) = flume::unbounded::<bool>();

    thread::scope(|s| {
        //
        // COMMANDER CHANNELS
        //
        let mut vec_communicate_exe_send = Vec::new();
        let mut vec_communicate_exe_recv = Vec::new();

        for _ in 0..NUM_EXECUTORS {
            let (send, recv) = flume::bounded::<String>(4096);

            vec_communicate_exe_send.push(send);
            vec_communicate_exe_recv.push(recv);
        }

        //
        // COMMANDER
        //
        let mut commander_stream = stream.try_clone().expect("failed to clone stream");

        s.builder()
            .name("commander".to_string())
            .spawn(move |_| {
                commander_main(
                    vec_communicate_exe_send,
                    &mut commander_stream,
                    id_send,
                    executor_ready_recv,
                    NUM_EXECUTORS,
                );
            })
            .unwrap();

        //
        // EXECUTORS
        //
        let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Packet>(8192);

        for (executor_id, communicate_exe_recv) in vec_communicate_exe_recv.into_iter().enumerate()
        {
            let tensor_recv = tensor_exe_recv.clone();
            let ready_sender = executor_ready_send.clone();

            s.builder()
                .name(format!("executor-{}", executor_id))
                .spawn(move |_| {
                    executor_main(
                        communicate_exe_recv,
                        tensor_recv,
                        MAX_BATCH_SIZE,
                        None,
                        Some(ready_sender),
                        executor_id,
                    );
                })
                .unwrap();
        }

        //
        // WAIT FOR NETWORK
        //
        println!("Waiting for executors/network...");
        std_thread::sleep(Duration::from_secs(3));

        //
        // EXPERIMENT
        //
        //
        // We deliberately keep the number of generators fixed.
        //
        // Changing the number of outstanding requests per generator
        // tells us whether the inference engine is being starved by
        // the request -> wait -> request architecture.
        //
        const NUM_GENERATORS: usize = 4096;

        let in_flight_counts = [1usize, 2, 4, 8, 16];

        println!();
        println!("==============================================");
        println!(" RESULTS");
        println!("==============================================");
        println!("{:>15} {:>15}", "In-flight/gen", "Eval/s");
        println!("----------------------------------------------");

        for &in_flight in &in_flight_counts {
            let result = run_test(
                &tensor_exe_send,
                NUM_GENERATORS,
                in_flight,
                Duration::from_secs(5),
            );

            println!("{:>15} {:>15.0}", in_flight, result.evals_per_second);

            println!();
        }

        println!("==============================================");
        println!(" EXPERIMENT COMPLETE");
        println!("==============================================");

        //
        // IMPORTANT:
        //
        // The executor receiver will disconnect when this scope exits.
        // executor_main currently unwraps that disconnect, so the
        // executor panic at shutdown is expected with the current
        // production executor implementation.
        //
    })
    .unwrap();
}

struct TestResult {
    evals_per_second: f64,
}

fn run_test(
    tensor_exe_send: &Sender<Packet>,
    num_generators: usize,
    in_flight: usize,
    duration: Duration,
) -> TestResult {
    println!(
        "Testing {} generators × {} in-flight for {:.1}s...",
        num_generators,
        in_flight,
        duration.as_secs_f64()
    );

    //
    // Completed evaluations.
    //
    let evals = Arc::new(AtomicUsize::new(0));

    //
    // Each generator gets its own OS thread.
    //
    let mut handles = Vec::with_capacity(num_generators);

    let start = Instant::now();

    for id in 0..num_generators {
        let sender = tensor_exe_send.clone();
        let evals = evals.clone();

        let handle = std_thread::Builder::new()
            .name(format!("generator-{}", id))
            .spawn(move || {
                //
                // Match production convert_board():
                //
                // [1, 1344] CPU tensor.
                //
                let input = Tensor::zeros([1, 1344], (Kind::Float, Device::Cpu));

                //
                // Maintain exactly `in_flight` requests.
                //
                //
                // For in_flight = 1:
                //
                // request
                //   ↓
                // wait
                //   ↓
                // request
                //
                //
                // For in_flight = 8:
                //
                // request ─┐
                // request ─┤
                // request ─┤
                // request ─┤
                // request ─┤
                // request ─┤
                // request ─┤
                // request ─┘
                //     ↓
                // collect results
                //
                let mut receivers = Vec::with_capacity(in_flight);

                //
                // Initial pipeline fill.
                //
                for _ in 0..in_flight {
                    let (tx, rx) = flume::bounded(1);

                    sender
                        .send(Packet {
                            job: input.shallow_clone(),
                            resender: tx,
                            id: id.to_string(),
                        })
                        .unwrap();

                    receivers.push(rx);
                }

                //
                // Continuously maintain the pipeline.
                //
                while start.elapsed() < duration {
                    //
                    // Wait for one completed evaluation.
                    //
                    let rx = receivers.remove(0);

                    match rx.recv() {
                        Ok(_) => {
                            evals.fetch_add(1, Ordering::Relaxed);
                        }

                        Err(_) => {
                            return;
                        }
                    }

                    //
                    // Immediately replace it with another request.
                    //
                    let (tx, rx) = flume::bounded(1);

                    sender
                        .send(Packet {
                            job: input.shallow_clone(),
                            resender: tx,
                            id: id.to_string(),
                        })
                        .unwrap();

                    receivers.push(rx);
                }

                //
                // Drain everything already submitted.
                //
                for rx in receivers {
                    if rx.recv().is_ok() {
                        evals.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
            .expect("failed to spawn generator thread");

        handles.push(handle);
    }

    //
    // Wait for the measurement window.
    //
    std_thread::sleep(duration);

    //
    // Join all generator threads.
    //
    for handle in handles {
        handle.join().expect("generator thread panicked");
    }

    let elapsed = start.elapsed().as_secs_f64();

    let completed = evals.load(Ordering::Relaxed);

    TestResult {
        evals_per_second: completed as f64 / elapsed,
    }
}

fn commander_main(
    vec_exe_sender: Vec<Sender<String>>,
    server_handle: &mut TcpStream,
    id_sender: Sender<usize>,
    executor_ready_receiver: Receiver<bool>,
    num_executors: usize,
) {
    let mut curr_net = String::new();
    let mut is_initialised = false;
    let mut net_path = String::new();

    let mut cloned_handle = server_handle.try_clone().unwrap();
    let mut reader = BufReader::new(server_handle);

    let mut net_path_counter = 0;
    let mut generator_id: usize = 0;

    let mut net_timestamp = SystemTime::now();

    let net_save_time_duration = net_timestamp
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");

    let mut net_save_timestamp = net_save_time_duration.as_nanos();

    let mut curr_net_checksum: Option<String> = None;

    loop {
        if !directory_exists(&data_path_str("nets")) {
            fs::create_dir(data_path("nets")).unwrap();
        }

        let mut recv_msg = String::new();

        if reader.read_line(&mut recv_msg).is_err() {
            return;
        }

        let message = match serde_json::from_str::<MessageServer>(&recv_msg) {
            Ok(message) => message,
            Err(_) => continue,
        };

        if is_initialised {
            match message.purpose {
                MessageType::NewNetworkData(data) => {
                    let checksum = net_checksum(&data);

                    let loaded_checksum = curr_net_checksum.clone().or_else(|| {
                        if curr_net.is_empty() {
                            None
                        } else {
                            net_file_checksum(&curr_net).ok()
                        }
                    });

                    if loaded_checksum.as_deref() == Some(checksum.as_str()) {
                        println!("[Datagen] Ignoring duplicate network {}", checksum);

                        curr_net_checksum = loaded_checksum;
                        continue;
                    }

                    println!("[Datagen] new net data {}", checksum);

                    net_path = data_path_str(&format!(
                        "nets/{}{}_{}_{}.pt",
                        TEMP_NETWORK_PREFIX, generator_id, net_path_counter, net_save_timestamp
                    ));

                    let mut file =
                        File::create(net_path.clone()).expect("Unable to create data file");

                    file.write_all(&data).expect("Unable to write data");

                    net_timestamp = SystemTime::now();

                    let net_save_time_duration = net_timestamp
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards");

                    net_save_timestamp = net_save_time_duration.as_nanos();

                    net_path_counter += 1;
                    curr_net_checksum = Some(checksum);
                }

                _ => {}
            }
        } else {
            match message.purpose {
                MessageType::IdentityConfirmation((entity, id)) => match entity {
                    Entity::RustDataGen => {
                        generator_id = id;

                        id_sender.send(id).unwrap();

                        is_initialised = true;
                    }

                    _ => {
                        println!("[Warning] Wrong entity, got {:?}", entity);
                    }
                },

                _ => {}
            }
        }

        //
        // Load new network into every executor.
        //
        if curr_net != net_path && !net_path.is_empty() {
            println!("updating net to: {}", net_path);

            if Path::new(&curr_net).is_file() {
                match fs::remove_file(curr_net.clone()) {
                    Ok(_) => println!("Deleted net {}", curr_net),

                    Err(e) => {
                        eprintln!("Error deleting the file: {}", e);
                    }
                }
            }

            for exe_sender in &vec_exe_sender {
                exe_sender.send(net_path.clone()).unwrap();
            }

            for _ in 0..num_executors {
                executor_ready_receiver
                    .recv()
                    .expect("executor disconnected before loading network");
            }

            if Path::new(&net_path)
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(TEMP_NETWORK_PREFIX))
            {
                fs::remove_file(&net_path).expect("failed to delete loaded temporary network");
            }

            curr_net = net_path.clone();
        }

        //
        // Request a network if we don't have one.
        //
        if net_path.is_empty() {
            let message = MessageServer {
                purpose: MessageType::RequestingNet(),
            };

            let mut serialised = serde_json::to_string(&message).expect("serialisation failed");

            serialised.push('\n');

            cloned_handle.write_all(serialised.as_bytes()).unwrap();
        }
    }
}

fn net_checksum(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn net_file_checksum(file_path: &str) -> io::Result<String> {
    let mut file = File::open(file_path)?;

    let mut checksum = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];

    loop {
        let bytes_read = file.read(&mut buffer)?;

        if bytes_read == 0 {
            break;
        }

        checksum.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:x}", checksum.finalize()))
}
