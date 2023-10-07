use std::{
    env,
    sync::mpsc::{self, Receiver, Sender},
    thread,
};
use tz_rust::{dataformat::Simulation, fileformat::BinaryOutput, selfplay::DataGen};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let (sender1, receiver1) = mpsc::channel();
    let num_threads = 4;

    let mut selfplay_masters: Vec<DataGen> = Vec::new();
    let mut selfplay_threads: Vec<thread::JoinHandle<()>> = Vec::new();

    for _ in 0..num_threads {
        let sender_clone = sender1.clone();
        let mut selfplay_master = DataGen { iterations: 1 };

        let selfplay_thread = thread::spawn(move || {
            generator_main(&sender_clone, &mut selfplay_master);
        });

        selfplay_masters.push(selfplay_master.clone());
        selfplay_threads.push(selfplay_thread);
    }

    // collector
    let col_thread = thread::spawn(move || {
        collector_main(&receiver1);
    });

    // join all selfplay threads
    for handle in selfplay_threads {
        handle.join().unwrap();
    }

    // wait for the collector to finish
    col_thread.join().unwrap();
}

fn generator_main(sender_collector: &Sender<Simulation>, datagen: &mut DataGen) {
    loop {
        let sim = datagen.play_game();
        sender_collector.send(sim).unwrap();
    }
}

fn collector_main(receiver: &Receiver<Simulation>) {
    let mut counter = 0;
    let mut bin_output = BinaryOutput::new(format!("games_{}", counter), "chess").unwrap();

    loop {
        let sim = receiver.recv().unwrap();
        let _ = bin_output.append(&sim).unwrap();
        println!("{}", bin_output.game_count());
        if bin_output.game_count() >= 200 {
            counter += 1;
            let _ = bin_output.finish().unwrap();
            bin_output = BinaryOutput::new(format!("games_{}", counter), "chess").unwrap();
            println!("FINALLY");
        }
    }
}
