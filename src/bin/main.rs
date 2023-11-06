use crossbeam::thread;
use flume::{Receiver, Sender};
use std::{env, thread::sleep, time::Duration};
use tch::Tensor;
use tz_rust::{
    dataformat::Simulation,
    executor::{executor_main, Message},
    fileformat::BinaryOutput,
    selfplay::DataGen,
};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let (game_sender, game_receiver) = flume::bounded::<Simulation>(1);
    let num_threads = 512;

    thread::scope(|s| {
        let mut selfplay_masters: Vec<DataGen> = Vec::new();

        // commander
        let (communicate_exe_send, communicate_exe_recv) = flume::bounded::<String>(1);

        s.builder()
            .name("commander".to_string())
            .spawn(|_| commander_main(communicate_exe_send))
            .unwrap();

        // s.spawn(move |_| {
        //     commander_main(communicate_exe_send);
        // });
        // selfplay threads
        let mut exe_resenders: Vec<Sender<Message>> = Vec::new(); // sent FROM executor to mcts
        let mut tensor_senders: Vec<Receiver<Tensor>> = Vec::new(); // sent FROM mcts to executor
        for n in 0..num_threads {
            // sender-receiver pair to communicate for each thread instance to the executor

            let (eval_exe_send, eval_exe_recv) = flume::bounded::<Message>(1); // executor to mcts
            let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Tensor>(1); // mcts to executor
            let sender_clone = game_sender.clone();
            let mut selfplay_master = DataGen { iterations: 1 };
            s.builder()
                .name(format!("generator_{}", n.to_string()))
                .spawn(move |_| {
                    generator_main(
                        &sender_clone,
                        &mut selfplay_master,
                        tensor_exe_send,
                        eval_exe_recv,
                    )
                })
                .unwrap();
            exe_resenders.push(eval_exe_send);
            tensor_senders.push(tensor_exe_recv);
            // s.spawn(move |_| {
            //     generator_main(&sender_clone, &mut selfplay_master);
            // });

            selfplay_masters.push(selfplay_master.clone());
        }
        // collector

        s.builder()
            .name("collector".to_string())
            .spawn(|_| collector_main(&game_receiver))
            .unwrap();

        // s.spawn(move |_| {
        //     collector_main(&game_receiver);
        // });
        // executor

        s.builder()
            .name("executor".to_string())
            .spawn(|_| {
                executor_main(
                    communicate_exe_recv,
                    tensor_senders,
                    num_threads,
                    exe_resenders,
                )
            })
            .unwrap();

        // s.spawn(move |_| {
        //     executor_main(communicate_exe_recv);
        // });
    })
    .unwrap();
}

fn generator_main(
    sender_collector: &Sender<Simulation>,
    datagen: &mut DataGen,
    tensor_exe_send: Sender<Tensor>,
    eval_exe_recv: Receiver<Message>,
) {
    loop {
        let sim = datagen.play_game(tensor_exe_send.clone(), eval_exe_recv.clone());
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
        if bin_output.game_count() >= 5 {
            counter += 1;
            let _ = bin_output.finish().unwrap();
            bin_output = BinaryOutput::new(format!("games_{}", counter), "chess").unwrap();
        }
    }
}
fn commander_main(exe_sender: Sender<String>) {
    loop {
        exe_sender
            .send("chess_16x128_gen3634.pt".to_string())
            .unwrap();
        // println!("SENT!");
        sleep(Duration::from_secs(10));
    }
}
