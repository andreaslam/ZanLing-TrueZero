use crate::{
    boardmanager::BoardStack,
    decoder::convert_board,
    executor::{Packet, ReturnMessage},
    utils::debug_print,
};
use cozy_chess::Board;
use flume::Sender;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub fn send_request(sender: Sender<Packet>, id: usize) {
    let board = Board::default();
    let bs = BoardStack::new(board);
    let input_tensor = convert_board(&bs);
    let (resender_send, recv) = flume::bounded::<ReturnMessage>(1); // mcts to executor

    loop {
        let pack = Packet {
            job: input_tensor.clone(&input_tensor),
            resender: resender_send.clone(),
            id: "dummy-req".to_string(),
        };
        // debug_print(&format!("HIIIIII");
        sender.send(pack).unwrap();
        let now_start_proc = SystemTime::now();
        let since_epoch_proc = now_start_proc
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");

        let epoch_seconds_start_proc = since_epoch_proc.as_nanos();
        recv.recv().unwrap();
        let now_end_proc = SystemTime::now();
        let since_epoch_proc = now_end_proc
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let epoch_seconds_end_proc = since_epoch_proc.as_nanos();
        if id % 512 == 0 {
            debug_print(&format!(
                "{} {} {} waiting_response_dummy",
                epoch_seconds_start_proc, epoch_seconds_end_proc, id,
            ));
        }
    }
}
pub async fn send_request_async(sender: Sender<Packet>, id: usize) {
    let board = Board::default();

    let bs = BoardStack::new(board);

    let input_tensor = convert_board(&bs);

    let (resender_send, recv) = flume::bounded::<ReturnMessage>(1); // mcts to executor
    let mut rng = StdRng::from_entropy();
    loop {
        let pack = Packet {
            job: input_tensor.clone(&input_tensor),
            resender: resender_send.clone(),
            id: "dummy-req".to_string(),
        };
        sender.send_async(pack).await.unwrap();

        let now_start_proc = SystemTime::now();
        let since_epoch_proc = now_start_proc
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");

        let epoch_seconds_start_proc = since_epoch_proc.as_nanos();
        recv.recv_async().await.unwrap();
        let now_end_proc = SystemTime::now();
        let since_epoch_proc = now_end_proc
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let epoch_seconds_end_proc = since_epoch_proc.as_nanos();
        // if id % 512 == 0 {
        //     debug_print(
        //         "{} {} {} waiting_response_dummy",
        //         epoch_seconds_start_proc, epoch_seconds_end_proc, id
        //     );
        // }

        // Generate a random number between 0 and 5 nanos
        // let nanos = rng.gen_range(0..=5);

        // Sleep for the randomly generated number of nanos
        // sleep(Duration::new(0, nanos)); // delay here
    }
}
