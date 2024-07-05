use crate::{
    boardmanager::BoardStack,
    decoder::convert_board,
    executor::{Packet, ReturnMessage},
};
use cozy_chess::Board;
use flume::Sender;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{SystemTime, UNIX_EPOCH};

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
        sender.send(pack).unwrap();
    }
}
pub async fn send_request_async(sender: Sender<Packet>, id: usize) {
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
        sender.send_async(pack).await.unwrap();
        recv.recv_async().await.unwrap();
    }
}
