use crate::{
    boardmanager::BoardStack,
    dataformat::{Position, Simulation},
    executor::{Message, Packet},
    mcts_trainer::{get_move, MAX_NODES},
};
use cozy_chess::{Board, GameStatus, Move};
use flume::{Receiver, Sender};
use rand::prelude::*;
use rand_distr::WeightedIndex;
use std::time::Instant;
// selfplay code

pub enum CollectorMessage {
    FinishedGame(Simulation),
    GeneratorStatistics(f32),
    ExecutorStatistics(f32),
}

#[derive(PartialEq, Clone, Debug, Copy)]

pub struct DataGen {
    pub iterations: u32, // number of games needed per batch of training data
}

impl DataGen {
    pub fn play_game(
        &self,
        tensor_exe_send: Sender<Packet>,
        nps_sender: Sender<CollectorMessage>,
    ) -> Simulation {
        let mut bs = BoardStack::new(Board::default());
        // let mut value: Vec<f32> = Vec::new();
        let mut positions: Vec<Position> = Vec::new();
        let thread_name = std::thread::current()
            .name()
            .unwrap_or("unnamed")
            .to_owned();
        while bs.status() == GameStatus::Ongoing {
            let sw = Instant::now();
            let (mv, v_p, move_idx_piece, search_data, visits) =
                get_move(bs.clone(), tensor_exe_send.clone());
            let elapsed = sw.elapsed().as_nanos() as f32 / 1e9;
            let final_mv = if positions.len() > 30 {
                // when tau is "infinitesimally small", pick the best move
                mv
            } else {
                let weighted_index = WeightedIndex::new(&search_data.policy).unwrap();

                let mut rng = rand::thread_rng();
                let sampled_idx = weighted_index.sample(&mut rng);
                let mut legal_moves: Vec<Move> = Vec::new();
                bs.board().generate_moves(|moves| {
                    // Unpack dense move set into move list
                    legal_moves.extend(moves);
                    false
                });
                legal_moves[sampled_idx]
            };

            let pos = Position {
                board: bs.clone(),
                is_full_search: true,
                played_mv: final_mv,
                zero_visits: visits as u64,
                zero_evaluation: search_data, // q
                net_evaluation: v_p,          // v
            };
            let nps = MAX_NODES as f32 / elapsed;
            // if thread_name == "generator_1" {
            //     println!("{:#}", final_mv,);
            // }
            // println!("thread {}, {:#}, {}nps", thread_name, final_mv, nps);
            println!("{:#}", final_mv);
            let _ = nps_sender.send(CollectorMessage::GeneratorStatistics(MAX_NODES as f32));
            bs.play(final_mv);
            positions.push(pos);
        }
        // let outcome: Option<Color> = match bs.status() {
        //     GameStatus::Drawn => None,
        //     GameStatus::Won => Some(!bs.board().side_to_move()),
        //     GameStatus::Ongoing => panic!("Game is still ongoing!"),
        // };
        let tz = Simulation {
            positions,
            final_board: bs,
        };
        // if thread_name == "generator_1" {
        //     println!("one done!");
        // }
        println!("one done!");
        tz
    }
}
