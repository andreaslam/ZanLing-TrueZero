use crate::{
    boardmanager::BoardStack,
    dataformat::{Position, Simulation},
    executor::{Packet, ReturnMessage},
    mcts_trainer::{get_move, ExpansionType, TypeRequest},
    settings::SearchSettings,
};
use cozy_chess::{Board, Color, GameStatus, Move};
use flume::Sender;
use rand::prelude::*;
use rand_distr::WeightedIndex;
use std::time::Instant;
// selfplay code
#[derive(Clone, Debug)]
pub enum CollectorMessage {
    FinishedGame(Simulation),
    GeneratorStatistics(f32),
    ExecutorStatistics(f32),
    GameResult(Option<Color>),
    TestingResult(Option<bool>), // engine_0 win = true, engine_0 loss = false
}

#[derive(PartialEq, Clone, Debug, Copy)]

pub struct DataGen {
    pub iterations: u32, // number of games needed per batch of training data
}

impl DataGen {
    pub async fn play_game(
        &self,
        tensor_exe_send: &Sender<Packet>,
        nps_sender: &Sender<CollectorMessage>,
        settings: &SearchSettings,
        id: usize,
    ) -> Simulation {
        let sw = Instant::now();
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
                get_move(bs.clone(), &tensor_exe_send, settings.clone(), id).await;
            let elapsed = sw.elapsed().as_nanos() as f32 / 1e9;
            let final_mv = if positions.len() > 30 {
                // when tau is "infinitesimally small", pick the best move
                mv
            } else {
                let weighted_index = WeightedIndex::new(&search_data.policy).unwrap();
                let mut rng = rand::thread_rng();
                // println!("{}, {:?}, {:?}", thread_name, weighted_index, rng);
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
            let nps = settings.max_nodes as f32 / elapsed;

            // println!("thread {}, {:#}, {}nps", thread_name, final_mv, nps);
            // println!("{:#}", final_mv);
            nps_sender
                .send_async(CollectorMessage::GeneratorStatistics(
                    settings.max_nodes as f32,
                ))
                .await
                .unwrap();
            bs.play(final_mv);
            positions.push(pos);
        }

        let tz = Simulation {
            positions,
            final_board: bs,
        };
        let elapsed_ms = sw.elapsed().as_nanos() as f32 / 1e9;
        // println!("one done {}s", elapsed_ms);
        // println!("one done!");
        tz
    }
}

// pub async fn synthetic_expansion(
//     simulation: Simulation,
//     position: usize,
//     sender: Sender<Packet>,
//     settings: SearchSettings,
// ) -> Simulation {
//     let mut synthetic_game_vec: Vec<Position> = simulation.positions[0..position].to_vec();
//     let mut move_list = Vec::new();
//     simulation.positions[position]
//         .board
//         .board()
//         .generate_moves(|moves| {
//             // Unpack dense move set into move list
//             move_list.extend(moves);
//             false
//         });
//     let mut expansion_bs = BoardStack::new(simulation.positions[position].board.board().clone());
//     match settings.search_type {
//         TypeRequest::TrainerSearch(expansion_type) => {
//             match expansion_type {
//                 Some(ExpansionType::PolicyExpansion) => {
//                     while expansion_bs.status() == GameStatus::Ongoing {
//                         let (mv, v_p, move_idx_piece, search_data, visits) =
//                             get_move(expansion_bs.clone(), sender.clone(), settings.clone()).await;
//                         let synthetic_pos = Position {
//                             board: expansion_bs.clone(),
//                             is_full_search: true,
//                             played_mv: mv,
//                             zero_visits: visits as u64,
//                             zero_evaluation: search_data, // q
//                             net_evaluation: v_p,          // v
//                         };
//                         expansion_bs.play(mv);
//                         synthetic_game_vec.push(synthetic_pos);
//                     }
//                 }
//                 Some(ExpansionType::RandomExpansion) => {
//                     while expansion_bs.status() == GameStatus::Ongoing {
//                         let mut rng = rand::thread_rng();
//                         let mut legal_moves: Vec<Move> = Vec::new();
//                         expansion_bs.board().generate_moves(|moves| {
//                             // Unpack dense move set into move list
//                             legal_moves.extend(moves);
//                             false
//                         });
//                         legal_moves.shuffle(&mut rng);
//                         let (_, v_p, _, search_data, _) =
//                             get_move(expansion_bs.clone(), sender.clone(), settings.clone()).await;
//                         let mv = legal_moves.first().unwrap();
//                         expansion_bs.play(*mv);
//                         let synthetic_pos = Position {
//                             board: expansion_bs.clone(),
//                             is_full_search: true,
//                             played_mv: *mv,
//                             zero_visits: 0,
//                             zero_evaluation: search_data, // q
//                             net_evaluation: v_p,          // v
//                         };
//                         synthetic_game_vec.push(synthetic_pos);
//                     }
//                 }
//                 None => panic!("unexpected type of expansion found!"),
//             }
//         }
//         _ => panic!("not expecting this type of search"),
//     }

//     let tz_synthetic = Simulation {
//         positions: synthetic_game_vec,
//         final_board: expansion_bs,
//     };
//     tz_synthetic
// }
