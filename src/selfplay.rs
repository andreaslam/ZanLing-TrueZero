use crate::dataformat::{Position, ZeroEvaluation};
use crate::mcts_trainer::get_move;
use crate::{boardmanager::BoardStack, dataformat::Simulation};
use cozy_chess::{Board, Color, GameStatus, Move};
use rand::prelude::*;
use rand_distr::WeightedIndex;
use rayon::prelude::*;
use std::{
    sync::{Arc, Mutex},
    thread,
};
// selfplay code
#[derive(PartialEq, Clone, Debug)]

pub struct DataGen {
    pub iterations: u32, // number of games needed per batch of training data
}

impl DataGen {
    pub fn play_game(&self) -> Simulation {
        let mut bs = BoardStack::new(Board::default());
        // let mut value: Vec<f32> = Vec::new();
        let mut positions: Vec<Position> = Vec::new();
        while bs.status() == GameStatus::Ongoing {
            let (mv, v_p, move_idx_piece, search_data, visits) = get_move(bs.clone());
            let final_mv = if positions.len() > 30 {
                // when tau is "infinitesimally small", pick the best move
                println!("{:#}", mv);
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
                println!("{:#}", legal_moves[sampled_idx]);
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

        tz
    }

    pub fn generate_batch(&mut self) -> Vec<Simulation> {
        // get number of cpu threads

        let threads_available = rayon::current_num_threads();
        // let threads_available = 3;

        println!("{}", threads_available);

        let tz_data: Arc<Mutex<Vec<Simulation>>> = Arc::new(Mutex::new(Vec::new()));
        while tz_data.lock().unwrap().len() < self.iterations.try_into().unwrap() {
            // easiest to compare
            let mut tz_data = tz_data.lock().unwrap();
            let batch: Vec<Simulation> = (0..threads_available)
                .into_par_iter()
                .map(|_| self.play_game())
                .collect();

            tz_data.extend(batch);
        }
        let tz_data = Arc::try_unwrap(tz_data).unwrap().into_inner().unwrap();
        tz_data
    }
}
