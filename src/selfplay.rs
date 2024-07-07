use crate::{
    boardmanager::BoardStack, cache::{CacheEntryKey, CacheEntryValue}, dataformat::{Position, Simulation}, debug_print, executor::{Packet, ReturnMessage}, mcts_trainer::{get_move, ExpansionType, TypeRequest}, settings::SearchSettings
};
use cozy_chess::{Board, Color, GameStatus, Move};
use flume::Sender;
use lru::LruCache;
use rand::prelude::*;
use rand_distr::WeightedIndex;
use std::time::Instant;
// selfplay code
#[derive(Clone, Debug)]
pub enum CollectorMessage {
    FinishedGame(Simulation),
    GeneratorStatistics(usize),
    ExecutorStatistics(usize),
    GameResult(Option<Color>),
    TestingResult(Option<bool>), // engine_0 win = true, engine_0 loss = false
}

#[derive(PartialEq, Clone, Debug, Copy)]

pub struct DataGen {
    pub iterations: u32, // number of games needed per batch of training data
}

impl DataGen {

    /// plays a game of chess given a sender for an `executor.rs` inference backend (tensor_exe_send), where said executor should be spawned as a seperate thread
    /// returns a `Simulation` containing key game metadata. for reference see `Simulation` and `Position` 
    pub async fn play_game(
        &self,
        tensor_exe_send: &Sender<Packet>,
        nps_sender: &Sender<CollectorMessage>,
        settings: &SearchSettings,
        id: usize,
        mut cache: &mut LruCache<CacheEntryKey, CacheEntryValue>,
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
            let (mv, v_p, move_idx_piece, search_data, visits) = get_move(
                bs.clone(),
                &tensor_exe_send,
                settings.clone(),
                id,
                &mut cache,
            )
            .await;
            let elapsed = sw.elapsed().as_nanos() as f32 / 1e9;
            let final_mv = if positions.len() > 30 {
                // when tau is "infinitesimally small", pick the best move
                mv
            } else {
                let weighted_index = WeightedIndex::new(&search_data.policy).unwrap();
                let mut rng = rand::thread_rng();
                // // debug_print(&format!("{}, {:?}, {:?}", thread_name, weighted_index, rng);
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

            debug_print!("{}", &format!("thread {}, {:#}, {}nps", thread_name, final_mv, nps));
            debug_print!("{}", &format!("{:#}", final_mv));
            nps_sender
                .send_async(CollectorMessage::GeneratorStatistics(
                    settings.max_nodes as usize,
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
        debug_print!("{}",&format!("one done {}s", elapsed_ms));
        debug_print!("{}",&format!("one done!"));
        tz
    }
}
