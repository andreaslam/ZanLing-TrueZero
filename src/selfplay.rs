use crate::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    dataformat::{Position, Simulation, ZeroEvaluationAbs, ZeroEvaluationPov},
    debug_print,
    executor::Packet,
    mcts_trainer::get_move,
    settings::SearchSettings,
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
        cache: &mut LruCache<CacheEntryKey, ZeroEvaluationAbs>,
    ) -> Simulation {
        let _sw = Instant::now();
        let mut bs = BoardStack::new(Board::default());
        // let mut value: Vec<f32> = Vec::new();
        let mut positions: Vec<Position> = Vec::new();
        let _thread_name = std::thread::current()
            .name()
            .unwrap_or("unnamed")
            .to_owned();
        while bs.status() == GameStatus::Ongoing {
            let _sw = Instant::now();
            let (mv, v_p, _move_idx_piece, search_data, visits) =
                get_move(bs.clone(), tensor_exe_send, *settings, id, cache).await;
            let _elapsed = _sw.elapsed().as_nanos() as f32 / 1e9;
            let final_mv = if positions.len() > 30 {
                // when tau is "infinitesimally small", pick the best move
                mv
            } else {
                let weighted_index = WeightedIndex::new(&search_data.policy).unwrap();
                let mut rng = rand::thread_rng();
                // // debug_print(&format!("{}, {:?}, {:?}", _thread_name, weighted_index, rng);
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
                zero_evaluation: ZeroEvaluationPov {
                    values: search_data.values.to_relative(bs.board().side_to_move()),
                    policy: search_data.policy,
                }, // q
                net_evaluation: ZeroEvaluationPov {
                    values: v_p.values.to_relative(bs.board().side_to_move()),
                    policy: v_p.policy,
                }, // v
            };

            let _nps = settings.max_nodes.unwrap() as f32 / _elapsed;

            debug_print!(
                "{}",
                &format!("thread {}, {:#}, {}nps", _thread_name, final_mv, _nps)
            );
            debug_print!("{}", &format!("{:#}", final_mv));
            nps_sender
                .send_async(CollectorMessage::GeneratorStatistics(
                    settings.max_nodes.unwrap() as usize,
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

        debug_print!(
            "{}",
            &format!("one done {}s", _sw.elapsed().as_nanos() as f32 / 1e9)
        );
        debug_print!("{}", &"one done!".to_string());
        tz
    }
}
