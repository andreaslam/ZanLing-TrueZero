use crate::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    dataformat::{Position, Simulation, ZeroEvaluationAbs, ZeroEvaluationPov},
    debug_print,
    executor::Packet,
    mcts_trainer::get_move,
    settings::SearchSettings,
    uci::UCIMsg,
};
use cozy_chess::{Board, Color, GameStatus, Move};
use flume::Sender;
use lru::LruCache;
use rand::prelude::*;
use rand_distr::WeightedIndex;
use std::time::Instant;
use tokio::sync::watch;
// selfplay code
#[derive(Clone, Debug)]
pub enum CollectorMessage {
    FinishedGame(Simulation),
    GeneratorStatistics(usize),
    ExecutorStatistics(usize),
    GameResult(Option<Color>),
    TestingResult(Option<bool>), // engine_0 win = true, engine_0 loss = false
    TestingControl(bool),        // true for continue, false for stop
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
        custom_startpos: Option<BoardStack>,
        mut pause_receiver: Option<watch::Receiver<bool>>,
    ) -> Simulation {
        let _sw = Instant::now();
        let mut bs = match custom_startpos {
            Some(bs) => bs,
            None => BoardStack::new(Board::default()),
        };
        // let mut value: Vec<f32> = Vec::new();
        let mut positions: Vec<Position> = Vec::new();
        let _thread_name = std::thread::current()
            .name()
            .unwrap_or("unnamed")
            .to_owned();
        while bs.status() == GameStatus::Ongoing {
            wait_until_resumed(&mut pause_receiver).await;

            let (stop_receiver, stop_task) = match pause_receiver.as_ref() {
                Some(receiver) => {
                    let (stop_sender, stop_receiver) = flume::bounded::<UCIMsg>(1);
                    let mut receiver = receiver.clone();
                    let task = tokio::spawn(async move {
                        while !*receiver.borrow() {
                            if receiver.changed().await.is_err() {
                                return;
                            }
                        }
                        let _ = stop_sender.send(UCIMsg::UCIStopMessage);
                    });
                    (Some(stop_receiver), Some(task))
                }
                None => (None, None),
            };

            let _sw = Instant::now();
            let (mv, v_p, _move_idx_piece, search_data, visits) = get_move(
                bs.clone(),
                tensor_exe_send,
                *settings,
                id,
                cache,
                stop_receiver,
            )
            .await;
            if let Some(task) = stop_task {
                task.abort();
            }

            wait_until_resumed(&mut pause_receiver).await;
            let _elapsed = _sw.elapsed().as_nanos() as f32 / 1e9;
            // sample for the first 30 full moves (60 plies), then play greedily.
            const TEMPERATURE_CUTOFF_PLIES: usize = 60;
            let final_mv = if positions.len() >= TEMPERATURE_CUTOFF_PLIES {
                // tau -> 0: pick the most-visited move (already returned as `mv`)
                mv
            } else {
                // tau = 1: sample proportional to MCTS visit counts.
                let mut legal_moves: Vec<Move> = Vec::new();
                bs.board().generate_moves(|moves| {
                    legal_moves.extend(moves);
                    false
                });

                assert_eq!(legal_moves.len(), search_data.policy.len());

                let legal_moves_with_policy: Vec<(Move, f32)> = legal_moves
                    .into_iter()
                    .zip(search_data.policy.iter().copied())
                    .collect();

                let weighted_index =
                    WeightedIndex::new(legal_moves_with_policy.iter().map(|(_, policy)| *policy))
                        .unwrap();

                let mut rng = rand::thread_rng();
                let sampled_idx = weighted_index.sample(&mut rng);

                legal_moves_with_policy[sampled_idx].0
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

async fn wait_until_resumed(pause_receiver: &mut Option<watch::Receiver<bool>>) {
    let Some(receiver) = pause_receiver.as_mut() else {
        return;
    };

    while *receiver.borrow() {
        if receiver.changed().await.is_err() {
            return;
        }
    }
}
