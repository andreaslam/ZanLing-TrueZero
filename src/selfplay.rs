use crate::{
    boardmanager::BoardStack,
    dataformat::{Position, Simulation, ZeroEvaluation},
    decoder::{convert_board, extract_from_tensor},
    executor::{Packet, ReturnMessage},
    mcts_trainer::{get_move, ExpansionType, TypeRequest},
    settings::SearchSettings,
};
use cozy_chess::{Board, Color, GameStatus, Move};
use flume::Sender;
use rand::{prelude::*, rngs::StdRng};
use rand_distr::{Normal, WeightedIndex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
// selfplay code
#[derive(Clone, Debug)]
pub enum CollectorMessage {
    FinishedGame(Simulation),
    GeneratorStatistics(f32),
    ExecutorStatistics(f32),

    GameResult(Option<Color>),
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
        // let thread_name = std::thread::current()
        //     .name()
        //     .unwrap_or("unnamed")
        //     .to_owned();
        while bs.status() == GameStatus::Ongoing {
            let sw = Instant::now();
            let (mv, v_p, move_idx_piece, search_data, visits) =
                get_move(bs.clone(), &tensor_exe_send, settings.clone(), id).await;
            let elapsed = sw.elapsed().as_nanos() as f32 / 1e9;
            let final_mv = if positions.len() > 30 {
                // when tau is "infinitesimally small", pick the best move
                // or if search nodes = 1, since search_data.policy would return a vec of NANs
                mv
            } else {
                let weighted_index = WeightedIndex::new(&search_data.policy).unwrap();

                let mut rng = StdRng::from_entropy();
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
        println!("one done {}s", elapsed_ms);
        // println!("one done!");
        tz
    }

    pub async fn fast_data(
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
        // let thread_name = std::thread::current()
        //     .name()
        //     .unwrap_or("unnamed")
        //     .to_owned();
        while bs.status() == GameStatus::Ongoing {
            let mut legal_moves: Vec<Move> = Vec::new();
            bs.board().generate_moves(|moves| {
                // Unpack dense move set into move list
                legal_moves.extend(moves);
                false
            });
            // reannotate with net

            let input_tensor = convert_board(&bs);

            // creating a send/recv pair for executor

            let (resender_send, resender_recv) = flume::bounded::<ReturnMessage>(1); // mcts to executor
                                                                                     // let thread_name = std::thread::current()
                                                                                     //     .name()
                                                                                     //     .unwrap_or("unnamed-generator")
                                                                                     //     .to_owned();
            let thread_name = std::thread::current()
                .name()
                .unwrap_or("unnamed-generator")
                .to_owned();
            let pack = Packet {
                job: input_tensor,
                resender: resender_send,
                id: thread_name.clone(),
            };
            tensor_exe_send.send_async(pack).await.unwrap();
            let now_start_proc = SystemTime::now();
            let since_epoch_proc = now_start_proc
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards");

            let epoch_seconds_start_proc = since_epoch_proc.as_nanos();
            let output = resender_recv.recv_async().await.unwrap();
            let now_end_proc = SystemTime::now();
            let since_epoch_proc = now_end_proc
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards");
            let epoch_seconds_end_proc = since_epoch_proc.as_nanos();
            if id % 512 == 0 {
                println!(
                    "{} {} {} recv_fast_data",
                    epoch_seconds_start_proc, epoch_seconds_end_proc, id
                );
            }

            let output = match output {
                ReturnMessage::ReturnMessage(Ok(output)) => output,
                ReturnMessage::ReturnMessage(Err(_)) => panic!("error in returning!"),
            };
            assert!(thread_name == output.id);
            // let (_, _, value, _, _, pol_list) = extract_from_tensor(output.packet, &bs);

            // let pol_list_rand = policy_modification(&pol_list);

            // let weighted_index = WeightedIndex::new(&pol_list_rand).unwrap();
            let mut rng = StdRng::from_entropy();

            // Generate random floats and calculate their sum
            let sum: f32 = (0..settings.max_nodes - 1)
                .map(|_| rng.gen_range(-1.0..=1.0))
                .sum();

            let value = 0.0; // dummy
            let pol_list: Vec<f32> = vec![f32::NAN; legal_moves.len()];
            let pol_list_rand = vec![f32::NAN; legal_moves.len()];
            // Calculate the average
            let q = (sum + value) / settings.max_nodes as f32;
            // let sampled_idx = weighted_index.sample(&mut rng);
            // let mv = legal_moves[sampled_idx];
            let mv = legal_moves.choose(&mut rng).unwrap();
            let mv = *mv;
            let pos = Position {
                board: bs.clone(),
                is_full_search: true,
                // played_mv: mv,
                played_mv: mv,
                zero_visits: 1,
                zero_evaluation: ZeroEvaluation {
                    values: value,
                    policy: pol_list,
                }, // q
                net_evaluation: ZeroEvaluation {
                    values: q,
                    policy: pol_list_rand,
                }, // v
            };
            positions.push(pos);
            bs.play(mv);
            nps_sender
                .send_async(CollectorMessage::GeneratorStatistics(
                    settings.max_nodes as f32,
                ))
                .await
                .unwrap();
        }
        let tz = Simulation {
            positions,
            final_board: bs,
        };
        let elapsed_ms = sw.elapsed().as_nanos() as f32 / 1e9;
        println!("one done {}s", elapsed_ms);
        tz
    }
}

fn policy_modification(pol_list: &Vec<f32>) -> Vec<f32> {
    let mut rng = StdRng::from_entropy();
    let std_dev = 0.5;
    let normal: Normal<f64> = Normal::new(0.0, std_dev).unwrap();
    let mut pol_list_rand = pol_list.clone();
    for value in &mut pol_list_rand {
        let noise = normal.sample(&mut rng);
        *value += noise.abs() as f32;
    }
    pol_list_rand
}
