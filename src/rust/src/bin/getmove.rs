use cozy_chess::Board;
use crossbeam::thread;
use lru::LruCache;
use std::{env, num::NonZeroUsize, panic, process, time::Instant};
use tokio::runtime::Runtime;

use tzrust::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    dataformat::ZeroEvaluationAbs,
    debug_print,
    executor::{
        executor_static,
        Message::{self, StopServer},
        Packet,
    },
    mcts::get_move,
    mcts_trainer::{EvalMode, TypeRequest::NonTrainerSearch},
    settings::{CPUCTSettings, FPUSettings, MovesLeftSettings, PSTSettings, SearchSettings},
};

fn main() {
    panic::set_hook(Box::new(|panic_info| {
        eprintln!("Panic occurred: {:?}", panic_info);
        std::process::exit(2);
    }));

    env::set_var("RUST_BACKTRACE", "1");

    let board = Board::from_fen(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        false,
    )
    .unwrap();

    // generate the legal moves once outside get_move
    // this is the ordering used by the root children when the root is expanded

    let mut move_list = Vec::new();

    board.generate_moves(|moves| {
        move_list.extend(moves);
        false
    });

    let total_moves = move_list.len();

    // start the executor

    let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Packet>(1);

    let (ctrl_sender, ctrl_recv) = flume::bounded::<Message>(1);

    thread::scope(|s| {
        s.builder()
            .name("executor".to_string())
            .spawn(move |_| {
                executor_static(
                    r"/Users/andreas/Desktop/Code/RemoteFolder/ZanLing-TrueZero/chess_16x128_gen3634.pt".to_string(),
                    tensor_exe_recv,
                    ctrl_recv,
                    1,
                )
            })
            .unwrap();

        debug_print!(
            "Number of legal moves: {}",
            total_moves
        );

        let bs =
            BoardStack::new(board);

        let sw =
            Instant::now();

        let m_settings =
            MovesLeftSettings {
                moves_left_weight: 0.03,
                moves_left_clip: 20.0,
                moves_left_sharpness: 0.5,
            };

        let max_nodes = 1;

        let settings =
            SearchSettings {
                fpu: FPUSettings {
                    root_fpu: 0.1,
                    children_fpu: 0.1,
                },

                wdl: EvalMode::Wdl,

                moves_left: Some(m_settings),

                c_puct: CPUCTSettings {
                    root_c_puct: 2.0,
                    children_c_puct: 2.0,
                },

                max_nodes: Some(max_nodes),

                alpha: 0.03,

                eps: 0.25,

                search_type:
                    NonTrainerSearch,

                pst: PSTSettings {
                    root_pst: 1.75,
                    children_pst: 1.5,
                },

                batch_size: 1,
            };

        let rt =
            Runtime::new().unwrap();

        let mut cache:
            LruCache<
                CacheEntryKey,
                ZeroEvaluationAbs,
            > =
            LruCache::new(
                NonZeroUsize::new(100000)
                    .unwrap()
            );

        let (
            best_move,
            nn_data,
            _move_idx,
            search_data,
            visits,
        ) =
            rt.block_on(async {
                get_move(
                    bs,
                    tensor_exe_send.clone(),
                    settings,
                    None,
                    &mut cache,
                )
                .await
            });

        let elapsed_ms =
            sw.elapsed()
                .as_nanos() as f32
                / 1e6;

        // network policy contains the root nn priors

        let network_policy =
            &nn_data.policy;

        // mcts policy contains the visit count distribution

        let mcts_policy =
            &search_data.policy;

        // print basic consistency information

        println!();
        println!("============================================================");
        println!("ROOT POLICY DIAGNOSTIC");
        println!("============================================================");
        println!();

        println!(
            "Legal moves:       {}",
            move_list.len()
        );

        println!(
            "Network policy:     {} entries",
            network_policy.len()
        );

        println!(
            "MCTS policy:        {} entries",
            mcts_policy.len()
        );

        println!(
            "Root visits:        {}",
            visits
        );

        println!(
            "Best move:          {}",
            best_move
        );

        println!(
            "Elapsed:            {:.3} ms",
            elapsed_ms
        );

        println!();

        // check that all policy vectors use the same move ordering

        assert_eq!(
            move_list.len(),
            network_policy.len(),
            "Move list and network policy have different lengths"
        );

        assert_eq!(
            move_list.len(),
            mcts_policy.len(),
            "Move list and MCTS policy have different lengths"
        );

        // check probability sums

        let network_sum:
            f32 =
            network_policy
                .iter()
                .sum();

        let mcts_sum:
            f32 =
            mcts_policy
                .iter()
                .sum();

        println!(
            "Network prior sum: {:.9}",
            network_sum
        );

        println!(
            "MCTS policy sum:   {:.9}",
            mcts_sum
        );

        println!();

        // print the full policy table

        println!(
            "{:<8} {:>14} {:>14} {:>14}",
            "MOVE",
            "NETWORK",
            "MCTS",
            "MCTS VISITS"
        );

        println!(
            "{:-<8} {:-<14} {:-<14} {:-<14}",
            "",
            "",
            "",
            ""
        );

        // recover the approximate visit count from the mcts policy
        // pi is visits divided by total root visits

        for i in 0..move_list.len() {
            let mv =
                move_list[i];

            let network =
                network_policy[i];

            let mcts =
                mcts_policy[i];

            let estimated_visits =
                mcts * visits as f32;

            println!(
                "{:<8} {:>14.6} {:>14.6} {:>14.2}",
                format!("{}", mv),
                network,
                mcts,
                estimated_visits
            );
        }

        println!();

        // sort a copy by network policy
        // this makes the raw nn move ordering easier to inspect

        let mut network_sorted:
            Vec<(usize, f32)> =
            network_policy
                .iter()
                .copied()
                .enumerate()
                .collect();

        network_sorted.sort_by(
            |a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap()
            }
        );

        println!(
            "NETWORK POLICY — SORTED"
        );

        println!(
            "{:<4} {:<8} {:>14}",
            "RANK",
            "MOVE",
            "PRIOR"
        );

        for (rank, (idx, prob)) in
            network_sorted.iter().enumerate()
        {
            println!(
                "{:<4} {:<8} {:>14.6}",
                rank + 1,
                move_list[*idx],
                prob
            );
        }

        println!();

        // sort a copy by mcts policy
        // this shows the final search distribution

        let mut mcts_sorted:
            Vec<(usize, f32)> =
            mcts_policy
                .iter()
                .copied()
                .enumerate()
                .collect();

        mcts_sorted.sort_by(
            |a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap()
            }
        );

        println!(
            "MCTS POLICY — SORTED"
        );

        println!(
            "{:<4} {:<8} {:>14} {:>14}",
            "RANK",
            "MOVE",
            "POLICY",
            "VISITS"
        );

        for (rank, (idx, prob)) in
            mcts_sorted.iter().enumerate()
        {
            let estimated_visits =
                prob * visits as f32;

            println!(
                "{:<4} {:<8} {:>14.6} {:>14.2}",
                rank + 1,
                move_list[*idx],
                prob,
                estimated_visits
            );
        }

        println!();

        // compare the network prior against the final mcts distribution

        println!(
            "NETWORK vs MCTS"
        );

        println!(
            "{:<8} {:>14} {:>14} {:>14}",
            "MOVE",
            "NETWORK",
            "MCTS",
            "MCTS/NN"
        );

        for i in 0..move_list.len() {
            let nn =
                network_policy[i];

            let mcts =
                mcts_policy[i];

            let ratio =
                if nn > 0.0 {
                    mcts / nn
                } else {
                    0.0
                };

            println!(
                "{:<8} {:>14.6} {:>14.6} {:>14.3}",
                move_list[i],
                nn,
                mcts,
                ratio
            );
        }

        println!();

        println!(
            "BEST MOVE: {}",
            best_move
        );

        println!();

        let nps =
            max_nodes as f32
                / (
                    sw.elapsed()
                        .as_nanos() as f32
                        / 1e9
                );

        println!(
            "Nodes per second: {:.1}",
            nps
        );

        ctrl_sender
            .send(StopServer)
            .unwrap();

        process::exit(0);
    })
    .unwrap();
}
