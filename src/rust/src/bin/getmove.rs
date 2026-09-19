use cozy_chess::{Board, Move};
use crossbeam::thread;
use crossterm::{
    event::{self, Event, KeyCode},
    terminal::{disable_raw_mode, enable_raw_mode},
};
use lru::LruCache;
use std::{
    env,
    io::{self, Write},
    num::NonZeroUsize,
    panic, process,
    time::Instant,
};
use tokio::runtime::Runtime;

use tzrust::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    dataformat::ZeroEvaluationAbs,
    executor::{
        executor_static,
        Message::{self, StopServer},
        Packet,
    },
    mcts_trainer::{EvalMode, Tree, TypeRequest::NonTrainerSearch},
    settings::{CPUCTSettings, FPUSettings, MovesLeftSettings, PSTSettings, SearchSettings},
};

fn print_tree(tree: &Tree) {
    if tree.nodes[0].children.is_empty() {
        println!("No root children yet.");
        return;
    }

    let root_visits = tree.nodes[0].visits;

    let mut legal_moves = Vec::new();
    tree.board.board().generate_moves(|moves| {
        legal_moves.extend(moves);
        false
    });

    let mut policy_data: Vec<(usize, Move, f32, u32)> = tree.nodes[0]
        .children
        .clone()
        .map(|child| {
            let node = &tree.nodes[child];
            let mv = node.mv.expect("Root child has no move");

            let legal_idx = legal_moves
                .iter()
                .position(|&legal_mv| legal_mv == mv)
                .expect("Tree child is not a legal move");

            (legal_idx, mv, node.policy, node.visits)
        })
        .collect();

    policy_data.sort_by(|a, b| b.3.cmp(&a.3).then_with(|| b.2.partial_cmp(&a.2).unwrap()));

    println!();
    println!("root visits: {}", root_visits);
    println!("idx | move  | network_policy | visits");
    println!("----+-------+-----------------+-------");

    for (idx, mv, policy, visits) in &policy_data {
        println!("{:>3} | {:>5} | {:>15.6} | {:>6}", idx, mv, policy, visits);
    }

    println!();

    if let Some(best) = tree.nodes[0]
        .children
        .clone()
        .max_by_key(|&child| tree.nodes[child].visits)
    {
        println!(
            "BEST MOVE: {} ({} visits)",
            tree.nodes[best].mv.unwrap(),
            tree.nodes[best].visits
        );
    }

    println!();
}

fn main() {
    panic::set_hook(Box::new(|panic_info| {
        eprintln!("Panic occurred: {:?}", panic_info);
        std::process::exit(2);
    }));

    env::set_var("RUST_BACKTRACE", "1");

    // -------------------------------------------------------------------------
    // Board
    // -------------------------------------------------------------------------

    let board = Board::default();

    let mut move_list: Vec<Move> = Vec::new();

    board.generate_moves(|moves| {
        move_list.extend(moves);
        false
    });

    println!("Number of legal moves: {}", move_list.len());

    // -------------------------------------------------------------------------
    // Executor
    // -------------------------------------------------------------------------

    let (tensor_exe_send, tensor_exe_recv) = flume::bounded::<Packet>(1);

    let (ctrl_sender, ctrl_recv) = flume::bounded::<Message>(1);

    thread::scope(|s| {
        s.builder()
            .name("executor".to_string())
            .spawn(move |_| {
                executor_static(
                    r"/Users/andreas/Desktop/Code/RemoteFolder/TrueZero/hidden/chess_16x128_gen3634.pt"
                        .to_string(),
                    tensor_exe_recv,
                    ctrl_recv,
                    1,
                )
            })
            .unwrap();

        // ---------------------------------------------------------------------
        // Search setup
        // ---------------------------------------------------------------------

        let bs = BoardStack::new(board);

        let m_settings = MovesLeftSettings {
            moves_left_weight: 0.03,
            moves_left_clip: 20.0,
            moves_left_sharpness: 0.5,
        };

        let settings = SearchSettings {
            fpu: FPUSettings {
                root_fpu: 1.0,
                children_fpu: 0.0,
            },

            wdl: EvalMode::Wdl,

            moves_left: Some(m_settings),

            c_puct: CPUCTSettings {
                root_c_puct: 2.0,
                children_c_puct: 2.0,
            },

            // No automatic visit limit.
            // We manually perform exactly one tree.step() per `s`.
            max_nodes: None,

            alpha: 0.03,
            eps: 0.25,

            search_type: NonTrainerSearch,

            pst: PSTSettings {
                root_pst: 1.75,
                children_pst: 1.5,
            },

            batch_size: 1,
        };

        let mut tree = Tree::new(bs, settings);

        if tree.board.is_terminal() {
            panic!("No valid move!/Board is already game over!");
        }

        let rt = Runtime::new().unwrap();

        let mut cache: LruCache<CacheEntryKey, ZeroEvaluationAbs> =
            LruCache::new(NonZeroUsize::new(100000).unwrap());

        // ---------------------------------------------------------------------
        // Interactive search
        // ---------------------------------------------------------------------

        println!();
        println!("Interactive MCTS");
        println!("  s = one MCTS visit");
        println!("  q = quit");
        println!();

        /*
         * Raw mode is ONLY enabled while waiting for keyboard input.
         *
         * It is deliberately disabled while tree.step() and all debug/TUI
         * output are running. This preserves the terminal behaviour that
         * existed before crossterm was introduced.
         */
        enable_raw_mode().unwrap();

        let mut running = true;

        while running {
            match event::read().unwrap() {
                Event::Key(key) => match key.code {
                    KeyCode::Char('s') => {
                        /*
                         * Stop raw mode BEFORE doing anything that prints.
                         *
                         * tree.step() itself produces debug/TUI output, so
                         * running it in raw mode causes the broken indentation.
                         */
                        disable_raw_mode().unwrap();

                        let visit_start = Instant::now();

                        let visit_number = tree.nodes[0].visits + 1;

                        println!();
                        println!("=== MCTS visit {} ===", visit_number);

                        /*
                         * Exactly ONE MCTS iteration:
                         *
                         * selection
                         * expansion
                         * NN evaluation
                         * backpropagation
                         *
                         * The tree remains alive after this call.
                         */
                        rt.block_on(async {
                            tree.step(
                                &tensor_exe_send,
                                Instant::now(),
                                0,
                                &mut cache,
                            )
                            .await;
                        });

                        let elapsed_ms =
                            visit_start.elapsed().as_nanos() as f32 / 1e6;

                        print_tree(&tree);

                        println!("{:.3} ms", elapsed_ms);
                        println!("Press `s` for another visit, `q` to quit.");

                        io::stdout().flush().unwrap();

                        /*
                         * Re-enter raw mode only after all output is finished.
                         */
                        enable_raw_mode().unwrap();
                    }

                    KeyCode::Char('q') => {
                        disable_raw_mode().unwrap();
                        running = false;
                    }

                    _ => {}
                },

                _ => {}
            }
        }

        // ---------------------------------------------------------------------
        // Final output
        // ---------------------------------------------------------------------

        disable_raw_mode().unwrap();

        println!();
        println!("Final tree:");
        print_tree(&tree);

        println!("Total root visits: {}", tree.nodes[0].visits);

        // ---------------------------------------------------------------------
        // Stop executor
        // ---------------------------------------------------------------------

        ctrl_sender.send(StopServer).unwrap();

        process::exit(0);
    })
    .unwrap();
}
