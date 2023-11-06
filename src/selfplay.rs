pub fn play_game(&self,tensor_exe_send:Sender<Tensor>, eval_exe_recv :Receiver<Message>) -> Simulation {
        let mut bs = BoardStack::new(Board::default());
        // let mut value: Vec<f32> = Vec::new();
        let mut positions: Vec<Position> = Vec::new();
        while bs.status() == GameStatus::Ongoing {
            let mut sw = Stopwatch::new();
            sw.start();
            let (mv, v_p, move_idx_piece, search_data, visits) = get_move(bs.clone(),tensor_exe_send.clone(), eval_exe_recv.clone());
            sw.stop();
            let final_mv = if positions.len() > 30 {
                // when tau is "infinitesimally small", pick the best move
                let nps = MAX_NODES as f32 / (sw.elapsed_ms() as f32 / 1000.0);
                println!("{:#}, {}nps", mv, nps);
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
                let nps = MAX_NODES as f32 / (sw.elapsed_ms() as f32 / 1000.0);
                println!("{:#}, {}nps", mv, nps);
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
        println!("one done!");
        tz
    }
