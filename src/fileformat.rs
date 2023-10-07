use bytemuck::cast_slice;
use cozy_chess::{Color, GameStatus, Move};
use serde::Serialize;
use std::{
    cmp::{max, min},
    fs::File,
    io::{self, BufWriter, Seek, Write},
    path::{Path, PathBuf},
};

use crate::{
    boardmanager::BoardStack,
    dataformat::{Position, Simulation},
    decoder::convert_board,
    mvs::get_contents,
};

#[derive(Serialize)]
struct MetaData<'a> {
    game: &'a str,

    input_bool_shape: &'a [usize],
    input_scalar_count: usize,
    policy_shape: &'a [usize],

    game_count: usize,
    position_count: usize,
    includes_terminal_positions: bool,
    includes_game_start_indices: bool,

    max_game_length: i32,
    min_game_length: i32,
    root_wdl: [f32; 3],
    hit_move_limit: f32,

    scalar_names: &'static [&'static str],
}

#[derive(Debug)]
pub struct BinaryOutput {
    game: String,
    path: PathBuf,

    bin_write: BufWriter<File>,
    off_write: BufWriter<File>,
    json_tmp_write: BufWriter<File>,

    game_count: usize,
    position_count: usize,

    max_game_length: Option<i32>,
    min_game_length: Option<i32>,

    total_root_wdl: [u64; 3],
    hit_move_limit_count: u64,

    next_offset: u64,
    game_start_indices: Vec<u64>,

    finished: bool,
}

#[derive(Debug)]
struct Scalars {
    game_id: usize,
    pos_index: usize,
    game_length: usize,
    zero_visits: u64,
    is_full_search: bool,
    is_final_position: bool,
    is_terminal: bool,
    hit_move_limit: bool,
    available_mv_count: usize,
    played_mv: isize,
    kdl_policy: f32,
    final_values: f32, // z
    zero_values: f32,  // q
    net_values: f32,   // v
}

impl BinaryOutput {
    pub fn new(path: impl AsRef<Path>, game: &str) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        assert!(
            path.extension().is_none(),
            "Binary output path should not have an extension, .bin and .json are added automatically"
        );

        //TODO try buffer sizes again
        let bin_write = BufWriter::new(File::create(path.with_extension("bin"))?);
        let off_write = BufWriter::new(File::create(path.with_extension("off"))?);
        let json_tmp_write = BufWriter::new(File::create(path.with_extension("json.tmp"))?);

        Ok(BinaryOutput {
            game: game.to_string(),

            bin_write,
            off_write,
            json_tmp_write,

            path,

            game_count: 0,
            position_count: 0,

            max_game_length: None,
            min_game_length: None,

            total_root_wdl: [0, 0, 0],
            hit_move_limit_count: 0,

            next_offset: 0,
            game_start_indices: vec![],

            finished: false,
        })
    }

    pub fn append(&mut self, simulation: &Simulation) -> io::Result<()> {
        let Simulation {
            positions,
            final_board,
        } = simulation;

        // collect metadata statistics
        let game_id = self.game_count;
        let game_length = positions.len();

        self.game_start_indices.push(self.position_count as u64);

        self.game_count += 1;
        self.position_count += 1 + game_length;

        self.max_game_length = Some(max(game_length as i32, self.max_game_length.unwrap_or(-1)));
        self.min_game_length = Some(min(
            game_length as i32,
            self.min_game_length.unwrap_or(i32::MAX),
        ));

        // let outcome = final_board.outcome().unwrap_or(Outcome::Draw);

        let outcome: Option<Color> = match final_board.status() {
            GameStatus::Drawn => None,
            GameStatus::Won => Some(!final_board.board().side_to_move()),
            GameStatus::Ongoing => panic!("Game is still ongoing!"),
        };

        // write the positions
        for (pos_index, position) in positions.iter().enumerate() {
            let &Position {
                ref board,
                is_full_search,
                played_mv,
                zero_visits,
                ref zero_evaluation,
                ref net_evaluation,
            } = position;

            let (available_mv_count, policy_indices) = collect_policy_indices(board);
            assert_eq!(available_mv_count, zero_evaluation.policy.len());
            assert_eq!(available_mv_count, net_evaluation.policy.len());
            let mut played_mv = played_mv;
            if board.board().side_to_move() == Color::Black {
                played_mv = Move {
                    from: played_mv.from.flip_rank(),
                    to: played_mv.to.flip_rank(),
                    promotion: played_mv.promotion,
                };
            }
            // get all idx from mvs.rs
            let all_idx = get_contents();

            let played_mv_index = all_idx.iter().position(|&x| x == played_mv).unwrap();

            // let played_mv_index = self.mapper.move_to_index(board, played_mv);
            // let kdl_policy = kdl_divergence(&zero_evaluation.policy, &net_evaluation.policy);
            let kdl_policy = f32::NAN;
            let moves_left = game_length + 1 - pos_index;
            let stored_policy = &zero_evaluation.policy;
            let final_values = match outcome {
                Some(Color::White) => net_evaluation.values,
                Some(Color::Black) => -net_evaluation.values,
                None => 0.0,
            };
            let scalars = Scalars {
                game_id,
                pos_index,
                game_length,
                zero_visits,
                is_full_search,
                is_final_position: false,
                is_terminal: false,
                hit_move_limit: false,
                available_mv_count: stored_policy.len(),
                played_mv: played_mv_index as isize,
                kdl_policy,
                final_values,
                zero_values: zero_evaluation.values,
                net_values: net_evaluation.values,
            };

            self.append_position(board, &scalars, &policy_indices, stored_policy)?;
        }
        let final_values = match outcome {
            Some(Color::White) => 1.0,
            Some(Color::Black) => -1.0,
            None => 0.0,
        };
        let scalars = Scalars {
            game_id,
            pos_index: game_length,
            game_length,
            zero_visits: 0,
            is_full_search: false,
            is_final_position: true,
            is_terminal: final_board.is_terminal(),
            hit_move_limit: !final_board.is_terminal(),
            available_mv_count: 0,
            played_mv: -1,
            kdl_policy: f32::NAN,
            final_values,
            zero_values: f32::NAN,
            //TODO in theory we could ask the network, but this is only really meaningful for muzero
            net_values: f32::NAN,
        };

        self.append_position(&final_board, &scalars, &[], &[])?;

        Ok(())
    }

    fn append_position(
        &mut self,
        board: &BoardStack,
        scalars: &Scalars,
        policy_indices: &[u32],
        policy_values: &[f32],
    ) -> io::Result<()> {
        // encode board
        let board_bools: Vec<f32> =
            Vec::try_from(convert_board(board)).unwrap()[8 * 64 - 1..].to_vec();
        let board_scalars: Vec<f32> =
            Vec::try_from(convert_board(board)).unwrap()[0..8 * 64].to_vec();
        // assert_eq!(self.mapper.input_bool_len(), board_bools.len());
        // assert_eq!(self.mapper.input_scalar_count(), board_scalars.len());
        // assert_eq!(
        //     (self.mapper.input_bool_len() + 7) / 8,
        //     board_bools.storage().len()
        // );

        // check that everything makes sense
        let policy_len = policy_indices.len();
        assert_eq!(policy_len, policy_values.len());
        // assert_normalized_or_nan(scalars.zero_values.wdl.sum());
        // assert_normalized_or_nan(scalars.net_values.wdl.sum());
        // assert_normalized_or_nan(scalars.final_values.wdl.sum());
        if policy_len != 0 {
            assert_normalized_or_nan(policy_values.iter().sum());
        }

        // save current offset
        // we keep track of the offset ourselves because seeking/stream_position flushes the buffer and is slow
        debug_assert_eq!(self.next_offset, self.bin_write.stream_position()?);
        self.off_write.write_all(&self.next_offset.to_le_bytes())?;

        // actually write stuff to the bin file
        let scalars = scalars.to_vec();
        let data_to_write: &[&[u8]] = &[
            cast_slice(&scalars),
            cast_slice(&board_bools),
            cast_slice(&board_scalars),
            cast_slice(policy_indices),
            cast_slice(policy_values),
        ];
        for &data in data_to_write {
            self.bin_write.write_all(data)?;
            self.next_offset += data.len() as u64;
        }

        Ok(())
    }

    pub fn finish(&mut self) -> io::Result<()> {
        if self.finished {
            panic!("This output is already finished")
        }
        self.finished = true;

        let meta = MetaData {
            game: &self.game,
            scalar_names: Scalars::NAMES,
            input_bool_shape: &[13, 8, 8], // pieces + EP
            input_scalar_count: 8,         // move turn, counters, castling
            policy_shape: &[1880 as usize],
            game_count: self.game_count,
            position_count: self.position_count,
            includes_terminal_positions: true,
            includes_game_start_indices: true,
            max_game_length: self.max_game_length.unwrap_or(-1),
            min_game_length: self.min_game_length.unwrap_or(-1),
            // root_wdl: (self.total_root_wdl.cast::<f32>() / self.game_count as f32).to_slice(),
            root_wdl: [0.0, 0.0, 0.0],
            hit_move_limit: self.hit_move_limit_count as f32 / self.game_count as f32,
        };

        serde_json::to_writer_pretty(&mut self.json_tmp_write, &meta)?;
        self.off_write
            .write_all(cast_slice(&self.game_start_indices))?;

        self.json_tmp_write.flush()?;
        self.bin_write.flush()?;
        self.off_write.flush()?;

        let path_json_tmp = self.path.with_extension("json.tmp");
        let path_json = self.path.with_extension("json");
        std::fs::rename(path_json_tmp, path_json)?;

        Ok(())
    }

    pub fn game_count(&self) -> usize {
        self.game_count
    }
}

fn collect_policy_indices(board: &BoardStack) -> (usize, Vec<u32>) {
    match board.status() {
        GameStatus::Ongoing => {
            let mut policy_indices: Vec<u32> = vec![];

            let mut move_list = Vec::new();
            board.board().generate_moves(|moves| {
                // Unpack dense move set into move list
                move_list.extend(moves);
                false
            });

            for m in move_list.clone() {
                policy_indices.push(move_list.iter().position(|&x| x == m).unwrap() as u32);
            }

            (policy_indices.len(), policy_indices)
        }
        GameStatus::Drawn | GameStatus::Won => (0, vec![]),
    }
}

fn assert_normalized_or_nan(x: f32) {
    assert!(x.is_nan() || (1.0 - x).abs() < 0.001);
}

impl Scalars {
    const NAMES: &'static [&'static str] = &[
        "game_id",
        "pos_index",
        "game_length",
        "zero_visits",
        "is_full_search",
        "is_final_position",
        "is_terminal",
        "hit_move_limit",
        "available_mv_count",
        "played_mv",
        "kdl_policy",
        "final_v",
        // "final_wdl_w",
        // "final_wdl_d",
        // "final_wdl_l",
        // "final_moves_left",
        "zero_v",
        // "zero_wdl_w",
        // "zero_wdl_d",
        // "zero_wdl_l",
        // "zero_moves_left",
        "net_v",
        // "net_wdl_w",
        // "net_wdl_d",
        // "net_wdl_l",
        // "net_moves_left",
    ];

    fn to_vec(&self) -> Vec<f32> {
        let mut result = vec![
            self.game_id as f32,
            self.pos_index as f32,
            self.game_length as f32,
            self.zero_visits as f32,
            self.is_full_search as u8 as f32,
            self.is_final_position as u8 as f32,
            self.is_terminal as u8 as f32,
            self.hit_move_limit as u8 as f32,
            self.available_mv_count as f32,
            self.played_mv as f32,
            self.kdl_policy as f32,
        ];

        result.extend_from_slice(&[self.final_values]);
        result.extend_from_slice(&[self.zero_values]);
        result.extend_from_slice(&[self.net_values]);

        assert_eq!(result.len(), Self::NAMES.len());
        result
    }
}
