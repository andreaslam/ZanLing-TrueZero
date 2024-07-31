// https://github.com/KarelPeeters/kZero/blob/883130717f1a9e2a12945579610ddcb881643469/rust/kz-selfplay/src/simulation.rs#

use cozy_chess::Move;

use crate::{boardmanager::BoardStack, mcts_trainer::Wdl};

#[derive(Debug, Clone)]
pub struct ZeroEvaluation {
    /// The (normalized) values.
    pub values: ZeroValuesPov,

    /// The (normalized) policy "vector", only containing the available moves in the order they are yielded by `available_moves`.
    pub policy: Vec<f32>,
}

/// A single position in a game.
#[derive(Debug, Clone)]
pub struct Position {
    pub board: BoardStack,
    pub is_full_search: bool,
    pub played_mv: Move,

    pub zero_visits: u64,
    pub zero_evaluation: ZeroEvaluation,
    pub net_evaluation: ZeroEvaluation,
}

/// A full game.
#[derive(Debug, Clone)]
pub struct Simulation {
    pub positions: Vec<Position>,
    // can be non-terminal if the game was stopped by the length limit
    pub final_board: BoardStack,
}

impl Simulation {
    pub fn start_board(&self) -> &BoardStack {
        match self.positions.get(0) {
            Some(pos) => &pos.board,
            None => &self.final_board,
        }
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct ZeroValuesPov {
    pub value: f32,
    pub wdl: Wdl,
    pub moves_left: f32,
}
impl ZeroValuesPov {
    pub fn to_slice(self) -> [f32; 5] {
        [
            self.value,
            self.wdl.w,
            self.wdl.d,
            self.wdl.l,
            self.moves_left,
        ]
    }
}
