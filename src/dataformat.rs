// https://github.com/KarelPeeters/kZero/blob/883130717f1a9e2a12945579610ddcb881643469/rust/kz-selfplay/src/simulation.rs#

use cozy_chess::{Color, Move};

use crate::{boardmanager::BoardStack, mcts_trainer::Wdl};

#[derive(Debug, Clone)]
pub struct ZeroEvaluationPov {
    /// The (normalized) values.
    pub values: ZeroValuesPov,

    /// The (normalized) policy "vector", only containing the available moves in the order they are yielded by `available_moves`.
    pub policy: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ZeroEvaluationAbs {
    /// The (normalized) values.
    pub values: ZeroValuesAbs,

    /// The (normalized) policy "vector", only containing the available moves in the order they are yielded by `available_moves`.
    pub policy: Vec<f32>,
}

impl ZeroEvaluationAbs {
    pub fn get_average(self, visits: u32) -> Self {
        Self {
            values: self.values.get_average(visits),
            policy: self.policy,
        }
    }
}

/// A single position in a game.
#[derive(Debug, Clone)]
pub struct Position {
    pub board: BoardStack,
    pub is_full_search: bool,
    pub played_mv: Move,

    pub zero_visits: u64,
    pub zero_evaluation: ZeroEvaluationPov,
    pub net_evaluation: ZeroEvaluationPov,
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
    pub fn to_absolute(self, player: Color) -> ZeroValuesAbs {
        match player {
            Color::White => ZeroValuesAbs {
                value: self.value,
                wdl: self.wdl,
                moves_left: self.moves_left,
            },
            Color::Black => ZeroValuesAbs {
                value: -self.value,
                wdl: self.wdl.flip(),
                moves_left: self.moves_left,
            },
        }
    }
    pub fn nan() -> Self {
        ZeroValuesPov {
            value: f32::NAN,
            wdl: Wdl::nan(),
            moves_left: f32::NAN,
        }
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct ZeroValuesAbs {
    pub value: f32,
    pub wdl: Wdl,
    pub moves_left: f32,
}

impl ZeroValuesAbs {
    pub fn to_relative(self, player: Color) -> ZeroValuesPov {
        match player {
            Color::White => ZeroValuesPov {
                value: self.value,
                wdl: self.wdl,
                moves_left: self.moves_left,
            },
            Color::Black => ZeroValuesPov {
                value: -self.value,
                wdl: self.wdl.flip(),
                moves_left: self.moves_left,
            },
        }
    }
    pub fn nan() -> Self {
        ZeroValuesAbs {
            value: f32::NAN,
            wdl: Wdl::nan(),
            moves_left: f32::NAN,
        }
    }
    pub fn zeros() -> Self {
        ZeroValuesAbs {
            value: 0.0,
            wdl: Wdl::zeros(),
            moves_left: 0.0,
        }
    }
    pub fn get_average(self, visits: u32) -> Self {
        Self {
            value: self.value / visits as f32,
            wdl: self.wdl.get_average(visits),
            moves_left: self.moves_left / visits as f32,
        }
    }
}

impl std::ops::AddAssign for ZeroValuesAbs {
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
        self.wdl += rhs.wdl;
        self.moves_left += rhs.moves_left;
    }
}
