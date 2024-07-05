use crate::mcts_trainer::{EvalMode, TypeRequest};

#[derive(Clone, Debug, PartialEq, Copy)]

pub struct SearchSettings {
    pub fpu: f32,
    pub wdl: EvalMode, // if WDL is None then automatically use value
    pub moves_left: Option<MovesLeftSettings>,
    pub c_puct: f32,
    pub max_nodes: u128,
    pub alpha: f32,
    pub eps: f32,
    pub search_type: TypeRequest,
    pub pst: f32,
    // pub cap_randomisation: Option<PlayoutCapSettings>, // "playout cap randomisation", TODO use option<playoutcapsettings>
}

#[derive(Clone, Debug, PartialEq, Copy)]
pub struct PlayoutCapSettings {
    pub start_min: usize,
    pub finish_min: usize,
    pub start_max: usize,
    pub finish_max: usize,
}

impl PlayoutCapSettings {
    pub fn new(start_min: usize, finish_min: usize, start_max: usize, finish_max: usize) -> Self {
        Self {
            start_min,
            finish_min,
            start_max,
            finish_max,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Copy)]
pub struct MovesLeftSettings {
    pub moves_left: f32,
    pub moves_left_weight: f32,
    pub moves_left_clip: f32,
    pub moves_left_sharpness: f32,
}

impl MovesLeftSettings {
    pub fn new(moves_left_weight: f32, moves_left_clip: f32, moves_left_sharpness: f32) -> Self {
        Self {
            moves_left: f32::NAN,
            moves_left_weight,
            moves_left_clip,
            moves_left_sharpness,
        }
    }
}
