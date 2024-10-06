use crate::mcts_trainer::{EvalMode, TypeRequest};

#[derive(Clone, Debug, PartialEq, Copy)]

pub struct SearchSettings {
    pub fpu: FPUSettings,
    pub wdl: EvalMode, // if WDL is None then automatically use value
    pub moves_left: Option<MovesLeftSettings>,
    pub c_puct: CPUCTSettings,
    pub max_nodes: Option<u128>,
    pub alpha: f32,
    pub eps: f32,
    pub search_type: TypeRequest,
    pub pst: PSTSettings,
    pub batch_size: usize,
    // pub cap_randomisation: Option<PlayoutCapSettings>,
}

#[derive(Clone, Debug, PartialEq, Copy)]
pub struct PSTSettings {
    pub root_pst: f32,
    pub children_pst: f32,
}
impl PSTSettings {
    /// returns the default value of FPUSettings
    pub fn default(self) -> PSTSettings {
        PSTSettings {
            root_pst: 1.75,
            children_pst: 1.5,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Copy)]
pub struct FPUSettings {
    pub root_fpu: f32,
    pub children_fpu: f32,
}
impl FPUSettings {
    /// returns the default value of FPUSettings
    pub fn default(self) -> FPUSettings {
        FPUSettings {
            root_fpu: 0.1,
            children_fpu: 0.2,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Copy)]
pub struct CPUCTSettings {
    pub root_c_puct: f32,
    pub children_c_puct: f32,
}
impl CPUCTSettings {
    /// returns the default value of CPUCTSettings
    pub fn default(self) -> CPUCTSettings {
        CPUCTSettings {
            root_c_puct: 0.1,
            children_c_puct: 0.2,
        }
    }
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
    pub moves_left_weight: f32,
    pub moves_left_clip: f32,
    pub moves_left_sharpness: f32,
}

impl MovesLeftSettings {
    pub fn new(moves_left_weight: f32, moves_left_clip: f32, moves_left_sharpness: f32) -> Self {
        Self {
            moves_left_weight,
            moves_left_clip,
            moves_left_sharpness,
        }
    }
}
