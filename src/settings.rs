use crate::mcts_trainer::{TypeRequest, Wdl};

#[derive(Clone, Debug, PartialEq, Copy)]

pub struct SearchSettings {
    pub fpu: f32,
    pub wdl: Option<Wdl>, // if WDL is None then automatically use value
    pub moves_left: Option<usize>,
    pub c_puct: f32,
    pub max_nodes: u128,
    pub alpha: f32,
    pub eps: f32,
    pub search_type: TypeRequest,
}
