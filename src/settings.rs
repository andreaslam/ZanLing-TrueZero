use crate::mcts_trainer::Wdl;

pub struct SearchSettings {
    fpu: f32,
    wdl: Wdl,
    c_puct: f32,
    max_nodes: u32,
    alpha: f32,
    eps: f32,
}
