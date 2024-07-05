use crate::mcts_trainer::Wdl;
use cozy_chess::Move;
#[derive(Eq, Hash, PartialEq, Clone)]
pub struct CacheEntryKey {
    pub hash: u64,
    pub halfmove_clock: u8,
}

#[derive(PartialEq, Clone)]
pub struct CacheEntryValue {
    pub eval_score: f32,
    pub policy: Vec<f32>,
    pub moves_left: f32,
    pub wdl: Wdl,
    pub mv: Option<Move>,
}
