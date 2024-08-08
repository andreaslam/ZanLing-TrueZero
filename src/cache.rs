use crate::{dataformat::ZeroValuesAbs, mcts_trainer::Wdl};
use cozy_chess::Move;
#[derive(Eq, Hash, PartialEq, Clone)]
pub struct CacheEntryKey {
    pub hash: u64,
    pub halfmove_clock: u8,
}
