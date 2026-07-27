#[derive(Eq, Hash, PartialEq, Clone)]
pub struct CacheEntryKey {
    pub hash: u64,
    pub halfmove_clock: u8,
    /// number of times the current position has already occurred (0, 1, or 2).
    /// Included so a cached evaluation is not reused across a 2-fold repetition
    /// boundary, where `BoardStack::status()` would instead report a draw.
    pub repetitions: u8,
}
