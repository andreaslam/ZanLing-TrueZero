use litchee::model::LichessPerfs;
use rand::seq::SliceRandom;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeControl {
    Classical,
    Rapid,
    Blitz,
    Bullet,
}

impl TimeControl {
    pub fn clock(self) -> (u32, u32) {
        match self {
            Self::Classical => (1_800, 0),
            Self::Rapid => (600, 5),
            Self::Blitz => (180, 2),
            Self::Bullet => (60, 0),
        }
    }

    pub fn has_rating(self, perfs: Option<&LichessPerfs>) -> bool {
        let Some(perfs) = perfs else {
            return false;
        };

        match self {
            Self::Classical => perfs.classical.is_some(),
            Self::Rapid => perfs.rapid.is_some(),
            Self::Blitz => perfs.blitz.is_some(),
            Self::Bullet => perfs.bullet.is_some(),
        }
    }
}

pub const MATCHMAKING_TIME_CONTROLS: [TimeControl; 4] = [
    TimeControl::Classical,
    TimeControl::Rapid,
    TimeControl::Blitz,
    TimeControl::Bullet,
];

pub fn random_time_control() -> TimeControl {
    *MATCHMAKING_TIME_CONTROLS
        .as_slice()
        .choose(&mut rand::thread_rng())
        .expect("matchmaking time controls must not be empty")
}
