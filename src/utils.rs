use crate::debug_print;
use std::time::{SystemTime, UNIX_EPOCH};
pub struct TimeStampDebugger {
    epoch_seconds_start: u128,
}

impl TimeStampDebugger {
    pub fn create_debug() -> Self {
        let now_start = SystemTime::now();
        let since_epoch = now_start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let epoch_seconds_start = since_epoch.as_nanos();
        TimeStampDebugger {
            epoch_seconds_start,
        }
    }

    pub fn reset(&mut self) {
        if cfg!(debug_assertions) {
            let now_start = SystemTime::now();
            let since_epoch = now_start
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards");
            self.epoch_seconds_start = since_epoch.as_nanos();
        }
    }

    pub fn record(&self, message: &str, thread_name: &str) {
        if cfg!(debug_assertions) {
            let now_end = SystemTime::now();
            let since_epoch_end = now_end
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards");
            let epoch_seconds_end = since_epoch_end.as_nanos();
            let _msg = format!(
                "{} {} {} {}",
                self.epoch_seconds_start, epoch_seconds_end, thread_name, message
            );
            debug_print!("{}", &_msg);
        }
    }
}
#[cfg(debug_assertions)]
#[macro_export]
macro_rules! debug_print {
    ($($x:tt)*) => { println!($($x)*) }
}
#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! debug_print {
    ($($x:tt)*) => {{}};
}
