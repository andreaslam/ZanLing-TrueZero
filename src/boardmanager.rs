use cozy_chess::*;
use std::collections::HashMap;
struct BoardStack {
    board: Board,
    move_stack: Vec<u64>,

}

impl BoardStack {

    pub fn compare(&mut self) -> bool {
        let target_hash = self.board.hash();
        let mut is_over = false;
        let is_found = self.move_stack.contains(&target_hash);
        if !is_found {
            self.move_stack.push(target_hash); // add hash into stack
        } else { // repetition found!
            self.move_stack.push(target_hash);
            if self.move_stack.iter().filter(|&x| *x == target_hash).count() >= 3 { // draw by repetition!
                is_over = true;
            }
        }
    is_over
    }
    // get number of repetitions for decoder.rs

    pub fn get_reps(&self) -> u8 {
        let mut count_map: HashMap<u64, usize> = HashMap::new();
        for &num in &self.move_stack {
            *count_map.entry(num).or_insert(0) += 1;
        }
        let max_repetitions = count_map.values().cloned().max().unwrap_or(0);
        max_repetitions as u8
    }

    pub fn play(&mut self, mv:&str) {
        let is_over_draws = self.compare();
        let status = self.board.status();
        let is_board_over = status != GameStatus::Ongoing;
        while !is_board_over | !is_over_draws {
            // convert move to playable ones and play it
            self.board.play(mv.parse().unwrap());
        } 
    }

}
