use cozy_chess::*;

pub struct BoardStack {
    pub board: Board,
    pub move_stack: Vec<u64>,

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
            if self.move_stack.iter().filter(|&x| *x == target_hash).count() == 3 { // draw by repetition!
                is_over = true;
            }
        }
    is_over
    }

    fn count_occurrences(&self,vec: &Vec<u64>, target: u64) -> usize {
        vec.iter().filter(|&&x| x == target).count()
    }
    // get number of repetitions for decoder.rs
    // remember to push current position BEFORE calling on get_reps
    pub fn get_reps(&self) -> u8 {
        // reps only for the current position, not the global maximum of repetitions recorded
        let max_repetitions = self.count_occurrences(&self.move_stack, self.move_stack[self.move_stack.len()-1]);
        max_repetitions as u8
    }

    // play function to be called in selfplay.rs
    pub fn play(&mut self, mv:&Move) {
        let is_over_draws = self.compare();
        let status = self.board.status();
        let is_board_over = status != GameStatus::Ongoing;
        if !is_board_over | !is_over_draws {
            // convert move to playable ones and play it
            self.board.play(*mv);
        } 
    }

}
