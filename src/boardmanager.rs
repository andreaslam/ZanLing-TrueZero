use cozy_chess::*;

struct BoardStack {
    board: Board,
    move_stack: Vec<u64>,

}

impl BoardStack {

    fn compare(&self) -> bool {
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

}
