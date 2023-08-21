use cozy_chess::*;

#[derive(PartialEq, Clone, Debug)]
pub struct BoardStack {
    board: Board,
    move_stack: Vec<u64>,
    status: GameStatus,
}

impl BoardStack {
    pub fn new(board: Board) -> Self {
        Self {
            status: board.status(),
            board,
            move_stack: Vec::new(),
        }
    }

    // get number of repetitions for decoder.rs
    // remember to push current position BEFORE calling on get_reps
    pub fn get_reps(&self) -> usize {
        // reps only for the current position, not the global maximum of repetitions recorded
        let target = self.move_stack.last().unwrap();
        (&self.move_stack).iter().filter(|&x| x == target).count() - 1
        
    }

    // play function to be called in selfplay.rs
    pub fn play(&mut self, mv: Move) {
        assert!(self.status == GameStatus::Ongoing); // check if prev board is valid (can play a move)
        self.board.play(mv);
        self.move_stack.push(self.board.hash());
        self.status = if self.get_reps() == 2 {
            GameStatus::Drawn
        } else {
            self.board.status()
        };
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn status(&self) -> GameStatus {
        self.status
    }
}
