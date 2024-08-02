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

    /// get number of repetitions for decoder.rs

    pub fn get_reps(&self) -> usize {
        // reps only for the current position, not the global maximum of repetitions recorded
        let target = self.board.hash();
        (&self.move_stack).iter().filter(|&x| *x == target).count()
    }

    /// play function to be called instead of calling directly on the cozy-chess board
    pub fn play(&mut self, mv: Move) {
        assert!(self.status == GameStatus::Ongoing); // check if prev board is valid (can play a move)
        self.move_stack.push(self.board.hash());
        self.board.play(mv);
        let is_all_gone = self.board.occupied().len() == 2;
        let is_sure_draw = self.board.occupied().len() <= 3
            && (self.board.pieces(Piece::Bishop).len() == 1
                || self.board.pieces(Piece::Knight).len() == 1); // (K or K+N or K+B) vs (K or K+N or K+B)
        self.status = if self.get_reps() == 2
            || self.board.halfmove_clock() == 100
            || is_all_gone
            || is_sure_draw
        {
            GameStatus::Drawn
        } else {
            self.board.status()
        };
    }

    pub fn is_terminal(&self) -> bool {
        let status = self.status();
        status != GameStatus::Ongoing // returns true if game is over (not ongoing)
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn status(&self) -> GameStatus {
        let is_all_gone = self.board.occupied().len() == 2;
        let is_sure_draw = self.board.occupied().len() <= 3
            && (self.board.pieces(Piece::Bishop).len() == 1
                || self.board.pieces(Piece::Knight).len() == 1); // (K or K+N or K+B) vs (K or K+N or K+B)
        if self.get_reps() == 2 || self.board.halfmove_clock() == 100 || is_all_gone || is_sure_draw
        {
            GameStatus::Drawn
        } else {
            self.board.status()
        }
    }
}
