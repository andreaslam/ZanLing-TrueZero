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

    // get the number of previous occurrences of the current position

    pub fn get_reps(&self) -> usize {
        let target = self.board.hash();

        self.move_stack
            .iter()
            .filter(|&hash| *hash == target)
            .count()
    }

    // return the square occupied by the pawn that just double-stepped
    // only return a square when an en passant capture is currently possible
    //
    // e2-e4 -> Some(e4)
    // e7-e5 -> Some(e5)

    pub fn en_passant(&self) -> Option<Square> {
        let file =
            self.board.en_passant()?;

        let rank =
            match self.board.side_to_move() {
                // black just moved e7-e5, so the pawn is on rank 5
                Color::White => Rank::Fifth,

                // white just moved e2-e4, so the pawn is on rank 4
                Color::Black => Rank::Fourth,
            };

        Some(Square::new(file, rank))
    }

    // play a move

    pub fn play(&mut self, mv: Move) {
        assert!(
            self.status == GameStatus::Ongoing,
            "Cannot play a move after the game has ended"
        );

        self.move_stack
            .push(self.board.hash());

        self.board.play(mv);

        let is_all_gone =
            self.board.occupied().len() == 2;

        let is_sure_draw =
            self.board.occupied().len() <= 3
                && (
                    self.board.pieces(Piece::Bishop).len() == 1
                    || self.board.pieces(Piece::Knight).len() == 1
                );

        self.status =
            if self.get_reps() == 2
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
        self.status() != GameStatus::Ongoing
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn status(&self) -> GameStatus {
        let is_all_gone =
            self.board.occupied().len() == 2;

        let is_sure_draw =
            self.board.occupied().len() <= 3
                && (
                    self.board.pieces(Piece::Bishop).len() == 1
                    || self.board.pieces(Piece::Knight).len() == 1
                );

        if self.get_reps() == 2
            || self.board.halfmove_clock() == 100
            || is_all_gone
            || is_sure_draw
        {
            GameStatus::Drawn
        } else {
            self.board.status()
        }
    }
}
