use crate::boardmanager::BoardStack;
use crate::mcts_trainer::get_move;
use cozy_chess::*;
use tch::Tensor;

// selfplay code

pub struct DataGen {
    pub iterations: u32, // number of games needed per batch of training data
    pub bs: BoardStack,
}

impl DataGen {
    fn play_game(&mut self) -> (Vec<BoardStack>, Vec<Vec<f32>>, Vec<Vec<usize>>, Option<cozy_chess::Color>) {
        let mut memory: Vec<BoardStack> = Vec::new();
        let mut pi: Vec<Vec<f32>> = Vec::new();
        let mut move_idx: Vec<Vec<usize>> = Vec::new();
        while self.bs.status() == GameStatus::Ongoing {
            let (mv, pi_piece, move_idx_piece) = get_move(&self.bs);
            memory.push(self.bs.clone());
            pi.push(pi_piece);
            move_idx.push(move_idx_piece);
            self.bs.play(mv);
        }
        let outcome: Option<Color> = match self.bs.status() {
            GameStatus::Drawn => None,
            GameStatus::Won => Some(!self.bs.board().side_to_move()),
            GameStatus::Ongoing => panic!(),
        };
        (memory, pi, move_idx, outcome)
    }

    pub fn generate_batch(
        &mut self,
    ) -> (
        Vec<Vec<Tensor>>,
        Vec<Vec<f32>>,
        Vec<Option<Vec<usize>>>,
        Vec<i8>,
    ) {
        let training_data = Vec::new();
        let pi_list = Vec::new();
        let move_idx_list = Vec::new();
        let results_list = Vec::new();
        for _ in 0..self.iterations {
            let mut training_data = Vec::new();
            let mut pi_list = Vec::new();
            let mut move_idx_list = Vec::new();
            let mut results_list = Vec::new();
            let (memory_piece_game, pi, move_idx, z) = self.play_game(); // z is final game result
            training_data.push(memory_piece_game);
            pi_list.push(pi);
            move_idx_list.push(move_idx);
            results_list.push(z); // can't append training data and results tgt :/
        }
        (training_data, pi_list, move_idx_list, results_list)
    }
}
