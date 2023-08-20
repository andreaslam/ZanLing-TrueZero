use crate::mcts_trainer::get_move;
use crate::boardmanager::BoardStack;
use cozy_chess::*;
// use tch::Tensor;

// selfplay code

pub struct DataGen {
    pub iterations: u32, // number of games needed per batch of training data
    pub stack_manager: BoardStack, 
}

impl DataGen {
    fn play_game(&mut self) -> (Vec<Vec<f32>>, Vec<f32>, Option<Vec<usize>>, i8) {
        
        let mut memory: Vec<Vec<f32>> = Vec::new();
        let mut pi: Vec<f32> = Vec::new();
        let mut move_idx:Option<Vec<usize>> = None;
        let mut is_over = self.stack_manager.compare();
        while !is_over {
            memory = Vec::new();
            let mv: cozy_chess::Move;
            let memory_piece: Vec<f32>;
            (mv, memory_piece, pi, move_idx) = get_move(&self.stack_manager.board, &self);// memory_piece is the input data used for training
            println!("{:?}", mv);
            memory.push(memory_piece);
            self.stack_manager.play(&mv);
            is_over = self.stack_manager.compare();
        }
        let o = self.stack_manager.board.side_to_move();
        let outcome: i8;
        match o {
            Color::White => {
                outcome = 1;
            }
            Color::Black => {
                outcome = -1;
            }
        }
        (memory,pi,move_idx, outcome)
    }

    pub fn generate_batch(&mut self) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>, Vec<Option<Vec<usize>>>, Vec<i8>) {
        let training_data = Vec::new();
        let pi_list = Vec::new();
        let move_idx_list = Vec::new();
        let results_list = Vec::new();
        for _ in 0..self.iterations {
            let mut training_data = Vec::new();
            let mut pi_list = Vec::new();
            let mut move_idx_list = Vec::new();
            let mut results_list = Vec::new();
            let (memory_piece_game,pi,move_idx, z) = self.play_game(); // z is final game result
            training_data.push(memory_piece_game);
            pi_list.push(pi);
            move_idx_list.push(move_idx);
            results_list.push(z); // can't append training data and results tgt :/
        }
        (training_data,pi_list,move_idx_list,results_list)
    }

}
