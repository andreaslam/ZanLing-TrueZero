use crate::boardmanager::BoardStack;
use crate::mcts_trainer::get_move;
use cozy_chess::{Board, Color, GameStatus, Move};
use tch::Tensor;

// selfplay code
#[derive(PartialEq, Clone, Debug)]
pub struct DataGen {
    pub iterations: u32, // number of games needed per batch of training data
}

impl DataGen {
    fn play_game(
        &mut self,
    ) -> (
        Vec<BoardStack>,
        Vec<Vec<f32>>,
        Vec<Vec<usize>>,
        Option<cozy_chess::Color>,
    ) {
        let mut memory: Vec<BoardStack> = Vec::new();
        let mut pi: Vec<Vec<f32>> = Vec::new();
        let mut move_idx: Vec<Vec<usize>> = Vec::new();
        let mut counter = 0;
        let mut tau = 1;
        let board = Board::default();
        let mut bs = BoardStack::new(board);

        while bs.status() == GameStatus::Ongoing {
            if counter == 30 {
                tau = 0
            }
            let (mv, pi_piece, move_idx_piece) = get_move(bs.clone());
            memory.push(bs.clone());
            pi.push(pi_piece);
            move_idx.push(move_idx_piece.unwrap());
            bs.play(mv);
            counter += 1;
        }
        let outcome: Option<Color> = match bs.status() {
            GameStatus::Drawn => None,
            GameStatus::Won => Some(!bs.board().side_to_move()),
            GameStatus::Ongoing => panic!("Game is still ongoing!"),
        };
        (memory, pi, move_idx, outcome)
    }

    pub fn generate_batch(
        &mut self,
    ) -> (
        Vec<Vec<BoardStack>>,
        Vec<Vec<Vec<f32>>>,
        Vec<Vec<Vec<usize>>>,
        Vec<Option<Color>>,
    ) {
        let mut pi_list: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut move_idx_list = Vec::new();
        let mut results_list: Vec<Option<Color>> = Vec::new(); // None for draw
        let mut boardstack_list: Vec<Vec<BoardStack>> = Vec::new();
        for _ in 0..self.iterations {
            let (memory, pi, move_idx, outcome) = self.play_game(); // z is final game result
                                                                    // memory in the form of boardstack
            println!("{:?}", outcome);
            boardstack_list.push(memory);
            pi_list.push(pi);
            move_idx_list.push(move_idx);
            results_list.push(outcome); // raw outcome of who won, not accounting for POV yet
        }
        (boardstack_list, pi_list, move_idx_list, results_list)
    }
}
