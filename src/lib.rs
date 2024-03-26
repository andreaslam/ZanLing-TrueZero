pub mod boardmanager;
pub mod dataformat;
pub mod decoder;
pub mod dirichlet;
pub mod dummyreq;
pub mod elo;
pub mod executor;
pub mod fileformat;
pub mod mcts;
pub mod mcts_trainer;
pub mod message_types;
pub mod mvs;
pub mod selfplay;
pub mod settings;
pub mod uci;
#[cfg(test)]

mod tests {
    use cozy_chess::{Board, GameStatus};

    use crate::boardmanager::BoardStack;

    #[test]
    fn test_rep_in_3s() {
        let b = Board::default();
        let mut bs = BoardStack::new(b);
        bs.play("g1f3".parse().unwrap());
        bs.play("g8f6".parse().unwrap());
        bs.play("f3g1".parse().unwrap());
        bs.play("f6g8".parse().unwrap());
        bs.play("g1f3".parse().unwrap());
        bs.play("g8f6".parse().unwrap());
        bs.play("f3g1".parse().unwrap());
        bs.play("f6g8".parse().unwrap());

        assert_eq!(bs.status(), GameStatus::Drawn);
    }

    #[test]
    fn test_count_reps() {
        let b = Board::default();
        let mut bs = BoardStack::new(b);
        assert_eq!(bs.get_reps(), 0);
        bs.play("g1f3".parse().unwrap());
        assert_eq!(bs.get_reps(), 0);
        bs.play("g8f6".parse().unwrap());
        assert_eq!(bs.get_reps(), 0);
    }

    #[test]
    fn test_draw_conditions_bare_kings() {
        let board = Board::from_fen("8/5K2/8/3k4/8/8/8/8 w - - 0 1", false).unwrap();
        let bs = BoardStack::new(board);
        assert_eq!(bs.status(), GameStatus::Drawn);
    }

    #[test]
    fn test_draw_conditions_knight_king() {
        let board = Board::from_fen("8/5K2/4N3/8/2k5/8/8/8 w - - 0 1", false).unwrap();
        let bs = BoardStack::new(board);
        assert_eq!(bs.status(), GameStatus::Drawn);
    }

    #[test]
    fn test_draw_conditions_bishop_king() {
        let board = Board::from_fen("8/5K2/5B2/8/4k3/8/8/8 w - - 0 1", false).unwrap();
        let bs = BoardStack::new(board);
        assert_eq!(bs.status(), GameStatus::Drawn);
    }
}
