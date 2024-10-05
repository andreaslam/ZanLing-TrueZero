pub mod boardmanager;
pub mod cache;
pub mod dataformat;
pub mod decoder;
pub mod dirichlet;
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
pub mod utils;

#[cfg(test)]

mod tests {
    use super::*;

    use boardmanager::BoardStack;
    use cozy_chess::{Board, Color, GameStatus};
    use mcts_trainer::{EvalMode, Node, Tree, TypeRequest};
    use settings::{CPUCTSettings, FPUSettings, SearchSettings};

    extern crate flume;

    /// helper function to create a dummy BoardStack

    fn create_board_stack() -> BoardStack {
        let board = Board::default();
        BoardStack::new(board)
    }
    /// test threefold repetition
    #[test]
    fn test_rep_in_3s() {
        let mut bs = create_board_stack();
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
    /// test repetition counter
    #[test]
    fn test_count_reps() {
        let mut bs = create_board_stack();
        assert_eq!(bs.get_reps(), 0);
        bs.play("g1f3".parse().unwrap());
        assert_eq!(bs.get_reps(), 0);
        bs.play("g8f6".parse().unwrap());
        assert_eq!(bs.get_reps(), 0);
    }
    /// test draw_conditions for bare kings
    #[test]
    fn test_draw_conditions_bare_kings() {
        let board = Board::from_fen("8/5K2/8/3k4/8/8/8/8 w - - 0 1", false).unwrap();
        let bs = BoardStack::new(board);
        assert_eq!(bs.status(), GameStatus::Drawn);
    }
    /// test draw_conditions for knight and king only
    #[test]
    fn test_draw_conditions_knight_king() {
        let board = Board::from_fen("8/5K2/4N3/8/2k5/8/8/8 w - - 0 1", false).unwrap();
        let bs = BoardStack::new(board);
        assert_eq!(bs.status(), GameStatus::Drawn);
    }
    /// test draw_condition for bishop and king only
    #[test]
    fn test_draw_conditions_bishop_king() {
        let board = Board::from_fen("8/5K2/5B2/8/4k3/8/8/8 w - - 0 1", false).unwrap();
        let bs = BoardStack::new(board);
        assert_eq!(bs.status(), GameStatus::Drawn);
    }

    /// helper function to create dummy SearchSettings
    fn create_search_settings() -> SearchSettings {
        SearchSettings {
            max_nodes: Some(100),
            c_puct: CPUCTSettings {
                root_c_puct: Some(2.0),
                children_c_puct: Some(2.0),
            },
            fpu: FPUSettings {
                root_fpu: Some(0.1),
                children_fpu: Some(0.1),
            },
            pst: 1.0,
            eps: 0.25,
            alpha: 0.03,
            search_type: TypeRequest::UCISearch,
            wdl: EvalMode::Wdl,
            moves_left: None,
            batch_size: 1,
        }
    }

    // test tree initialisation
    #[test]
    fn test_tree_initialisation() {
        let board_stack = create_board_stack();
        let settings = create_search_settings();
        let tree = Tree::new(board_stack, settings);
        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.nodes[0].visits, 0);
    }

    /// test node initialisation
    #[test]
    fn test_node_initialisation() {
        let node = Node::new(0.5, None, None);
        assert_eq!(node.policy, 0.5);
        assert!(node.parent.is_none());
        assert!(node.mv.is_none());
        assert_eq!(node.visits, 0);
        assert!(node.net_evaluation.value.is_nan());
    }

    /// test node selection
    #[test]
    fn test_node_selection() {
        let board_stack = create_board_stack();
        let settings = create_search_settings();
        let mut tree = Tree::new(board_stack, settings);
        let (selected_node, _, _) = tree.select();
        assert_eq!(selected_node, 0);
    }

    /// test PUCT with non-zero FPU, where the initial visit count is 0
    #[test]

    fn test_puct_with_nonzero_fpu() {
        let settings = create_search_settings();
        let node = Node::new(0.5, None, None);
        assert_eq!(
            settings.fpu.root_fpu.unwrap(),
            node.puct_formula(1, 2.0, Color::White, settings)
        )
    }
}
