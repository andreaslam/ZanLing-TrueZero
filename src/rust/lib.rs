use std::path::{Path, PathBuf};

pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

pub fn data_path(rel: &str) -> PathBuf {
    workspace_root().join(rel)
}
pub fn data_path_str(rel: &str) -> String {
    data_path(rel).to_string_lossy().into_owned()
}

pub mod boardmanager;
pub mod cache;
pub mod dataformat;
pub mod decoder;
pub mod dirichlet;
pub mod elo;
pub mod executor;
pub mod fileformat;
pub mod lichess_graph;
pub mod lichess_time_control;
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
    use cozy_chess::{Board, Color, GameStatus, Move, Piece, Square};
    use mcts_trainer::{EvalMode, Node, Tree, TypeRequest};
    use settings::{CPUCTSettings, FPUSettings, PSTSettings, SearchSettings};

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

    fn play_uci(board: &mut BoardStack, uci: &str) {
        let from = uci[0..2].parse::<Square>().unwrap();
        let to = uci[2..4].parse::<Square>().unwrap();

        let promotion = uci.chars().nth(4).map(|c| match c {
            'q' => Piece::Queen,
            'r' => Piece::Rook,
            'b' => Piece::Bishop,
            'n' => Piece::Knight,
            _ => panic!("Invalid promotion piece"),
        });

        board.play(Move {
            from,
            to,
            promotion,
        });
    }

    #[test]
    fn en_passant_white_to_move_returns_black_pawn_square() {
        // 1. e2-e4
        //
        // Black to move.
        //  e4.
        let mut board = BoardStack::new(Board::default());

        play_uci(&mut board, "e2e4");

        assert_eq!(board.en_passant(), Some(Square::E4));
    }

    #[test]
    fn en_passant_after_black_double_pawn_push() {
        let mut board = BoardStack::new(Board::default());

        // 1. d2-d4
        play_uci(&mut board, "d2d4");

        // 1... h7-h5
        play_uci(&mut board, "h7h5");

        // 2. d4-d5
        play_uci(&mut board, "d4d5");

        // 2... e7-e5
        play_uci(&mut board, "e7e5");

        assert_eq!(board.en_passant(), Some(Square::E5));
    }
    #[test]
    fn en_passant_d_file() {
        // 1. d2-d4
        //
        // Black to move.
        //  d4.
        let mut board = BoardStack::new(Board::default());

        play_uci(&mut board, "d2d4");

        assert_eq!(board.en_passant(), Some(Square::D4));
    }

    #[test]
    fn en_passant_h_file() {
        // 1. h7-h5
        //
        // Need White to make a legal waiting move first.
        let mut board = BoardStack::new(Board::default());

        play_uci(&mut board, "a2a3");
        play_uci(&mut board, "h7h5");

        assert_eq!(board.en_passant(), Some(Square::H5));
    }

    #[test]
    fn en_passant_disappears_after_non_double_pawn_move() {
        let mut board = BoardStack::new(Board::default());

        play_uci(&mut board, "e2e4");

        assert_eq!(board.en_passant(), Some(Square::E4));

        // Black makes a normal move.
        play_uci(&mut board, "a7a6");

        assert_eq!(board.en_passant(), None);
    }

    #[test]
    fn en_passant_disappears_after_pawn_moves_again() {
        let mut board = BoardStack::new(Board::default());

        play_uci(&mut board, "e2e4");

        assert_eq!(board.en_passant(), Some(Square::E4));

        // Black moves.
        play_uci(&mut board, "a7a6");

        // White moves the pawn again.
        play_uci(&mut board, "e4e5");

        assert_eq!(board.en_passant(), None);
    }

    /// helper function to create dummy SearchSettings
    fn create_search_settings() -> SearchSettings {
        SearchSettings {
            max_nodes: Some(100),
            c_puct: CPUCTSettings {
                root_c_puct: 2.0,
                children_c_puct: 2.0,
            },
            fpu: FPUSettings {
                root_fpu: 0.1,
                children_fpu: 0.1,
            },
            pst: PSTSettings {
                root_pst: 1.75,
                children_pst: 1.5,
            },
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
            settings.fpu.root_fpu,
            node.puct_formula(1, 2.0, Color::White, settings)
        )
    }

    #[test]
    fn test_backup_adds_no_extra_move_at_evaluated_node() {
        let board_stack = create_board_stack();
        let settings = create_search_settings();
        let mut tree = Tree::new(board_stack, settings);
        tree.nodes[0].net_evaluation.moves_left = 0.0;

        tree.backpropagate(0);

        assert_eq!(tree.nodes[0].total_evaluation.moves_left, 0.0);
    }

    #[test]
    fn matchmaking_pool_contains_expected_controls() {
        use lichess_time_control::{TimeControl, MATCHMAKING_TIME_CONTROLS};

        assert_eq!(
            MATCHMAKING_TIME_CONTROLS,
            [
                TimeControl::Classical,
                TimeControl::Rapid,
                TimeControl::Blitz,
                TimeControl::Bullet,
            ]
        );
        assert_eq!(TimeControl::Classical.clock(), (1_800, 0));
        assert_eq!(TimeControl::Rapid.clock(), (600, 5));
        assert_eq!(TimeControl::Blitz.clock(), (180, 2));
        assert_eq!(TimeControl::Bullet.clock(), (60, 0));
        assert!(!TimeControl::Rapid.has_rating(None));
    }

    fn temporary_database_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "truezero-player-graph-{}-{:?}.sqlite3",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn migrates_legacy_graph_and_keeps_normalized_relationships() {
        use lichess_graph::{graph_insert_edge, graph_insert_player, open_player_graph_database};
        use rusqlite::Connection;

        let path = temporary_database_path();
        let _ = std::fs::remove_file(&path);
        let path_string = path.to_string_lossy().into_owned();

        {
            let legacy = Connection::open(&path_string).unwrap();
            legacy
                .execute_batch(
                    r#"
                    CREATE TABLE players (
                        username TEXT PRIMARY KEY,
                        is_bot INTEGER,
                        first_seen_at INTEGER NOT NULL,
                        last_seen_online_at INTEGER,
                        last_expanded_at INTEGER,
                        discovered_from TEXT
                    );
                    CREATE TABLE player_edges (
                        username TEXT NOT NULL,
                        opponent TEXT NOT NULL,
                        last_seen_at INTEGER NOT NULL,
                        PRIMARY KEY (username, opponent)
                    );
                    CREATE TABLE human_challenges (
                        username TEXT PRIMARY KEY,
                        last_challenged_at INTEGER NOT NULL
                    );
                    INSERT INTO players VALUES
                        ('HumanSeed', NULL, 1, 2, 3, 'online_human'),
                        ('Opponent', 0, 4, NULL, NULL, 'game: HumanSeed');
                    INSERT INTO player_edges VALUES
                        ('HumanSeed', 'Opponent', 5);
                    INSERT INTO human_challenges VALUES
                        ('HumanSeed', 6);
                    "#,
                )
                .unwrap();
        }

        let db = open_player_graph_database(&path_string).unwrap();
        let player_count: i64 = db
            .query_row("SELECT COUNT(*) FROM players", [], |row| row.get(0))
            .unwrap();
        let edge_count: i64 = db
            .query_row("SELECT COUNT(*) FROM player_edges", [], |row| row.get(0))
            .unwrap();
        let challenge_count: i64 = db
            .query_row("SELECT COUNT(*) FROM human_challenges", [], |row| {
                row.get(0)
            })
            .unwrap();

        assert_eq!(player_count, 2);
        assert_eq!(edge_count, 1);
        assert_eq!(challenge_count, 1);

        graph_insert_player(&db, "HumanSeed", None, "online_human").unwrap();
        graph_insert_edge(&db, "HumanSeed", "Opponent").unwrap();

        let discovery_count: i64 = db
            .query_row("SELECT COUNT(*) FROM player_discoveries", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(discovery_count, 2);

        drop(db);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn prunes_old_unplayed_players_before_connected_players() {
        use lichess_graph::{
            graph_insert_edge, graph_prune_players, open_player_graph_database, MAX_GRAPH_PLAYERS,
        };
        use rusqlite::params;

        let path = temporary_database_path();
        let _ = std::fs::remove_file(&path);
        let path_string = path.to_string_lossy().into_owned();
        let db = open_player_graph_database(&path_string).unwrap();

        for username in ["connected", "oldest", "played-peer"] {
            db.execute(
                "INSERT INTO players
                 (username, first_seen_at, last_seen_online_at)
                 VALUES (?1, 1, 1)",
                params![username],
            )
            .unwrap();
        }
        for index in 0..(MAX_GRAPH_PLAYERS - 2) {
            db.execute(
                "INSERT INTO players
                 (username, first_seen_at, last_seen_online_at)
                 VALUES (?1, 2, 2)",
                params![format!("player-{index}")],
            )
            .unwrap();
        }

        graph_insert_edge(&db, "connected", "played-peer").unwrap();
        assert_eq!(graph_prune_players(&db).unwrap(), 1);

        let oldest_exists: bool = db
            .query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM players
                    WHERE username = 'oldest'
                )",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let connected_exists: bool = db
            .query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM players
                    WHERE username = 'connected'
                )",
                [],
                |row| row.get(0),
            )
            .unwrap();

        assert!(!oldest_exists);
        assert!(connected_exists);

        drop(db);
        std::fs::remove_file(path).unwrap();
    }
}
