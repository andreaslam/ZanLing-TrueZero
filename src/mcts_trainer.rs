use cozy_chess::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;
use tch::*;
use crate::decoder::eval_board;
use crate::decoder::convert_board;
use crate::selfplay::DataGen;
use std::fmt;
use crate::dirichlet::StableDirichlet;
// define tree and node classes
#[derive(PartialEq,Clone, Debug)] // maybe display and debug as helper funcs to check impl
struct Node {
    parent: Option<Box<Node>>,
    children: Vec<Node>,
    policy: f32,
    visits: i32,
    eval_score: f32,
    board: Board,
    total_action_value: f32,
    move_name: Option<cozy_chess::Move>,
    move_idx: Option<Vec<usize>>,
}

impl Node {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let u: f32;
        let puct: f32;

        match &self.parent {
            Some(parent) => {
                u = 2.0 * self.policy * ((parent.visits - 1) as f32).sqrt() / (1.0 + self.visits as f32);
                puct = self.puct_formula(parent.visits);
            }
            None => {
                u = f32::NAN;
                puct = f32::NAN;
            }
        }
        let mv_n: String;
        match &self.move_name {
            Some(move_name) => {
                mv_n = format!("{}",move_name);
            }
            None => {
                mv_n = "Null".to_string();
            }
        }
        write!(f, "Node(action= {}, V= {}, N={}, W={}, P={}, Q={}, U={}, PUCT={}, len_children={})", mv_n, self.eval_score, self.visits, self.total_action_value, self.policy, self.get_q_val(), u, puct, self.children.len())
    }

    // fn is_leaf(&self) -> bool {
    //     self.visits == 0
    // }

    fn get_q_val(&self) -> f32 {
        let fpu = 0.0; // First Player Urgency
        if self.visits > 0 {
            self.total_action_value / (self.visits as f32)
        } else {
            fpu
        }
    }

    fn puct_formula(&self, parent_visits: i32) -> f32 {
        let c_puct = 2.0; // "constant determining the level of exploration"
        let u = c_puct * self.policy * ((parent_visits - 1) as f32).sqrt() / (1.0 + self.visits as f32);
        let q = self.get_q_val();
        -q + u
    }

    fn is_terminal(&self,board:&Board) -> bool {
        let status = board.status();
        status != GameStatus::Ongoing // returns true if game is over (not ongoing)
    }

    fn new(board: Board, policy: f32, parent: Option<Box<Node>>, move_name: Option<cozy_chess::Move>) -> Node {
        Node {
            parent,
            children: vec![],
            policy,
            visits: 0,
            eval_score: 0.0,
            board,
            total_action_value: 0.0,
            move_name,
            move_idx: None,
        }
    }

    fn eval_and_expand(&mut self, board: &Board, bs:&DataGen) -> Vec<usize> {
        
        // return the indexes as Vec<usize>

        let (value, policy, idx_li) = eval_board(board, bs);

        self.eval_score = value;
        for (p, pol) in &policy {
            // get fen
            // set b as fen
            let fenstr = format!("{}",board);
            let mut b = Board::from_fen(&fenstr, false).expect("Error");
            b.play(*p);
            
            let child = Node::new(b, *pol, Some(Box::new(self.clone())), Some(*p));
            self.children.push(child);
        }
        idx_li
    }
}
#[derive(PartialEq)]
struct Tree {
    board: Board, 
    root_node: Node,
}

impl Tree {
    fn new(board: Board) -> Tree {
        let root_node = Node::new(board.clone(), 0.0, None, None);
        Tree { board, root_node }
    }

    fn select(&self) -> &Node {
        let mut curr = &self.root_node;
        while !curr.children.is_empty() {
            curr = curr.children.iter().max_by(|a, b| {
                a.puct_formula(curr.visits)
                    .partial_cmp(&b.puct_formula(curr.visits))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }).expect("Error");
        }
        curr
    }

    fn backpropagate(&mut self, node: &mut Node) {
        let mut n = node.eval_score;
        let mut curr = node;
        while let Some(mut parent) = curr.parent.take() {
            parent.visits += 1;
            parent.total_action_value += n;
            curr.parent = Some(parent);
            curr = curr.parent.as_mut().expect("Error");
            n = -n;
        }
        println!("{:?}", self.root_node);
    }

    fn step(&mut self, bs:&DataGen) {
        const EPS:f32  = 0.3; // 0.3 for chess
        let mut selected_node = self.select().clone();
        if !selected_node.is_terminal(&self.board) {
            let idx_li = selected_node.eval_and_expand(&self.board, bs);
            self.root_node.move_idx = Some(idx_li);  
            if selected_node == self.root_node {
                let mut legal_moves = Vec::new();
                selected_node.board.generate_moves(|moves| {
                // Unpack dense move set into move list
                legal_moves.extend(moves);
                false
                });
                // add Dirichlet noise
                let mut std_rng = StdRng::from_entropy();
                let distr = StableDirichlet::new(0.3,legal_moves.len()).expect("wrong params");
                let sample = std_rng.sample(distr);
                for i in 0..self.root_node.children.len() {
                    let child = &mut self.root_node.children[i];
                    child.policy = (1.0-EPS) * child.policy + (EPS * sample[i]);
                }
                }
            }
            self.backpropagate(&mut selected_node);
        }
}


pub const MAX_NODES: i32 = 10;

pub fn get_move(board:&Board, bs:&DataGen) -> (Move, Vec<f32>, Vec<f32>, Option<Vec<usize>>){ //equiv to move() in mcts_trainer.py
    let mut tree = Tree::new(board.clone()); // change if needed, maybe take a &mut of it

    while tree.root_node.visits < MAX_NODES {
        tree.step(bs);
    }

    
    let best_move_node = tree.root_node.children.iter().max_by_key(|&n| n.visits).expect("Error");
    let best_move = best_move_node.move_name; 
    let mut total_visits_list = Vec::new();

    for child in &tree.root_node.children {
        total_visits_list.push(child.visits);
    } // hmm is it possible to change to .push()
    let total_visits: i32 = total_visits_list.iter().sum();

    let mut pi: Vec<f32> = Vec::new();

    for &t in &total_visits_list {
        let prob = t as f32/ total_visits as f32;
        pi.push(prob);
    }

    let rb_input = convert_board(&tree.root_node.board, bs);
    let memory_piece = rb_input;
    let memory_piece: Result<Vec<f32>> = Vec::try_from(memory_piece);
    let m: Vec<f32> = vec![];
    match memory_piece {
        Ok(memory_piece) => {
            memory_piece; 
        }
        Err(err_msg) => {
            println!("Error: {}", err_msg);
        }
    }

    let memory_piece = m;
    (best_move.unwrap(), memory_piece, pi, tree.root_node.move_idx)
}
