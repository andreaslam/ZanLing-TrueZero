use cozy_chess::*;
use tch::*;
use crate::decoder::eval_board;
use crate::decoder::convert_board;
use crate::BoardStack;

// define tree and node classes
#[derive(PartialEq, Clone)] // maybe display and debug as helper funcs to check impl
struct Node {
    parent: Option<Box<Node>>,
    children: Vec<Node>,
    policy: f32,
    visits: i32,
    eval_score: f32,
    board: Board,
    total_action_value: f32,
    move_name: String,
    move_idx: Option<Vec<usize>>,
}

impl Node {
    fn is_leaf(&self) -> bool {
        self.visits == 0
    }

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

    fn is_terminal(&self,board:Board) -> bool {
        let status = board.status();
        status != GameStatus::Ongoing // returns true if game is over (not ongoing)
    }

    fn new(board: Board, policy: f32, parent: Option<Box<Node>>, move_name: String) -> Node {
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

    fn eval_and_expand(&mut self, board: &Board) -> Vec<usize> {
        
        // return the indexes as Vec<usize>

        let (value, policy, idx_li) = eval_board(board);

        self.eval_score = value;
        for (p, pol) in &policy {
            let mut x = board.clone();
            x.play(p.parse().unwrap());
            
            let mut child = Node::new(x, *pol, Some(Box::new(self.clone())), p.clone());
            self.children.push(child);
            child.board = x;
        }
        idx_li
    }
}

struct Tree {
    board: Board, 
    root_node: Node,
}

impl Tree {
    fn new(board: Board) -> Tree {
        let root_node = Node::new(board.clone(), 0.0, None, String::new());
        Tree { board, root_node }
    }

    fn select(&self) -> &Node {
        let mut curr = &self.root_node;
        while !curr.children.is_empty() {
            curr = curr.children.iter().max_by(|a, b| {
                a.puct_formula(curr.visits)
                    .partial_cmp(&b.puct_formula(curr.visits))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }).unwrap();
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
            curr = curr.parent.as_mut().unwrap();
            n = -n;
        }
    }

    fn step(&mut self) {
        let eps = 0.3; // 0.3 for chess
        let mut selected_node = self.select().clone();
        if !selected_node.is_terminal(self.board) {
            let idx_li = selected_node.eval_and_expand(&self.board);
            self.root_node.move_idx = Some(idx_li);
            if selected_node == self.root_node {
                
            }
        }
        self.backpropagate(&mut selected_node);
    }
}

// skip read txt

pub const MAX_NODES: i32 = 10000;

pub fn get_move(board:Board) -> (String, Tensor, Vec<f32>, Option<Vec<usize>>){ //equiv to move() in mcts_trainer.py
    let mut tree = Tree::new(board); // change if needed, maybe take a &mut of it

    while tree.root_node.visits < MAX_NODES {
        tree.step();
    }

    
    let best_move_node = tree.root_node.children.iter().max_by_key(|&n| n.visits).unwrap();
    let best_move_node = *best_move_node;
    let best_move:String = best_move_node.move_name; 
    let mut total_visits_list = Vec::new();

    for child in tree.root_node.children {
        total_visits_list.push(child.visits);
    } // hmm is it possible to change to .push()
    let total_visits: i32 = total_visits_list.iter().sum();

    let mut pi: Vec<f32> = Vec::new();

    for &t in &total_visits_list {
        let prob = t as f32/ total_visits as f32;
        pi.push(prob);
    }

    let rb_input = convert_board(&tree.root_node.board);
    let memory_piece = rb_input;
    (best_move, memory_piece, pi, tree.root_node.move_idx)
}
