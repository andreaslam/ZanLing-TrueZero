use crate::boardmanager::BoardStack;
use crate::decoder::eval_board;
use crate::dirichlet::StableDirichlet;
use cozy_chess::*;
use rand::SeedableRng;
use rand::{rngs::StdRng, Rng};
use std::{fmt, thread};

// define tree and node classes
#[derive(PartialEq, Clone, Debug)] // maybe display and debug as helper funcs to check impl

// struct Packet {
//     visits: i32,
//     top: Option<Move>,
// }

struct Node {
    parent: Option<usize>,
    children: Vec<usize>,
    policy: f32,
    visits: i32,
    eval_score: f32,
    board: BoardStack,
    total_action_value: f32,
    move_name: Option<cozy_chess::Move>,
    move_idx: Option<Vec<usize>>,
}

impl Node {
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
        let u =
            c_puct * self.policy * ((parent_visits - 1) as f32).sqrt() / (1.0 + self.visits as f32);
        let q = self.get_q_val();
        -q + u
    }

    fn is_terminal(&self, board: &BoardStack) -> bool {
        let status = board.status();
        status != GameStatus::Ongoing // returns true if game is over (not ongoing)
    }

    fn new(
        board: BoardStack,
        policy: f32,
        parent: Option<usize>,
        move_name: Option<cozy_chess::Move>,
    ) -> Node {
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
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // let u: f32;
        // let puct: f32;

        // match &self.parent {
        //     Some(parent) => {
        //         u = 2.0 * self.policy * ((parent.visits - 1) as f32).sqrt()
        //             / (1.0 + self.visits as f32);
        //         puct = self.puct_formula(parent.visits);
        //     }
        //     None => {
        //         u = f32::NAN;
        //         puct = f32::NAN;
        //     }
        // }
        let mv_n: String;
        match &self.move_name {
            Some(move_name) => {
                mv_n = format!("{}", move_name);
            }
            None => {
                mv_n = "Null".to_string();
            }
        }

        write!(
            f,
            "Node(action= {}, V= {}, N={}, W={}, P={}, Q={}, len_children={})",
            mv_n,
            self.eval_score,
            self.visits,
            self.total_action_value,
            self.policy,
            self.get_q_val(),
            // u,
            // puct,
            self.children.len()
        )
    }
}

#[derive(PartialEq, Debug)]
struct Tree {
    board: BoardStack,
    nodes: Vec<Node>,
}

impl Tree {
    //     fn layer_p(&self, depth: u8, max_tree_print_depth: u8,index:usize) {
    //         let indent = "    ".repeat(depth as usize);
    //         if depth <= max_tree_print_depth {
    //             println!("{}{}", indent, self.nodes[index]);

    //             for (index, _) in self.nodes.iter().enumerate() {
    //                 if index > 0 {
    //                     self.layer_p(depth + 1, max_tree_print_depth, index);
    //                 }
    //             }
    //         }
    // }

    fn new(board: BoardStack) -> Tree {
        let root_node = Node::new(board.clone(), 0.0, None, None);
        let mut container: Vec<Node> = Vec::new();
        container.push(root_node);
        Tree {
            board,
            nodes: container,
        }
    }

    fn eval_and_expand(
        &mut self,
        selected_node_idx: usize,
        bs: &BoardStack,
    ) -> (usize, Vec<usize>) {
        let (value, policy, idx_li) = eval_board(&bs);
        let fenstr = format!("{}", bs.board());
        println!("    board FEN: {}", fenstr);
        println!("    ran NN:");
        println!("        V={}, \n        policy={:?}", &value, &policy);
        self.nodes[selected_node_idx].eval_score = value;
        let ct = self.nodes[selected_node_idx].children.len();
        let mut counter = self.nodes.len();

        for (p, pol) in &policy {
            // get fen
            let mut b = bs.clone();
            b.play(*p);
            let child = Node::new(b, *pol, Some(selected_node_idx), Some(*p));
            self.nodes.push(child); // push child to the tree Vec<Node>
            self.nodes[selected_node_idx].children.push(counter + ct); // push numbers
            counter += 1
        }
        // println!("        children:");
        (selected_node_idx, idx_li)
    }

    fn select(&mut self) -> usize {
        let mut curr: usize = 0;
        println!("    selection:");
        loop {
            let curr_node = &self.nodes[curr];

            if curr_node.children.is_empty() {
                break;
            }
            // get number of visits for children
            // step 1, use curr.children vec<usize> to index tree.nodes (get a slice)
            let children = &curr_node.children;
            // step 2, iterate over them and get the child with most visits
            curr = *children
                .iter()
                .max_by(|a, b| {
                    self.nodes[**a]
                        .puct_formula(curr_node.visits)
                        .partial_cmp(&self.nodes[**b].puct_formula(curr_node.visits))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
        }
        // println!("    {}", curr);
        println!("        children:");

        curr
    }

    fn backpropagate(&mut self, node: usize) {
        println!("    backup:");
        let mut n = self.nodes[node].eval_score;
        let mut curr: Option<usize> = Some(node); // used to index parent
        while let Some(current) = curr {
            self.nodes[current].visits += 1;
            self.nodes[current].total_action_value += n;
            curr = self.nodes[current].parent;
            println!("        updated node to {}", self.nodes[current]);
            n = -n;
        }
    }
    fn step(&mut self, bs: BoardStack) {
        println!("root node: {}", &self.nodes[0]);
        const EPS: f32 = 0.3; // 0.3 for chess
                              // self.display_full_tree();
        let selected_node: usize;
        selected_node = self.select();
        // println!("{:p},{:p}", &selected_node, &self);
        let mut selected_node = selected_node;
        let idx_li: Vec<usize>;

        if !self.nodes[selected_node].is_terminal(&bs) {
            let bc = self.board.clone();
            (selected_node, idx_li) = self.eval_and_expand(selected_node, &bc);
            println!("{}", self);
            self.nodes[0].move_idx = Some(idx_li);
            let mut legal_moves: Vec<Move>;
            if self.nodes[selected_node] == self.nodes[0] {
                legal_moves = Vec::new();
                self.nodes[selected_node]
                    .board
                    .board()
                    .generate_moves(|moves| {
                        // Unpack dense move set into move list
                        legal_moves.extend(moves);
                        false
                    });
                // add Dirichlet noise
                // let mut std_rng = StdRng::from_entropy();
                // let distr = StableDirichlet::new(0.3, legal_moves.len()).expect("wrong params");
                // let sample = std_rng.sample(distr);
                // println!("noise: {:?}", sample);
                // for i in 0..self.nodes[0].children.len() {
                //     let child = self.nodes[0].children[i];
                //     self.nodes[child].policy =
                //         (1.0 - EPS) * self.nodes[child].policy + (EPS * sample[i]);
                // }
                // self.display_full_tree();
            }
        }
        // println!("        root node: {}", &self.nodes[0]);
        self.backpropagate(selected_node);
        // println!("{:?}", self);
        // self.display_full_tree();
    }
    // fn display_full_tree(&self) {
    //     // println!("        root node:");
    //     // println!("            {}", self.nodes[0]);
    //     // println!("        children:");
    //     let max_tree_print_depth: u8 = 10;
    //     println!("    {}", self.nodes[0]);
    //     self.layer_p(0, max_tree_print_depth, 0);
    // }
}

impl fmt::Display for Tree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg: String;
        match &self.nodes[0].move_name {
            Some(move_name) => {
                let m1 = "This is object of type Node and represents action ";
                let m2 = format!("{}", move_name);
                msg = m1.to_string() + &m2;
            }
            None => {
                msg = "Node at starting board position".to_string();
            }
        }
        write!(f, "{}", msg)
    }
}

pub const MAX_NODES: i32 = 100;

pub fn get_move(bs: BoardStack) -> (Move, Vec<f32>, Option<Vec<usize>>) {
    // equiv to move() in mcts_trainer.py
    // println!("{}", board);
    // spawn processes
    // let mut pack = Packet{
    //     visits:0,
    //     top:None,
    // }
    let mut tree = Tree::new(bs.clone()); // change if needed, maybe take a &mut of it
    while tree.nodes[0].visits < MAX_NODES {
        println!("step {}", tree.nodes[0].visits);
        tree.step(bs.clone());
    }

    let best_move_node = &tree.nodes[0]
        .children
        .iter()
        .max_by_key(|&n| tree.nodes[*n].visits)
        .expect("Error");
    let best_move = tree.nodes[**best_move_node].move_name;
    let mut total_visits_list = Vec::new();

    for child in &tree.nodes[0].children {
        total_visits_list.push(tree.nodes[*child].visits);
    }

    let total_visits: i32 = total_visits_list.iter().sum();

    let mut pi: Vec<f32> = Vec::new();

    println!("{:?}", &total_visits_list);

    for &t in &total_visits_list {
        let prob = t as f32 / total_visits as f32;
        pi.push(prob);
    }

    println!("{:?}", &pi);
    (best_move.unwrap(), pi, tree.nodes[0].clone().move_idx)
}
