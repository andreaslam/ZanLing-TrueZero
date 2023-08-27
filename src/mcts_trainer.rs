use crate::boardmanager::BoardStack;
use crate::decoder::eval_board;
use crate::dirichlet::StableDirichlet;
use cozy_chess::{GameStatus, Move};
use rand::SeedableRng;
use rand::{rngs::StdRng, Rng};
use std::ops::Range;
use std::{fmt, thread};
use tch::{CModule, Device, Kind};

// define tree and node classes

// struct Packet {
//     visits: i32,
//     top: Option<Move>,
// }

pub struct Net {
    pub net: CModule,
    pub device: Device,
}

impl Net {
    pub fn new() -> Self {
        Self {
            net: tch::CModule::load("chess_16x128_gen3634.pt").expect("ERROR"),
            device: Device::cuda_if_available(),
        }
    }
}

#[derive(PartialEq, Clone, Debug)] // maybe display and debug as helper funcs to check impl
pub struct Node {
    parent: Option<usize>,
    pub children: Vec<usize>,
    policy: f32,
    visits: u32,
    pub eval_score: f32,
    total_action_value: f32,
    pub mv: Option<cozy_chess::Move>,
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

    fn puct_formula(&self, parent_visits: u32) -> f32 {
        let c_puct = 2.0; // "constant determining the level of exploration"
        let u =
            c_puct * self.policy * ((parent_visits - 1) as f32).sqrt() / (1.0 + self.visits as f32);
        let q = self.get_q_val();
        // println!("{},{}", self, -q+u);
        -q + u
    }

    fn is_terminal(&self, board: &BoardStack) -> bool {
        let status = board.status();
        status != GameStatus::Ongoing // returns true if game is over (not ongoing)
    }

    pub fn new(policy: f32, parent: Option<usize>, mv: Option<cozy_chess::Move>) -> Node {
        Node {
            parent,
            children: vec![],
            policy,
            visits: 0,
            eval_score: 0.0,
            total_action_value: 0.0,
            mv,
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
        match &self.mv {
            Some(mv) => {
                mv_n = format!("{}", mv);
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
pub struct Tree {
    pub board: BoardStack,
    pub nodes: Vec<Node>,
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
        let root_node = Node::new(0.0, None, None);
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
        bs: &mut BoardStack,
        net: &Net,
    ) -> (usize, Vec<usize>) {
        let fenstr = format!("{}", bs.board());
        println!("    board FEN: {}", fenstr);
        println!("    ran NN:");
        let idx_li = eval_board(&bs, &net, self, &selected_node_idx);
        (selected_node_idx, idx_li)
    }

    fn select(&mut self) -> (usize, BoardStack) {
        let mut curr: usize = 0;
        println!("    selection:");
        let mut input_b: BoardStack;
        input_b = self.board.clone();
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
            // let fenstr = format!("{}", input_b.board());
            // println!("{}", fenstr);
            println!("{:?}", self.nodes[curr].mv);
            input_b.play(self.nodes[curr].mv.expect("Error"));
        }
        // println!("    {}", curr);
        println!("        children:");

        (curr, input_b)
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
    fn step(&mut self, net: &Net) {
        println!("root node: {}", &self.nodes[0]);
        const EPS: f32 = 0.3; // 0.3 for chess
                              // self.display_full_tree();
        let selected_node: usize;
        let mut input_b: BoardStack;
        (selected_node, input_b) = self.select();
        // println!("{:p},{:p}", &selected_node, &self);
        let mut selected_node = selected_node;
        let idx_li: Vec<usize>;

        if !self.nodes[selected_node].is_terminal(&self.board) {
            (selected_node, idx_li) = self.eval_and_expand(selected_node, &mut input_b, &net);
            println!("{}", self);
            self.nodes[0].move_idx = Some(idx_li);
            let mut legal_moves: Vec<Move>;
            if self.nodes[selected_node] == self.nodes[0] {
                legal_moves = Vec::new();
                self.board.board().generate_moves(|moves| {
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
        match &self.nodes[0].mv {
            Some(mv) => {
                let m1 = "This is object of type Node and represents action ";
                let m2 = format!("{}", mv);
                msg = m1.to_string() + &m2;
            }
            None => {
                msg = "Node at starting board position".to_string();
            }
        }
        write!(f, "{}", msg)
    }
}

pub const MAX_NODES: u32 = 10;

pub fn get_move(bs: BoardStack) -> (Move, Vec<f32>, Option<Vec<usize>>) {
    // equiv to move() in mcts_trainer.py
    // println!("{}", board);
    // spawn processes
    // let mut pack = Packet{
    //     visits:0,
    //     top:None,
    // }

    // load nn and pass to eval if needed

    let mut net = Net::new();
    net.net.set_eval();
    net.net.to(net.device, Kind::Float, true);
    let mut tree = Tree::new(bs); // change if needed, maybe take a &mut of it
    while tree.nodes[0].visits < MAX_NODES {
        println!("step {}", tree.nodes[0].visits);
        tree.step(&net);
    }

    let best_move_node = &tree.nodes[0]
        .children
        .iter()
        .max_by_key(|&n| tree.nodes[*n].visits)
        .expect("Error");
    let best_move = tree.nodes[**best_move_node].mv;
    let mut total_visits_list = Vec::new();

    for child in &tree.nodes[0].children {
        total_visits_list.push(tree.nodes[*child].visits);
    }

    let total_visits: u32 = total_visits_list.iter().sum();

    let mut pi: Vec<f32> = Vec::new();

    println!("{:?}", &total_visits_list);

    for &t in &total_visits_list {
        let prob = t as f32 / total_visits as f32;
        pi.push(prob);
    }

    println!("{:?}", &pi);
    println!("best move: {}", best_move.unwrap());
    (best_move.unwrap(), pi, tree.nodes[0].clone().move_idx)
}
