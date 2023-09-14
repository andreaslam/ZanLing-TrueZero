use crate::boardmanager::BoardStack;
use crate::decoder::eval_board;
use crate::dirichlet::StableDirichlet;
use cozy_chess::{Color, GameStatus, Move};
use rand::SeedableRng;
use rand::{rngs::StdRng, Rng};
use std::ops::Range;
use std::{fmt, thread};
use tch::{CModule, Device, Kind};

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
        let path = "chess_16x128_gen3634.pt";
        // let path = "tz.pt";
        // println!("{}", path);
        Self {
            net: tch::CModule::load(path).expect("ERROR"),
            // device: Device::Cpu,
            device: Device::cuda_if_available(),
        }
    }
}

// define tree and node classes

#[derive(PartialEq, Debug)]
pub struct Tree {
    pub board: BoardStack,
    pub nodes: Vec<Node>,
}

impl Tree {
    fn new(board: BoardStack) -> Tree {
        let root_node = Node::new(0.0, None, None);
        let mut container: Vec<Node> = Vec::new();
        container.push(root_node);
        Tree {
            board,
            nodes: container,
        }
    }

    fn step(&mut self, net: &Net) {
        let display_str = self.display_node(0);
        // // println!("root node: {}", &display_str);
        const EPS: f32 = 0.3; // 0.3 for chess
        let (selected_node, input_b) = self.select();

        // self.nodes[0].display_full_tree(self);

        let mut selected_node = selected_node;
        let idx_li: Vec<usize>;

        // check for terminal state
        if !input_b.is_terminal() {
            (selected_node, idx_li) = self.eval_and_expand(selected_node, &input_b, &net);
            // // println!("{}", self);
            self.nodes[selected_node].move_idx = Some(idx_li);
            let mut legal_moves: Vec<Move>;
            if selected_node == 0 {
                legal_moves = Vec::new();
                self.board.board().generate_moves(|moves| {
                    // Unpack dense move set into move list
                    legal_moves.extend(moves);
                    false
                });
                // add Dirichlet noise
                let mut std_rng = StdRng::from_entropy();
                let distr = StableDirichlet::new(0.3, legal_moves.len()).expect("wrong params");
                let sample = std_rng.sample(distr);
                // // println!("noise: {:?}", sample);
                for child in self.nodes[0].children.clone() {
                    self.nodes[child].policy =
                        (1.0 - EPS) * self.nodes[child].policy + (EPS * sample[child - 1]);
                }
                self.nodes[0].display_full_tree(self);
            }
        } else {
            self.nodes[selected_node].eval_score = match input_b.status() {
                GameStatus::Drawn => 0.0,
                GameStatus::Won => match !input_b.board().side_to_move() {
                    Color::White => 1.0,
                    Color::Black => -1.0,
                },
                GameStatus::Ongoing => {
                    unreachable!()
                }
            }
        }
        self.backpropagate(selected_node);
        // for child in &self.nodes[0].children {
        //     let display_str = self.display_node(*child);
        //     // // println!("children: {}", &display_str);
        // }
        // self.nodes[0].display_full_tree(self);
    }

    fn select(&mut self) -> (usize, BoardStack) {
        let mut curr: usize = 0;
        // // println!("    selection:");
        let mut input_b: BoardStack;
        input_b = self.board.clone();
        let fenstr = format!("{}", &input_b.board());
        // // println!("    board FEN: {}", fenstr);
        loop {
            let curr_node = &self.nodes[curr];
            if curr_node.children.is_empty() || input_b.is_terminal() {
                break;
            }
            // // println!("{:?}", curr);
            // get number of visits for children
            // step 1, use curr.children vec<usize> to index tree.nodes (get a slice)
            let children = &curr_node.children;
            // step 2, iterate over them and get the child with highest PUCT value
            let mut total_visits = 0;
            for child in children.clone() {
                total_visits += &self.nodes[child].visits;
            }
            curr = children
                .clone()
                .max_by(|a, b| {
                    let a_node = &self.nodes[*a];
                    let b_node = &self.nodes[*b];
                    let a_puct =
                        a_node.puct_formula(curr_node.visits, input_b.board().side_to_move());
                    let b_puct =
                        b_node.puct_formula(curr_node.visits, input_b.board().side_to_move());
                    // // println!("{}, {}", self.display_node(**a), self.display_node(**b));
                    // // println!("{}, {}", a_puct, b_puct);
                    if a_puct == b_puct || curr_node.visits == 0 {
                        // if PUCT values are equal or parent visits == 0, use largest policy as tiebreaker
                        let a_policy = a_node.policy;
                        let b_policy = b_node.policy;
                        a_policy.partial_cmp(&b_policy).unwrap()
                    } else {
                        a_puct.partial_cmp(&b_puct).unwrap()
                    }
                })
                .expect("Error");
            // // println!("{}, {}", total_visits + 1, curr_node.visits);
            assert!(total_visits + 1 == curr_node.visits);
            let display_str = self.display_node(curr);
            // // println!("    selected: {}", display_str);
            input_b.play(self.nodes[curr].mv.expect("Error"));
            let fenstr = format!("{}", &input_b.board());
            // // println!("    board FEN: {}", fenstr);
        }
        let display_str = self.display_node(curr);
        // // println!("    {}", display_str);
        // // println!("        children:");

        (curr, input_b)
    }

    fn eval_and_expand(
        &mut self,
        selected_node_idx: usize,
        bs: &BoardStack,
        net: &Net,
    ) -> (usize, Vec<usize>) {
        let fenstr = format!("{}", bs.board());
        // // println!("    board FEN: {}", fenstr);
        // // println!("    ran NN:");
        let idx_li = eval_board(&bs, &net, self, &selected_node_idx);
        (selected_node_idx, idx_li)
    }

    fn backpropagate(&mut self, node: usize) {
        // // println!("    backup:");
        let n = self.nodes[node].eval_score;
        let mut curr: Option<usize> = Some(node); // used to index parent
                                                  // // println!("    curr: {:?}", curr);
        while let Some(current) = curr {
            self.nodes[current].visits += 1;
            self.nodes[current].total_action_value += n;
            // // println!("    updated total action value: {}", self.nodes[current].total_action_value);
            curr = self.nodes[current].parent;
            let display_str = self.display_node(current);
            // // println!("        updated node to {}", display_str);
        }
    }

    fn display_node(&self, id: usize) -> String {
        let u: f32;
        let puct: f32;

        match &self.nodes[id].parent {
            Some(parent) => {
                if self.nodes[*parent].visits == 0 {
                    u = f32::NAN;
                    puct = f32::NAN;
                } else {
                    u = self.nodes[id].get_u_val(self.nodes[*parent].visits);
                    puct = self.nodes[id].puct_formula(
                        self.nodes[*parent].visits,
                        self.board.board().side_to_move(),
                    );
                }
            }
            None => {
                u = f32::NAN;
                puct = f32::NAN;
            }
        }
        let mv_n: String;
        match &self.nodes[id].mv {
            Some(mv) => {
                mv_n = format!("{}", mv);
            }
            None => {
                mv_n = "Null".to_string();
            }
        }

        format!(
            "Node(action= {}, V= {}, N={}, W={}, P={}, Q={}, U={}, PUCT={}, len_children={})",
            mv_n,
            self.nodes[id].eval_score,
            self.nodes[id].visits,
            self.nodes[id].total_action_value,
            self.nodes[id].policy,
            self.nodes[id].get_q_val(),
            u,
            puct,
            self.nodes[id].children.len()
        )
    }
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

#[derive(PartialEq, Clone, Debug)] // maybe display and debug as helper funcs to check impl
pub struct Node {
    parent: Option<usize>,
    pub children: Range<usize>,
    pub policy: f32,
    pub visits: u32,
    pub eval_score: f32, // -1 for black and 1 for white
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

    fn get_u_val(&self, parent_visits: u32) -> f32 {
        let c_puct = 2.0; // "constant determining the level of exploration"
        c_puct * self.policy * ((parent_visits - 1) as f32).sqrt() / (1.0 + self.visits as f32)
    }

    fn puct_formula(&self, parent_visits: u32, player: Color) -> f32 {
        let u = self.get_u_val(parent_visits);
        let q = self.get_q_val();
        match player {
            Color::Black => -q + u,
            Color::White => q + u,
        }
    }

    pub fn new(policy: f32, parent: Option<usize>, mv: Option<cozy_chess::Move>) -> Node {
        Node {
            parent,
            children: 0..0,
            policy,
            visits: 0,
            eval_score: f32::NAN,
            total_action_value: 0.0,
            mv,
            move_idx: None,
        }
    }

    fn layer_p(&self, depth: u8, max_tree_print_depth: u8, tree: &Tree) {
        let indent = "    ".repeat(depth as usize + 2);
        if depth <= max_tree_print_depth {
            if !self.children.is_empty() {
                for c in self.children.clone() {
                    let display_str = tree.display_node(c);
                    // println!("{}{}", indent, display_str);
                    tree.nodes[c].layer_p(depth + 1, max_tree_print_depth, tree);
                }
            }
        }
    }

    fn display_full_tree(&self, tree: &Tree) {
        // // println!("        root node:");
        let display_str = tree.display_node(0);
        // // println!("            {}", display_str);
        // // println!("        children:");
        let max_tree_print_depth: u8 = 3;
        // // println!("    {}", display_str);
        self.layer_p(0, max_tree_print_depth, tree);
    }
}

pub const MAX_NODES: u32 = 100;

pub fn get_move(bs: BoardStack) -> (Move, Vec<f32>, Option<Vec<usize>>) {
    // equiv to move() in mcts_trainer.py
    // spawn processes
    // let mut pack = Packet{
    //     visits:0,
    //     top:None,
    // }

    // load nn and pass to eval if needed

    let mut net = Net::new();
    net.net.set_eval();
    net.net.to(net.device, Kind::Float, true);
    let mut tree = Tree::new(bs);
    if tree.board.is_terminal() {
        panic!("No valid move!/Board is already game over!");
    }
    while tree.nodes[0].visits < MAX_NODES {
        // // println!("step {} :", tree.nodes[0].visits);
        tree.step(&net);
    }
    let best_move_node = tree.nodes[0]
        .children
        .clone()
        .max_by_key(|&n| tree.nodes[n].visits)
        .expect("Error");
    let best_move = tree.nodes[best_move_node].mv;
    let mut total_visits_list = Vec::new();
    // // println!("{:#}", best_move.unwrap());
    for child in tree.nodes[0].children.clone() {
        total_visits_list.push(tree.nodes[child].visits);
    }

    let display_str = tree.display_node(0); // print root node
                                            // // println!("{}", display_str);
    let total_visits: u32 = total_visits_list.iter().sum();

    let mut pi: Vec<f32> = Vec::new();

    // // println!("{:?}", &total_visits_list);

    for &t in &total_visits_list {
        let prob = t as f32 / total_visits as f32;
        pi.push(prob);
    }

    // // println!("{:?}", &pi);
    // // println!("{}", best_move.expect("Error").to_string());
    // // println!("best move: {}", best_move.expect("Error").to_string());

    // for child in tree.nodes[0].children.clone() {
    //     let display_str = tree.display_node(child);
    //     // println!("{}", display_str);
    // }
    tree.nodes[0].display_full_tree(&tree);
    (
        best_move.expect("Error"),
        pi,
        tree.nodes[0].clone().move_idx,
    )
}
