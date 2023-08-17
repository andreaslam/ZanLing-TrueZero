use cozy_chess::*;
use tch::*;
use crate::decoder::eval_board;
use crate::decoder::convert_board;

// define tree and node classes
#[derive(PartialEq)]
struct Node {
    parent: Option<Box<Node>>, // or none
    children: Option<Box<Vec<Node>>>,
    policy: f32,
    visits: u32,
    eval_score: f32,
    board: Board,
    total_action_value: f32,
    move_name: cozy_chess::Move,
    move_idx: Vec<u16>,

} 

impl Node {

    fn is_leaf(&self) -> bool {
        self.visits == 0
    }

    fn get_q_val(&self) -> f32 {
        const FPU: f32 = 0.0;
        let s = self.visits as f32;
        let ans: f32;
        if self.visits > 0 {
            ans = self.total_action_value / s;
        } else {
            ans = FPU;
        }
    ans
    }
    fn puct_formula(&self, parent_visits:u32) -> f32 {
        const C_PUCT: f32 = 2.0;
        let pv = parent_visits as f32;
        let sv = self.visits as f32;
        let u = C_PUCT * self.policy * (pv - 1.0).sqrt() / (1.0 + sv);
        let q = self.get_q_val();
        let result = -q + u;
        result
    }

    fn is_terminal(&self,board:Board) -> bool {
        let status = board.status();
        status != GameStatus::Ongoing // returns true if game is over (not ongoing)
    }
    // lol skip __str__ and layer p lol

    fn eval_and_expand(&self,board:Board) -> Vec<u16>{ 
        let (value, policy, idx_li) = eval_board(board);
        self.eval_score=  value;
        for p in policy {
            let x = board.clone();
            x.play(p);
            let child = Node{
                board:board,
                children:Vec::new(),
                policy:policy[p], 
                parent:self, 
                move_name:p
            };
            self.children.extend(child);
            child.board = x;
        }
        idx_li
    }
}


#[derive(PartialEq)]
struct Tree {
    board: Board,
    root_node:Node
}

impl Tree {
    fn select(&self) -> Node {
        let curr = self.root_node;
        while curr.children.is_some() {
            // max bad
        }
    curr
    }

    fn backpropagate(&self, node:Node) {
        let n = node.eval_score;
        while node.is_some() {
            node.visits += 1;
            node.total_action_value += n;
            node = node.parent;
            n = -n;
        }
    }

    // skip display full tree

    fn step(&self) { // skip custom net
        const EPS:f32 = 0.3;
        let selected_node = self.select();
        if !selected_node.is_terminal(selected_node.board) {
            let idx_li = selected_node.eval_and_expand(selected_node.board);
            let are_same_instance = std::ptr::eq(&selected_node, &self.root_node);
            if are_same_instance {
                // add Dirichlet noise, skip for now
            }
            
            self.backpropagate(selected_node)

        } 
    }

}

// skip read txt

pub const MAX_NODES: u32 = 10000;

pub fn get_move(board:Board) -> (PieceMoves, Tensor, Vec<f32>, Vec<u16>){
    let tree = Tree{
        board:board,
        root_node: Node { parent: (None), children: (Vec<Node>), policy: (0.0), visits: (0), eval_score: (0.0), board: (board), total_action_value: (0.0), move_name: (PieceMoves), move_idx: (Vec<u16>) }};

    while tree.root_node.visits < MAX_NODES {
        tree.step();
    }


    let best_move_node = tree.root_node.children.iter().max_by_key(&tree.root_node.children.visits);

    let best_move = best_move_node.move_name;

    let total_visits_list = Vec::new();

    for child in tree.root_node.children {
        total_visits_list.extend(child.visits.iter());
    }
    let total_visits = total_visits_list.iter().sum();

    let pi = Vec::new();

    for t in total_visits_list {
        pi.extend(t / total_visits);
    }

    let rb_input = convert_board(tree.root_node.board);
    let memory_piece = rb_input;
    (best_move, memory_piece, pi, tree.root_node.move_idx)
}
