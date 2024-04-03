use crate::{
    boardmanager::BoardStack,
    dataformat::ZeroEvaluation,
    decoder::{convert_board, process_board_output},
    dirichlet::StableDirichlet,
    executor::{Packet, ReturnMessage},
    settings::SearchSettings,
    superluminal::{CL_GREEN, CL_PINK},
    uci::eval_in_cp,
};
use cozy_chess::{Color, GameStatus, Move};
use flume::Sender;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    cmp::{max, min},
    fmt,
    ops::Range,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use superluminal_perf::{begin_event_with_color, end_event};
use tch::{
    utils::{has_cuda, has_mps},
    CModule, Device,
};

pub struct Net {
    pub net: CModule,
    pub device: Device,
}

#[derive(Clone, Copy, Debug, PartialEq)]

pub enum TypeRequest {
    TrainerSearch(Option<ExpansionType>),
    NonTrainerSearch,
    SyntheticSearch,

    UCISearch,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExpansionType {
    PolicyExpansion,
    RandomExpansion,
}

#[derive(Clone, Copy)]

pub enum EvalMode {
    WdlMode,   // use wdl
    ValueMode, // use value itself (1 value)
}

impl Net {
    pub fn new(path: &str) -> Self {
        // let path = "tz.pt";
        // // println!("{}", path);
        let mut net = tch::CModule::load(path).expect("ERROR");

        let device = if has_cuda() {
            Device::Cuda(0)
        } else if has_mps() {
            Device::Mps
        } else {
            Device::Cpu
        };

        net.set_eval();
        Self {
            net: net,
            device: device,
        }
    }
}

// define tree and node classes

#[derive(PartialEq, Debug)]
pub struct Tree {
    pub board: BoardStack,
    pub nodes: Vec<Node>,
    pub settings: SearchSettings,
    pub pv: String, // Some() when it is UCI code only
}

impl Tree {
    pub fn new(board: BoardStack, settings: SearchSettings) -> Tree {
        let root_node = Node::new(0.0, None, None);
        let mut container: Vec<Node> = Vec::new();
        container.push(root_node);
        let pv = String::new();
        Tree {
            board,
            nodes: container,
            settings,
            pv,
        }
    }

    pub async fn step(&mut self, tensor_exe_send: &Sender<Packet>, sw: Instant, id: usize) {
        // let sw = Instant::now();
        let display_str = self.display_node(0);
        // // println!("root node: {}", &display_str);
        // const EPS: f32 = 0.3; // 0.3 for chess
        let now_start_proc = SystemTime::now();
        let since_epoch_proc = now_start_proc
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");

        let epoch_seconds_start_proc = since_epoch_proc.as_nanos();
        let (selected_node, input_b, (min_depth, max_depth)) = self.select();
        let now_end_proc = SystemTime::now();
        let since_epoch_proc = now_end_proc
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let epoch_seconds_end_proc = since_epoch_proc.as_nanos();
        //     if id % 512 == 0 {
        //     println!(
        //         "{} {} {} select",
        //         epoch_seconds_start_proc, epoch_seconds_end_proc, id
        //     );
        // }

        // self.nodes[0].display_full_tree(self);

        let mut selected_node = selected_node;
        let idx_li: Vec<usize>;

        // check for terminal state
        if !input_b.is_terminal() {
            (selected_node, idx_li) = self
                .eval_and_expand(selected_node, &input_b, &tensor_exe_send, id)
                .await;

            self.nodes[selected_node].move_idx = Some(idx_li);
            let mut legal_moves: Vec<Move>;
            if selected_node == 0 {
                legal_moves = Vec::new();
                self.board.board().generate_moves(|moves| {
                    // Unpack dense move set into move list
                    legal_moves.extend(moves);
                    false
                });
                match self.settings.search_type {
                    TypeRequest::TrainerSearch(_) => {
                        // add policy softmax temperature and Dirichlet noise
                        // TODO extract below as a function?
                        let mut sum = 0.0;
                        for child in self.nodes[0].children.clone() {
                            self.nodes[child].policy =
                                self.nodes[child].policy.powf(self.settings.pst);
                            sum += self.nodes[child].policy;
                        }
                        for child in self.nodes[0].children.clone() {
                            self.nodes[child].policy /= sum;
                        }

                        // add Dirichlet noise
                        let mut std_rng = StdRng::from_entropy();
                        let distr = StableDirichlet::new(self.settings.alpha, legal_moves.len())
                            .expect("wrong params");
                        let sample = std_rng.sample(distr);
                        // // println!("noise: {:?}", sample);
                        for child in self.nodes[0].children.clone() {
                            self.nodes[child].policy = (1.0 - self.settings.eps)
                                * self.nodes[child].policy
                                + (self.settings.eps * sample[child - 1]);
                        }
                    }
                    TypeRequest::NonTrainerSearch => {}
                    TypeRequest::SyntheticSearch => {}
                    TypeRequest::UCISearch => {}
                }
                // self.nodes[0].display_full_tree(self);
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
        let now_start_proc = SystemTime::now();
        let since_epoch_proc = now_start_proc
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");

        let epoch_seconds_start_proc = since_epoch_proc.as_nanos();
        self.backpropagate(selected_node);
        let now_end_proc = SystemTime::now();
        let since_epoch_proc = now_end_proc
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let epoch_seconds_end_proc = since_epoch_proc.as_nanos();
        // if id % 512 == 0 {
        //     println!(
        //         "{} {} {} backprop_tree",
        //         epoch_seconds_start_proc, epoch_seconds_end_proc, id
        //     );
        // }
        // for child in &self.nodes[0].children {
        //     let display_str = self.display_node(*child);
        //     // // println!("children: {}", &display_str);
        // }
        // self.nodes[0].display_full_tree(self);
        match self.settings.search_type {
            TypeRequest::UCISearch => {
                let cp_eval = eval_in_cp(self.nodes[selected_node].eval_score);
                let elapsed_ms = sw.elapsed().as_nanos() as f32 / 1e6;
                let nps = self.nodes[0].visits as f32 / (sw.elapsed().as_nanos() as f32 / 1e9);
                let pv = self.get_pv();
                if self.pv != pv {

                    println!(
                        "info depth {} seldepth {} score cp {} nodes {} nps {} time {} pv {}",
                        min_depth,
                        max_depth,
                        (cp_eval * 100.).round().max(-1000.).min(1000.) as i64,
                        self.nodes.len(),
                        nps as usize,
                        elapsed_ms as usize,
                        pv,
                    );
                    self.pv = pv;
                } else {

                }
            }
            _ => {}
        }
    }

    fn get_pv(&self) -> String {
        let mut pv_nodes: Vec<usize> = vec![];
        let mut curr_node = 0;
        loop {
            if self.nodes[curr_node].children.is_empty() || self.board.is_terminal() {
                break;
            }
            curr_node = self.nodes[curr_node]
                .children
                .clone()
                .max_by_key(|&n| self.nodes[n].visits)
                .expect("Error");
            pv_nodes.push(curr_node);
        }
        let mut pv_string: String = String::new();
        if pv_nodes.is_empty() {
            pv_string
        } else {
            for item in pv_nodes {
                pv_string.push_str(&format!("{} ", self.nodes[item].mv.unwrap()));
            }
            pv_string
        }
    }

    pub fn depth_range(&self, node: usize) -> (usize, usize) {
        match self.settings.search_type {
            TypeRequest::UCISearch => match self.nodes[node].children.len() {
                0 => (0, 0),
                _ => {
                    let mut total_min = usize::MAX;
                    let mut total_max = usize::MIN;

                    for child in self.nodes[node].children.clone() {
                        let (c_min, c_max) = self.depth_range(child);
                        total_min = min(total_min, c_min);
                        total_max = max(total_max, c_max);
                    }

                    (total_min + 1, total_max + 1)
                }
            },
            _ => (0, 0),
        }
    }
    fn select(&mut self) -> (usize, BoardStack, (usize, usize)) {
        let mut curr: usize = 0;
        // println!("    selection:");
        let mut input_b: BoardStack;
        input_b = self.board.clone();
        let fenstr = format!("{}", &input_b.board());
        // // println!("    board FEN: {}", fenstr);
        let mut depth = 1;
        let mut max_depth: usize = 1;
        loop {
            let curr_node = &self.nodes[curr];
            if curr_node.children.is_empty() || input_b.is_terminal() {
                break;
            }
            // get number of visits for children
            // step 1, use curr.children vec<usize> to index tree.nodes (get a slice)
            let children = &curr_node.children;
            // step 2, iterate over them and get the child with highest PUCT value
            let mut total_visits = 0;
            for child in children.clone() {
                total_visits += &self.nodes[child].visits;
            }
            (_, max_depth) = self.depth_range(curr);
            curr = children
                .clone()
                .max_by(|a, b| {
                    let a_node = &self.nodes[*a];
                    let b_node = &self.nodes[*b];
                    let a_puct = a_node.puct_formula(
                        curr_node.visits,
                        curr_node.moves_left,
                        input_b.board().side_to_move(),
                        self.settings,
                    );
                    let b_puct = b_node.puct_formula(
                        curr_node.visits,
                        curr_node.moves_left,
                        input_b.board().side_to_move(),
                        self.settings,
                    );
                    // // println!("{}, {}", self.display_node(**a), self.display_node(**b));
                    // // println!("{}, {}", a_puct, b_puct);
                    // // println!("    CURRENT {:?}, {:?}", &a_node, &b_node);
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
            // let display_str = self.display_node(curr);
            // // println!("        selected: {}", display_str);
            input_b.play(self.nodes[curr].mv.expect("Error"));
            depth += 1;
        }
        // let display_str = self.display_node(curr);
        // // println!("    {}", display_str);
        // // println!("        children:");

        (curr, input_b, (depth, max_depth))
    }

    async fn eval_and_expand(
        &mut self,
        selected_node_idx: usize,
        bs: &BoardStack,
        tensor_exe_send: &Sender<Packet>,
        id: usize,
    ) -> (usize, Vec<usize>) {
        let sw = Instant::now();
        let fenstr = format!("{}", bs.board());
        // // println!("    board FEN: {}", fenstr);P
        // // println!("    ran NN:");

        let input_tensor = convert_board(&bs);

        // creating a send/recv pair for executor

        let (resender_send, resender_recv) = flume::bounded::<ReturnMessage>(1); // mcts to executor
        let thread_name = std::thread::current()
            .name()
            .unwrap_or("unnamed-generator")
            .to_owned();
        let pack = Packet {
            job: input_tensor,
            resender: resender_send,
            id: thread_name.clone(),
        };
        // println!("pre-requesting eval {}", id);
        let sw = Instant::now();
        let now_start_send = SystemTime::now();
        let since_epoch_send = now_start_send
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");

        let epoch_seconds_start_send = since_epoch_send.as_nanos();
        begin_event_with_color("send_req", CL_GREEN);
        tensor_exe_send.send_async(pack).await.unwrap();
        end_event();
        let now_end_send = SystemTime::now();
        let since_epoch_send = now_end_send
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let epoch_seconds_end_send = since_epoch_send.as_nanos();

        // if id % 512 == 0 {
        //     println!(
        //         "{} {} {} send_request",
        //         epoch_seconds_start_send, epoch_seconds_end_send, id
        //     );
        // }

        let now_start_recv = SystemTime::now();
        let since_epoch_recv = now_start_recv
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");

        let epoch_seconds_start_recv = since_epoch_recv.as_nanos();
        begin_event_with_color("recv_request", CL_PINK);
        let output = resender_recv.recv_async().await.unwrap();
        end_event();
        // println!("total_waiting_for_gpu_eval {}s", sw.elapsed().as_nanos() as f32 / 1e9);
        let now_end_recv = SystemTime::now();
        let since_epoch_recv = now_end_recv
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let epoch_seconds_end_recv = since_epoch_recv.as_nanos();
        // if id % 512 == 0 {
        //     println!(
        //         "{} {} {} recv_request",
        //         epoch_seconds_start_recv, epoch_seconds_end_recv, id
        //     );

        //     // println!("THREAD ID {} CHANNEL_LEN {}", id, tensor_exe_send.len());
        // }
        let output = match output {
            ReturnMessage::ReturnMessage(Ok(output)) => output,
            ReturnMessage::ReturnMessage(Err(_)) => panic!("error in returning!"),
        };
        // // println!("{},{}", thread_name,output.id);
        assert!(thread_name == output.id);
        let now_start_proc = SystemTime::now();
        let since_epoch_proc = now_start_proc
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");

        let epoch_seconds_start_proc = since_epoch_proc.as_nanos();
        let idx_li = process_board_output(output.packet, &selected_node_idx, self, &bs);
        let now_end_proc = SystemTime::now();
        let since_epoch_proc = now_end_proc
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let epoch_seconds_end_proc = since_epoch_proc.as_nanos();
        // if id % 512 == 0 {
        //     println!(
        //         "{} {} {} proc",
        //         epoch_seconds_start_proc, epoch_seconds_end_proc, id
        //     );
        // }
        // let idx_li = eval_board(&bs, &net, self, &selected_node_idx);
        (selected_node_idx, idx_li)
    }

    fn backpropagate(&mut self, node: usize) {
        // println!("    backup:");
        let n: f32 = match self.settings.wdl {
            Some(_) => {
                1.0 * self.nodes[node].wdl.w
                    + (-1.0 * self.nodes[node].wdl.l)
                    + (self.nodes[node].wdl.d)
            }
            None => self.nodes[node].eval_score,
        };
        let mut curr: Option<usize> = Some(node); // used to index parent
                                                  // // println!("    curr: {:?}", curr);
        while let Some(current) = curr {
            self.nodes[current].visits += 1;
            self.nodes[current].total_action_value += n;
            // // println!("    updated total action value: {}", self.nodes[current].total_action_value);
            curr = self.nodes[current].parent;
            // let display_str = self.display_node(current);
            // // println!("        updated node to {}", display_str);
        }
    }

    pub fn display_node(&self, id: usize) -> String {
        let u: f32;
        let puct: f32;

        match &self.nodes[id].parent {
            Some(parent) => {
                if self.nodes[*parent].visits == 0 {
                    u = f32::NAN;
                    puct = f32::NAN;
                } else {
                    u = self.nodes[id].get_u_val(self.nodes[*parent].visits, self.settings);
                    puct = self.nodes[id].puct_formula(
                        self.nodes[*parent].visits,
                        self.nodes[*parent].moves_left,
                        self.board.board().side_to_move(),
                        self.settings,
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
            "Node(action= {}, V= {}, N={}, W={}, P={}, Q={}, U={}, PUCT={}, len_children={}, wdl={}, w={}, d={}, l={})",
            mv_n,
            self.nodes[id].eval_score,
            self.nodes[id].visits,
            self.nodes[id].total_action_value,
            self.nodes[id].policy,
            self.nodes[id].get_q_val(self.settings),
            u,
            puct,
            self.nodes[id].children.len(),
            self.nodes[id].wdl.w,
            self.nodes[id].wdl.d,
            self.nodes[id].wdl.l,
            1.0 * self.nodes[id].wdl.w
                    + (-1.0 * self.nodes[id].wdl.l)
            ,
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
    pub parent: Option<usize>,
    pub children: Range<usize>,
    pub policy: f32,
    pub visits: u32,
    pub eval_score: f32, // -1 for black and 1 for white
    pub wdl: Wdl,
    pub total_wdl: Wdl,
    pub total_action_value: f32,
    pub mv: Option<Move>,
    pub moves_left: f32,
    pub moves_left_total: f32,
    pub move_idx: Option<Vec<usize>>,
}
#[derive(PartialEq, Clone, Debug, Copy)]
pub struct Wdl {
    pub w: f32,
    pub d: f32,
    pub l: f32,
}

impl Node {
    // fn is_leaf(&self) -> bool {
    //     self.visits == 0
    // }

    pub fn get_q_val(&self, settings: SearchSettings) -> f32 {
        let fpu = settings.fpu; // First Player Urgency
        if self.visits > 0 {
            self.total_action_value / (self.visits as f32)
        } else {
            fpu
        }
    }

    pub fn get_u_val(&self, parent_visits: u32, settings: SearchSettings) -> f32 {
        let c_puct = settings.c_puct; // "constant determining the level of exploration"
        c_puct * self.policy * ((parent_visits - 1) as f32).sqrt() / (1.0 + self.visits as f32)
    }

    pub fn puct_formula(
        &self,
        parent_visits: u32,
        parent_moves_left: f32,
        player: Color,
        settings: SearchSettings,
    ) -> f32 {
        let u = self.get_u_val(parent_visits, settings);
        let q = self.get_q_val(settings);
        let puct_logit = match settings.moves_left {
            Some(weights) => {
                let m = if self.visits == 0 {
                    // don't even bother with moves_left if we don't have any information
                    0.0
                } else {
                    // this node has been visited, so we know parent_moves_left is also a useful value
                    self.moves_left - (parent_moves_left - 1.0)
                };
                let m_unit = if weights.moves_left_weight == 0.0 {
                    0.0
                } else {
                    let m_clipped = self
                        .moves_left
                        .clamp(-weights.moves_left_clip, weights.moves_left_clip);
                    (weights.moves_left_sharpness * m_clipped * -q).clamp(-1.0, 1.0)
                };

                q + settings.c_puct * u + weights.moves_left_weight * m_unit
            }
            None => q + u,
        };
        match player {
            Color::Black => -puct_logit,
            Color::White => puct_logit,
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
            wdl: Wdl {
                w: f32::NAN,
                d: f32::NAN,
                l: f32::NAN,
            },
            total_wdl: Wdl {
                w: 0.0,
                d: 0.0,
                l: 0.0,
            },
            moves_left: f32::NAN,
            moves_left_total: f32::NAN,
        }
    }

    pub fn layer_p(&self, depth: u8, max_tree_print_depth: u8, tree: &Tree) {
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

    pub fn display_full_tree(&self, tree: &Tree) {
        // // println!("        root node:");
        let display_str = tree.display_node(0);
        // // println!("            {}", display_str);
        // // println!("        children:");
        let max_tree_print_depth: u8 = 3;
        // // println!("    {}", display_str);
        self.layer_p(0, max_tree_print_depth, tree);
    }
}

pub async fn get_move(
    bs: BoardStack,
    tensor_exe_send: &Sender<Packet>,
    settings: SearchSettings,
    id: usize,
) -> (
    Move,
    ZeroEvaluation,
    Option<Vec<usize>>,
    ZeroEvaluation,
    u32,
) {
    // equiv to move() in mcts_trainer.py

    // load nn and pass to eval if needed

    // let mut net = Net::new("chess_16x128_gen3634.pt");
    // net.net.set_eval();
    // net.net.to(net.device, Kind::Float, true);
    // // println!("{:?}", &bs);
    let mut tree = Tree::new(bs, settings);
    if tree.board.is_terminal() {
        panic!("No valid move!/Board is already game over!");
    }

    let search_type = TypeRequest::TrainerSearch;
    let sw = Instant::now();
    while tree.nodes[0].visits < settings.max_nodes as u32 {
        let thread_name = std::thread::current()
            .name()
            .unwrap_or("unnamed")
            .to_owned();
        // println!("step {}", tree.nodes[0].visits);
        // println!("thread {}, step {}", thread_name, tree.nodes[0].visits);
        let now_start_proc = SystemTime::now();
        let since_epoch_proc = now_start_proc
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");

        let epoch_seconds_start_proc = since_epoch_proc.as_nanos();
        tree.step(&tensor_exe_send, sw, id).await;
        let now_end_proc = SystemTime::now();
        let since_epoch_proc = now_end_proc
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let epoch_seconds_end_proc = since_epoch_proc.as_nanos();
        //     if id % 512 == 0 {
        //     println!(
        //         "{} {} {} step",
        //         epoch_seconds_start_proc, epoch_seconds_end_proc, id
        //     );
        // }
    }

    let mut child_visits: Vec<u32> = Vec::new();

    for child in tree.nodes[0].children.clone() {
        child_visits.push(tree.nodes[child].visits);
    }

    let all_same = child_visits.iter().all(|&x| x == child_visits[0]);

    let best_move_node = if !all_same {
        // if visits to nodes are the same eg max_nodes=1
        tree.nodes[0]
            .children
            .clone()
            .max_by_key(|&n| tree.nodes[n].visits)
            .expect("Error")
    } else {
        tree.nodes[0]
            .children
            .clone()
            .max_by(|a, b| {
                let a_node = &tree.nodes[*a];
                let b_node = &tree.nodes[*b];
                let a_policy = a_node.policy;
                let b_policy = b_node.policy;
                a_policy.partial_cmp(&b_policy).unwrap()
            })
            .expect("Error")
    };
    let best_move = tree.nodes[best_move_node].mv;
    let mut total_visits_list = Vec::new();
    // // println!("{:#}", best_move.unwrap());
    for child in tree.nodes[0].children.clone() {
        total_visits_list.push(tree.nodes[child].visits);
    }

    // let display_str = tree.display_node(0); // print root node
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
    // tree.nodes[0].display_full_tree(&tree);

    let mut all_pol = Vec::new();

    for child in tree.nodes[0].clone().children {
        all_pol.push(tree.nodes[child].policy);
    }

    let v_p = ZeroEvaluation {
        // network evaluation, NOT search/empirical data
        values: tree.nodes[0].eval_score,
        policy: all_pol,
    };

    let search_data = ZeroEvaluation {
        // search data
        values: tree.nodes[0].get_q_val(tree.settings),
        policy: pi,
    };

    (
        best_move.expect("Error"),
        v_p,
        tree.nodes[0].clone().move_idx,
        search_data,
        tree.nodes[0].visits,
    )
}
