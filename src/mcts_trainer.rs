use crate::debug_print;
use crate::{
    boardmanager::BoardStack,
    cache::{CacheEntryKey, CacheEntryValue},
    dataformat::ZeroEvaluation,
    decoder::{convert_board, extract_policy, process_board_output},
    dirichlet::StableDirichlet,
    executor::{Packet, ReturnMessage},
    mvs::get_contents,
    settings::SearchSettings,
    superluminal::{CL_GREEN, CL_PINK},
    uci::eval_in_cp,
    utils::TimeStampDebugger,
};
use cozy_chess::{Color, GameStatus, Move};
use flume::Sender;
use lru::LruCache;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    cmp::{max, min},
    fmt,
    ops::Range,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use superluminal_perf::{begin_event_with_color, end_event};
use tch::{
    maybe_init_cuda,
    utils::{has_cuda, has_mps},
    CModule, Cuda, Device,
};

#[derive(Clone, Copy, Debug, PartialEq)]

pub enum TypeRequest {
    TrainerSearch(Option<ExpansionType>),
    NonTrainerSearch,
    SyntheticSearch,

    UCISearch,
    VisualiserMode,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExpansionType {
    PolicyExpansion,
    RandomExpansion,
}

#[derive(Clone, Copy, Debug, PartialEq)]

pub enum EvalMode {
    Wdl,   // use wdl
    Value, // use value itself (1 value)
}

pub struct Net {
    pub net: CModule,
    pub device: Device,
}

impl Net {
    /// creates a new `Net` instance by loading a model from the specified path
    pub fn new(path: &str) -> Self {
        let device = if has_cuda() {
            if Cuda::cudnn_is_available() {
                Cuda::cudnn_set_benchmark(true);
            }
            Device::Cuda(0)
        } else {
            Device::Cpu
        };

        let mut net = tch::CModule::load_on_device(path, device).expect("ERROR");
        net.set_eval();

        Self {
            net: net,
            device: device,
        }
    }
    /// creates a new `Net` instance with a specified device ID (supports only CUDA)
    pub fn new_with_device_id(path: &str, id: usize) -> Self {
        let device = if has_cuda() {
            if Cuda::cudnn_is_available() {
                Cuda::cudnn_set_benchmark(true);
            }
            let id = if Cuda::device_count() <= id.try_into().unwrap() {
                0
            } else {
                id
            };
            Device::Cuda(id)
        } else if has_mps() {
            Device::Mps
        } else {
            Device::Cpu
        };

        let mut net = tch::CModule::load_on_device(path, device).expect("ERROR");
        net.set_eval();

        Self {
            net: net,
            device: device,
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct Tree {
    pub board: BoardStack,
    pub nodes: Vec<Node>, // this is where all the `Nodes` are stored, as opposed to storing them individually in `Node.children`
    pub settings: SearchSettings,
    pub pv: String,
}

impl Tree {
    /// creates a new `Tree` instance with a given board state and search settings
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

    /// executes one iteration of MCTS (selection, expansion, evaluation and backpropagation)

    pub async fn step(
        &mut self,
        tensor_exe_send: &Sender<Packet>, // sender to send tensors to `executor.rs``, which gathers MCTS simulations to execute in a batch
        sw: Instant,                      // timer for UCI
        id: usize,                        // unique ID assigned to each thread for debugging
        mut cache: &mut LruCache<CacheEntryKey, CacheEntryValue>,
    ) {
        let thread_name = format!("mcts-{}", id);

        let step_debugger = TimeStampDebugger::create_debug();

        let (selected_node, input_b, (min_depth, max_depth)) = self.select();
        if id % 512 == 0 {
            step_debugger.record("mcts select", &thread_name);
        }

        self.nodes[0].display_full_tree(self);

        let selected_node = selected_node;
        let idx_li: Vec<usize>;

        // check if the board has reached terminal state in selection

        if !input_b.is_terminal() {
            // check whether board position is in cache

            let cache_key = CacheEntryKey {
                hash: input_b.board().hash(),
                halfmove_clock: input_b.board().halfmove_clock(),
            };

            idx_li = match cache.get(&cache_key) {
                Some(packet) => {
                    // retrieve non-policy data
                    let ct = self.nodes.len();

                    self.nodes[selected_node].value = packet.eval_score;
                    self.nodes[selected_node].wdl = packet.wdl;
                    self.nodes[selected_node].moves_left = packet.moves_left;

                    // retrieve policy data and children

                    let contents = get_contents(); // this extracts the mapping for policy nodes according to `mvs.rs`
                    let (_, idx_li) = extract_policy(&input_b, contents); // filters the correct indices of policy nodes according to current board legal moves
                    let mut legal_moves: Vec<Move> = Vec::new();
                    input_b.board().generate_moves(|moves| {
                        // Unpack dense move set into move list
                        legal_moves.extend(moves);
                        false
                    });

                    let mut counter = 0;

                    for mv in legal_moves {
                        let pol = packet.policy[counter];
                        let new_child = Node::new(pol, Some(selected_node), Some(mv));
                        self.nodes.push(new_child); // push child to the tree Vec<Node>
                        counter += 1;
                    }
                    self.nodes[selected_node].children = ct..ct + counter; // push indices of `Node.children` (type `Range<usize>`)

                    idx_li // returns  idx_li, which is used for indexing legal moves
                }
                None => {
                    self.eval_and_expand(&selected_node, &input_b, &tensor_exe_send, id, cache)
                        .await // if there are no corresponding entries in the cache, request a nn evaluation
                }
            };

            self.nodes[selected_node].move_idx = Some(idx_li);
            let mut legal_moves: Vec<Move>;
            if selected_node == 0 {
                legal_moves = Vec::new();
                self.board.board().generate_moves(|moves| {
                    // Unpack dense move set into move list
                    legal_moves.extend(moves);
                    false
                });
                // add policy softmax temperature and Dirichlet noise
                let mut sum = 0.0;
                for child in self.nodes[0].children.clone() {
                    self.nodes[child].policy = self.nodes[child].policy.powf(self.settings.pst);
                    sum += self.nodes[child].policy;
                }
                for child in self.nodes[0].children.clone() {
                    self.nodes[child].policy /= sum;
                }
                match self.settings.search_type {
                    TypeRequest::TrainerSearch(_) => {
                        // add Dirichlet noise
                        let mut std_rng = StdRng::from_entropy();
                        let distr = StableDirichlet::new(self.settings.alpha, legal_moves.len())
                            .expect("wrong params");
                        let sample = std_rng.sample(distr);
                        //  debug_print(&format!("noise: {:?}", sample));
                        for child in self.nodes[0].children.clone() {
                            self.nodes[child].policy = (1.0 - self.settings.eps)
                                * self.nodes[child].policy
                                + (self.settings.eps * sample[child - 1]);
                        }
                    }
                    TypeRequest::NonTrainerSearch => {}
                    TypeRequest::SyntheticSearch => {}
                    TypeRequest::UCISearch => {}
                    TypeRequest::VisualiserMode => {}
                }
                self.nodes[0].display_full_tree(self);
            }
        } else {
            // handle terminal nodes - value and WDL assignment
            let wdl = match input_b.status() {
                GameStatus::Drawn => Wdl {
                    w: 0.0,
                    d: 1.0,
                    l: 0.0,
                },
                GameStatus::Won => match !input_b.board().side_to_move() {
                    Color::White => Wdl {
                        w: 1.0,
                        d: 0.0,
                        l: 0.0,
                    },
                    Color::Black => Wdl {
                        w: 0.0,
                        d: 0.0,
                        l: 1.0,
                    },
                },
                GameStatus::Ongoing => {
                    unreachable!()
                }
            };
            self.nodes[selected_node].value = wdl.w - wdl.l;
            self.nodes[selected_node].wdl = wdl;
        }

        self.backpropagate(selected_node);
        let backprop_debug = TimeStampDebugger::create_debug();
        if id % 512 == 0 {
            backprop_debug.record("backpropagation", &thread_name);
        }
        for child in self.nodes[0].children.clone() {
            let display_str = self.display_node(child);
            debug_print!("{}", &format!("children: {}", &display_str))
        }
        self.nodes[0].display_full_tree(self);
        match self.settings.search_type {
            TypeRequest::UCISearch => {
                let cp_eval = eval_in_cp(self.nodes[selected_node].value);
                let elapsed_ms = sw.elapsed().as_nanos() as f32 / 1e6;
                let nps =
                    self.nodes[0].visits as f32 / (sw.elapsed().as_nanos() as f32 / 1e9 as f32);
                let (pv, mate) = self.get_pv();
                let eval_string = {
                    if mate {
                        format!("mate {}", pv.len())
                    } else {
                        format!(
                            "cp {}",
                            (cp_eval * 100.).round().max(-1000.).min(1000.) as i64,
                        )
                    }
                };
                if self.pv != pv {
                    println!(
                        "info depth {} seldepth {} score {} nodes {} nps {} time {} pv {}",
                        min_depth,
                        max_depth,
                        eval_string,
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

    fn get_pv(&self) -> (String, bool) {
        let mut pv_nodes: Vec<usize> = vec![];
        let mut curr_node = 0;
        let mut terminal = false; // by default assume the pv has not reached mate
        loop {
            if self.nodes[curr_node].children.is_empty() || self.board.is_terminal() {
                if self.board.is_terminal() {
                    terminal = true;
                }
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
            (pv_string, terminal)
        } else {
            for item in pv_nodes {
                pv_string.push_str(&format!("{} ", self.nodes[item].mv.unwrap()));
            }
            (pv_string, terminal)
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

    /// selects the node to expand based on the PUCT formula/policy score if the visit count is 0
    fn select(&mut self) -> (usize, BoardStack, (usize, usize)) {
        let mut curr: usize = 0;
        debug_print!("{}", &format!("    selection:"));
        let mut input_b: BoardStack;
        input_b = self.board.clone();
        let fenstr = format!("{}", &input_b.board());
        debug_print!("{}", &format!("    board FEN: {}", fenstr));
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
                     debug_print!("{}",&format!(
                        "{}, {}",
                        self.display_node(*a),
                        self.display_node(*b)
                    ));
                     debug_print!("{}",&format!("{}, {}", a_puct, b_puct));
                     debug_print!("{}",&format!("    CURRENT {:?}, {:?}", &a_node, &b_node));
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
            debug_print!("{}",&format!("{}, {}", total_visits + 1, curr_node.visits));
            assert!(total_visits + 1 == curr_node.visits);
            let display_str = self.display_node(curr);
            debug_print!("{}",&format!("        selected: {}", display_str));
            input_b.play(self.nodes[curr].mv.expect("Error"));
            depth += 1;
        }
        let display_str = self.display_node(curr);
         debug_print!("{}",&format!("    {}", display_str));
         debug_print!("{}",&format!("        children:"));

        (curr, input_b, (depth, max_depth))
    }

    /// evaluates selected node using the neural network by sending the input tensors to `executor.rs` through async sender
    /// then the results are sent back and calls `decoder.rs` to process the nn ouputs and update the tree
    async fn eval_and_expand(
        &mut self,
        selected_node_idx: &usize,
        bs: &BoardStack,
        tensor_exe_send: &Sender<Packet>,
        id: usize,
        mut cache: &mut LruCache<CacheEntryKey, CacheEntryValue>,
    ) -> Vec<usize> {
        let input_tensor = convert_board(&bs);

        let (resender_send, resender_recv) = flume::bounded::<ReturnMessage>(1); // mcts to `executor.rs`
        let thread_name = std::thread::current()
            .name()
            .unwrap_or("unnamed-generator")
            .to_owned();
        let pack = Packet {
            job: input_tensor,
            resender: resender_send,
            id: thread_name.clone(),
        };

        begin_event_with_color("send_request", CL_GREEN);
        let send_logger = TimeStampDebugger::create_debug();
        tensor_exe_send.send_async(pack).await.unwrap();
        if id % 512 == 0 {
            send_logger.record("send_request", thread_name.as_str());
        }
        end_event();

        begin_event_with_color("recv_request", CL_PINK);
        let output = resender_recv.recv_async().await.unwrap();
        end_event();
        let recv_logger = TimeStampDebugger::create_debug();
        if id % 512 == 0 {
            recv_logger.record("recv_request", thread_name.as_str());
        }
        let output = match output {
            ReturnMessage::ReturnMessage(Ok(output)) => output,
            ReturnMessage::ReturnMessage(Err(_)) => panic!("error in returning!"),
        };

        assert!(thread_name == output.id);
        let idx_li = process_board_output(
            (&output.packet.0, &output.packet.1),
            &selected_node_idx,
            self,
            &bs,
            cache,
        );

        idx_li
    }

    /// backpropagates the evaluation results up the tree
    fn backpropagate(&mut self, node: usize) {
        debug_print!("{}", &format!("    backpropagation:"));
        let mut curr: Option<usize> = Some(node); // used to index parent
        debug_print!("{}", &format!("    curr: {:?}", curr));
        let wdl = self.nodes[node].wdl;
        let value = self.nodes[node].value;
        let mut moves_left = self.nodes[node].moves_left;
        while let Some(current) = curr {
            self.nodes[current].visits += 1;
            self.nodes[current].total_action_value += value;
            self.nodes[current].total_wdl += wdl;
            self.nodes[current].moves_left_total += moves_left;
            moves_left += 1.0;
            debug_print!("{}", &format!(
                "    updated total action value: {}",
                self.nodes[current].total_action_value
            ));
            curr = self.nodes[current].parent;
            let display_str = self.display_node(current);
            debug_print!("{}",&format!("        updated node to {}", display_str));
        }
    }
    /// Displays information on a node [debug]
    pub fn display_node(&self, id: usize) -> String {
        if cfg!(debug_assertions) {
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
                self.nodes[id].value,
                self.nodes[id].visits,
                self.nodes[id].total_action_value,
                self.nodes[id].policy,
                self.nodes[id].get_q_val(self.settings),
                u,
                puct,
                self.nodes[id].children.len(),
                self.nodes[id].wdl.w - self.nodes[id].wdl.l,
                self.nodes[id].wdl.w,
                self.nodes[id].wdl.d,
                self.nodes[id].wdl.l,
            )
        } else {
            String::new()
        }
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

#[derive(PartialEq, Clone, Debug)]
pub struct Node {
    pub parent: Option<usize>,
    pub children: Range<usize>,
    pub policy: f32,
    pub visits: u32,
    pub value: f32, // -1 for black and 1 for white
    pub total_action_value: f32,
    pub wdl: Wdl,
    pub total_wdl: Wdl,
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

impl Wdl {
    /// utility function that inverts the winning and losing probabilities
    pub fn flip(self) -> Self {
        Wdl {
            w: self.l,
            d: self.d,
            l: self.w,
        }
    }
}

impl std::ops::AddAssign for Wdl {
    fn add_assign(&mut self, rhs: Self) {
        self.w += rhs.w;
        self.d += rhs.d;
        self.l += rhs.l;
    }
}

impl Node {
    pub fn get_q_val(&self, settings: SearchSettings) -> f32 {
        let fpu = settings.fpu; // First Player Urgency
        if self.visits > 0 {
            let total = match settings.wdl {
                EvalMode::Wdl => self.total_wdl.w - self.total_wdl.l,
                EvalMode::Value => self.total_action_value,
            };
            total / (self.visits as f32)
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
                    let m_clipped = m.clamp(-weights.moves_left_clip, weights.moves_left_clip);
                    (weights.moves_left_sharpness * m_clipped * -q).clamp(-1.0, 1.0)
                }; // tries to speed up the game if winning and vice versa

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
            value: f32::NAN,
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

    /// recursively prints the tree containing information about each node [debug]
    pub fn layer_p(&self, depth: u8, max_tree_print_depth: u8, tree: &Tree) {
        if cfg!(debug_assertions) {
            let indent = "    ".repeat(depth as usize + 2);
            if depth <= max_tree_print_depth {
                if !self.children.is_empty() {
                    for c in self.children.clone() {
                        let display_str = tree.display_node(c);

                        debug_print!("{}", &format!("{}{}", indent, display_str));
                        tree.nodes[c].layer_p(depth + 1, max_tree_print_depth, tree);
                    }
                }
            }
        }
    }

    /// prints the entire tree with all node information [debug]
    pub fn display_full_tree(&self, tree: &Tree) {
        if cfg!(debug_assertions) {
            println!("yoo");
            debug_print!("{}", &format!("        root node:"));
            let display_str = tree.display_node(0);
            debug_print!("{}", &format!("            {}", display_str));
            debug_print!("{}", &format!("        children:"));
            let max_tree_print_depth: u8 = 3;
            debug_print!("{}", &format!("    {}", display_str));
            self.layer_p(0, max_tree_print_depth, tree);
        }
    }
}

/// computes the best move using MCTS with nn given the number of total MCTS iteration. (handles tree creation as well). 
/// to reuse cache simply pass a mutable reference `&mut LruCache<CacheEntryKey, CacheEntryValue>` while repeatedly calling the function
pub async fn get_move(
    bs: BoardStack,
    tensor_exe_send: &Sender<Packet>,
    settings: SearchSettings,
    id: usize,
    mut cache: &mut LruCache<CacheEntryKey, CacheEntryValue>,
) -> (
    Move,
    ZeroEvaluation,
    Option<Vec<usize>>,
    ZeroEvaluation,
    u32,
) {
    let sw_uci = Instant::now(); // timer for uci
    let thread_name = format!("mcts-{}", id);
    //  debug_print(&format!("{:?}", &bs));
    let mut tree = Tree::new(bs, settings);
    if tree.board.is_terminal() {
        panic!("No valid move!/Board is already game over!");
    }

    // let search_type = TypeRequest::TrainerSearch;
    while tree.nodes[0].visits < settings.max_nodes as u32 {
        debug_print!("{}", &format!("step {}", tree.nodes[0].visits));
        debug_print!(
            "{}",
            &format!("thread {}, step {}", thread_name, tree.nodes[0].visits)
        );

        let get_move_debugger = TimeStampDebugger::create_debug();

        tree.step(&tensor_exe_send, sw_uci, id, cache).await;

        if id % 512 == 0 {
            get_move_debugger.record("mcts_step", &thread_name);
        }
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
    debug_print!("{}", &format!("{:#}", best_move.unwrap()));
    for child in tree.nodes[0].children.clone() {
        total_visits_list.push(tree.nodes[child].visits);
    }

    let display_str = tree.display_node(0); // print root node
    debug_print!("{}", &format!("{}", display_str));
    let total_visits: u32 = total_visits_list.iter().sum();

    let mut pi: Vec<f32> = Vec::new();

    debug_print!("{}", &format!("{:?}", &total_visits_list));

    for &t in &total_visits_list {
        let prob = t as f32 / total_visits as f32;
        pi.push(prob);
    }

    debug_print!("{}", &format!("{:?}", &pi));
    debug_print!("{}", &format!("{}", best_move.expect("Error").to_string()));
    debug_print!(
        "{}",
        &format!("best move: {}", best_move.expect("Error").to_string())
    );

    for child in tree.nodes[0].children.clone() {
        let display_str = tree.display_node(child);
        debug_print!("{}", &format!("{}", display_str));
    }
    tree.nodes[0].display_full_tree(&tree);

    let mut all_pol = Vec::new();

    for child in tree.nodes[0].clone().children {
        all_pol.push(tree.nodes[child].policy);
    }

    let v_p = ZeroEvaluation {
        // network evaluation, NOT search/empirical data
        values: tree.nodes[0].value,
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
