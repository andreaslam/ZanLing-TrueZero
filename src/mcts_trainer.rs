use crate::{
    boardmanager::BoardStack,
    cache::CacheEntryKey,
    dataformat::{ZeroEvaluationAbs, ZeroValuesAbs, ZeroValuesPov},
    debug_print,
    decoder::{convert_board, extract_policy, process_board_output},
    dirichlet::StableDirichlet,
    executor::{Packet, ReturnMessage},
    mvs::get_contents,
    settings::SearchSettings,
    uci::{check_castling_move_on_output, eval_in_cp},
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
    time::Instant,
};
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
        maybe_init_cuda();
        let device = if has_cuda() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        let mut net = tch::CModule::load_on_device(path, device).expect("ERROR");
        net.set_eval();

        Self { net, device }
    }
    /// creates a new `Net` instance with a specified device ID (supports only CUDA)
    pub fn new_with_device_id(path: &str, id: usize) -> Self {
        maybe_init_cuda();
        let device = if has_cuda() {
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
        // let device = Device::Cpu;
        let mut net = tch::CModule::load_on_device(path, device).expect("ERROR");
        net.set_eval();

        Self { net, device }
    }
}

#[derive(Debug)]
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
        tensor_exe_send: &Sender<Packet>, // sender to send tensors to `executor.rs`, which gathers MCTS simulations to execute in a batch
        sw: Instant,                      // timer for UCI
        id: usize,                        // unique ID assigned to each thread for debugging
        cache: &mut LruCache<CacheEntryKey, ZeroEvaluationAbs>,
    ) {
        let thread_name = format!("generator-{}", id);
        let step_debugger = TimeStampDebugger::create_debug();

        let (selected_node, input_b, (min_depth, max_depth)) = self.select();
        if id % 512 == 0 {
            step_debugger.record("mcts select", &thread_name);
        }

        // self.nodes[0].display_full_tree(self);

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
                    self.nodes[selected_node].net_evaluation = packet.values;

                    // retrieve policy data and children

                    let contents = get_contents(); // this extracts the mapping for policy nodes according to `mvs.rs`
                    let (_, idx_li) = extract_policy(&input_b, contents); // filters the correct indices of policy nodes according to current board legal moves
                    let mut legal_moves: Vec<Move> = Vec::new();
                    input_b.board().generate_moves(|moves| {
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

                    idx_li // returns idx_li, which is used for indexing legal moves
                }
                None => {
                    self.eval_and_expand(&selected_node, &input_b, tensor_exe_send, id, cache)
                        .await // if there are no corresponding entries in the cache, request a nn evaluation
                }
            };

            self.nodes[selected_node].move_idx = Some(idx_li);
            let mut legal_moves: Vec<Move>;
            if selected_node == 0 {
                legal_moves = Vec::new();
                self.board.board().generate_moves(|moves| {
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
                if let TypeRequest::TrainerSearch(_) = self.settings.search_type {
                    let mut std_rng = StdRng::from_entropy();
                    let distr = StableDirichlet::new(self.settings.alpha, legal_moves.len())
                        .expect("wrong params");
                    let sample = std_rng.sample(distr);
                    for child in self.nodes[0].children.clone() {
                        self.nodes[child].policy = (1.0 - self.settings.eps)
                            * self.nodes[child].policy
                            + (self.settings.eps * sample[child - 1]);
                    }
                }
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
            debug_print!("terminal! {:#} {:?}", input_b.board(), wdl);
            self.nodes[selected_node].net_evaluation.value = wdl.w - wdl.l;
            self.nodes[selected_node].net_evaluation.wdl = wdl;
            self.nodes[selected_node].net_evaluation.moves_left = 0.0; //TODO find alternative
        }

        self.backpropagate(selected_node);
        let backprop_debug = TimeStampDebugger::create_debug();
        if id % 512 == 0 {
            backprop_debug.record("backpropagation", &thread_name);
        }

        debug_print!("root node: {}", self.display_node(0));

        debug_print!("    all children:");
        for _child in self.nodes[0].children.clone() {
            debug_print!("        {}", &self.display_node(_child));
        }

        if let TypeRequest::UCISearch = self.settings.search_type {
            let cp_eval = eval_in_cp(self.nodes[selected_node].net_evaluation.value);
            let elapsed_ms = sw.elapsed().as_nanos() as f32 / 1e6;
            let nps = self.nodes[0].visits as f32 / (sw.elapsed().as_nanos() as f32 / 1e9_f32);
            let (pv, mate) = self.get_pv();
            let eval_string = {
                if mate {
                    let mut mate_score =
                        pv.split_whitespace().collect::<Vec<&str>>().len() as isize;
                    if cp_eval < 0.0 {
                        mate_score *= -1;
                    }
                    format!("mate {}", (mate_score as f32 / 2.0).ceil())
                } else {
                    format!(
                        "cp {}",
                        (cp_eval * 100.).round().max(-1000.).min(1000.) as i64,
                    )
                }
            };
            if self.pv != pv {
                println!(
                    "info depth {} seldepth {} score {} nodes {} nps {} hashfull {} time {} pv {}",
                    min_depth,
                    max_depth,
                    eval_string,
                    self.nodes.len(),
                    nps as usize,
                    ((cache.len() as f32 / cache.cap().get() as f32) * 1000.0) as usize,
                    elapsed_ms as usize,
                    pv,
                );
                self.pv = pv;
            }
        }
    }

    fn get_pv(&self) -> (String, bool) {
        let mut pv_nodes: Vec<usize> = vec![];
        let mut curr_node = 0;
        let mut terminal = false; // by default assume the pv has not reached mate
        let mut pv_board = self.board.clone();
        loop {
            if self.nodes[curr_node].children.is_empty() || pv_board.is_terminal() {
                if pv_board.status() == GameStatus::Won {
                    terminal = true;
                }
                break;
            }
            curr_node = self.nodes[curr_node]
                .children
                .clone()
                .max_by(|a, b| {
                    let a_node = &self.nodes[*a];
                    let b_node = &self.nodes[*b];
                    let a_visits = a_node.visits;
                    let b_visits = b_node.visits;

                    if a_visits == b_visits || self.nodes[curr_node].visits == 0 {
                        let a_policy = a_node.policy;
                        let b_policy = b_node.policy;
                        a_policy.partial_cmp(&b_policy).unwrap()
                    } else {
                        a_visits.partial_cmp(&b_visits).unwrap()
                    }
                })
                .expect("Error");
            pv_board.play(self.nodes[curr_node].mv.unwrap());
            pv_nodes.push(curr_node);
        }
        let mut pv_string: String = String::new();
        let mut pv_string_board = self.board.clone();
        if pv_nodes.is_empty() {
            (pv_string, terminal)
        } else {
            for item in pv_nodes {
                let pv_move =
                    check_castling_move_on_output(&pv_string_board, self.nodes[item].mv.unwrap());
                pv_string.push_str(&format!("{} ", pv_move));
                pv_string_board.play(self.nodes[item].mv.unwrap());
            }
            (pv_string, terminal)
        }
    }

    pub fn depth_range(&self, node: usize) -> (usize, usize) {
        if let TypeRequest::UCISearch = self.settings.search_type {
            match self.nodes[node].children.len() {
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
            }
        } else {
            (0, 0)
        }
    }

    /// selects the node to expand based on the PUCT formula/policy score if the visit count is 0
    pub fn select(&mut self) -> (usize, BoardStack, (usize, usize)) {
        let mut curr: usize = 0;
        let mut visited_nodes = Vec::new();
        debug_print!("{}", &"    selection:".to_string());
        let mut input_b: BoardStack = self.board.clone();
        debug_print!("{}", &format!("        board FEN: {}", &input_b.board()));
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
                total_visits += self.nodes[child].visits;
            }
            (_, max_depth) = self.depth_range(curr);
            curr = children
                .clone()
                .max_by(|a, b| {
                    let a_node = &self.nodes[*a];
                    let b_node = &self.nodes[*b];
                    let a_puct = a_node.puct_formula(
                        curr_node.visits,
                        curr_node.net_evaluation.moves_left,
                        input_b.board().side_to_move(),
                        self.settings,
                    );
                    let b_puct = b_node.puct_formula(
                        curr_node.visits,
                        curr_node.net_evaluation.moves_left,
                        input_b.board().side_to_move(),
                        self.settings,
                    );

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

            assert!(total_visits + 1 == curr_node.visits);
            if !visited_nodes.contains(&curr) {
                visited_nodes.push(curr);
                debug_print!(
                    "{}",
                    &format!("        selected: {}", self.display_node(curr))
                );
            }
            input_b.play(self.nodes[curr].mv.expect("Error"));
            depth += 1;
        }

        debug_print!("{}", &format!("    {}", self.display_node(curr)));
        debug_print!("{}", &"        children:".to_string());

        for _node_id in visited_nodes {
            debug_print!(
                "{}",
                &format!("        node: {}", self.display_node(_node_id))
            );
        }

        (curr, input_b, (depth, max_depth))
    }

    /// evaluates selected node using the neural network by sending the input tensors to `executor.rs` through async sender
    /// then the results are sent back and calls `decoder.rs` to process the nn ouputs and update the tree
    async fn eval_and_expand(
        &mut self,
        selected_node_idx: &usize,
        bs: &BoardStack,
        tensor_exe_send: &Sender<Packet>,
        _id: usize,
        cache: &mut LruCache<CacheEntryKey, ZeroEvaluationAbs>,
    ) -> Vec<usize> {
        debug_print!("    eval_and_expand:");

        let input_tensor = convert_board(bs);

        let (resender_send, resender_recv) = flume::bounded::<ReturnMessage>(1);
        let thread_name = std::thread::current()
            .name()
            .unwrap_or("unnamed-generator")
            .to_owned();
        let pack = Packet {
            job: input_tensor,
            resender: resender_send,
            id: thread_name.clone(),
        };

        tensor_exe_send.send_async(pack).await.unwrap();

        let output = resender_recv.recv_async().await.unwrap();
        let output = match output {
            ReturnMessage::ReturnMessage(Ok(output)) => output,
            ReturnMessage::ReturnMessage(Err(_)) => panic!("error in returning!"),
        };

        assert!(thread_name == output.id);
        let idx_li = process_board_output(
            (&output.packet.0, &output.packet.1),
            selected_node_idx,
            self,
            bs,
            cache,
        );
        debug_print!("        expanded {}", self.display_node(*selected_node_idx));
        idx_li
    }

    /// backpropagates the evaluation results up the tree
    pub fn backpropagate(&mut self, node: usize) {
        debug_print!("{}", &"    backpropagation:".to_string());
        let mut curr: Option<usize> = Some(node);
        let mut backprop_nodes_vec: Vec<usize> = Vec::new(); // keep track of the nodes used for backpropagation, for debug
        let mut num_backprop_times = 1.0;
        let current_node_net_eval = self.nodes[node].net_evaluation;
        while let Some(current) = curr {
            self.nodes[current].visits += 1;
            self.nodes[current].total_evaluation += current_node_net_eval;
            debug_print!(
                "{:?} {:?}",
                self.nodes[current].total_evaluation,
                current_node_net_eval
            );
            self.nodes[current].total_evaluation.moves_left += num_backprop_times;
            backprop_nodes_vec.push(current);
            curr = self.nodes[current].parent;
            num_backprop_times += 1.0;
        }

        for _current in backprop_nodes_vec {
            debug_print!(
                "{}",
                &format!("        updated node to {}", self.display_node(_current))
            );
        }
    }

    /// Displays information on a node (debug)
    pub fn display_node(&self, id: usize) -> String {
        if cfg!(debug_assertions) {
            let u: f32;
            let puct: f32;

            // get moves
            let mut curr: Option<usize> = Some(id);
            let mut bs_clone = self.board.clone();
            let mut mv_vec: Vec<Move> = Vec::new();
            while let Some(current) = curr {
                if let Some(mv) = self.nodes[current].mv {
                    mv_vec.push(mv)
                }
                curr = self.nodes[current].parent;
            }
            mv_vec.reverse();
            for mv in mv_vec {
                bs_clone.play(mv);
            }

            match &self.nodes[id].parent {
                Some(parent) => {
                    if self.nodes[*parent].visits == 0 {
                        u = f32::NAN;
                        puct = f32::NAN;
                    } else {
                        u = self.nodes[id].get_u_val(self.nodes[*parent].visits, self.settings);
                        puct = self.nodes[id].puct_formula(
                            self.nodes[*parent].visits,
                            self.nodes[*parent].net_evaluation.moves_left,
                            !bs_clone.board().side_to_move(),
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
            let relative_evaluation = self.nodes[id]
                .total_evaluation
                .to_relative(!bs_clone.board().side_to_move());
            format!(
                "Node(action= {}, V= {}, N={}, W={}, P={}, Q={}, U={}, PUCT={}, len_children={}, wdl={}, w={}, d={}, l={}, M={}, M_total={})",
                mv_n,
                self.nodes[id].net_evaluation.value,
                self.nodes[id].visits,
                self.nodes[id].total_evaluation.value,
                self.nodes[id].policy,
                self.nodes[id].get_q_val(self.settings, relative_evaluation),
                u,
                puct,
                self.nodes[id].children.len(),
                self.nodes[id].net_evaluation.wdl.w - self.nodes[id].net_evaluation.wdl.l,
                self.nodes[id].net_evaluation.wdl.w,
                self.nodes[id].net_evaluation.wdl.d,
                self.nodes[id].net_evaluation.wdl.l,
                self.nodes[id].net_evaluation.moves_left,
                self.nodes[id].total_evaluation.moves_left/self.nodes[id].visits as f32,
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

#[derive(Clone, Debug)]
pub struct Node {
    pub parent: Option<usize>,
    pub children: Range<usize>,
    pub policy: f32,
    pub visits: u32,
    pub net_evaluation: ZeroValuesAbs,
    pub total_evaluation: ZeroValuesAbs,
    pub mv: Option<Move>,
    pub move_idx: Option<Vec<usize>>,
}

impl Node {
    pub fn new(policy: f32, parent: Option<usize>, mv: Option<Move>) -> Node {
        Node {
            parent,
            children: 0..0,
            policy,
            visits: 0,
            net_evaluation: ZeroValuesAbs::nan(),
            total_evaluation: ZeroValuesAbs::zeros(),
            mv,
            move_idx: None,
        }
    }

    pub fn get_q_val(&self, settings: SearchSettings, relative_evaluation: ZeroValuesPov) -> f32 {
        let fpu = settings.fpu;
        if self.visits > 0 {
            let total = match settings.wdl {
                EvalMode::Wdl => relative_evaluation.wdl.w - relative_evaluation.wdl.l,
                EvalMode::Value => relative_evaluation.value,
            };
            total / (self.visits as f32)
        } else {
            match self.parent {
                Some(_) => fpu.children_fpu.unwrap_or_else(|| fpu.default()),
                None => fpu.root_fpu.unwrap_or_else(|| fpu.default()),
            }
        }
    }

    pub fn get_u_val(&self, parent_visits: u32, settings: SearchSettings) -> f32 {
        let c_puct = match self.parent {
            Some(_) => settings
                .c_puct
                .children_c_puct
                .unwrap_or_else(|| settings.c_puct.default()),
            None => settings
                .c_puct
                .root_c_puct
                .unwrap_or_else(|| settings.c_puct.default()),
        };

        c_puct * self.policy * ((parent_visits - 1) as f32).sqrt() / (1.0 + self.visits as f32)
    }

    pub fn puct_formula(
        &self,
        parent_visits: u32,
        parent_moves_left: f32,
        player: Color,
        settings: SearchSettings,
    ) -> f32 {
        assert!(self.visits < parent_visits);

        let relative_evaluation = self.total_evaluation.to_relative(player);
        let u = self.get_u_val(parent_visits, settings);
        let q = self.get_q_val(settings, relative_evaluation);
        if let Some(weights) = settings.moves_left {
            let m = if self.visits == 0 {
                0.0
            } else {
                (relative_evaluation.moves_left / self.visits as f32) - (parent_moves_left - 1.0)
            };

            let m_unit = if weights.moves_left_weight == 0.0 {
                0.0
            } else {
                let m_clipped = m.clamp(-weights.moves_left_clip, weights.moves_left_clip);
                (weights.moves_left_sharpness * m_clipped * -q).clamp(-1.0, 1.0)
            };

            q + u + weights.moves_left_weight * m_unit
        } else {
            q + u
        }
    }

    /// recursively prints the tree containing information about each node (debug)
    pub fn layer_p(&self, depth: u8, max_tree_print_depth: u8, tree: &Tree) {
        if cfg!(debug_assertions) {
            let _indent = "    ".repeat(depth as usize + 2);
            if depth <= max_tree_print_depth && !self.children.is_empty() {
                for c in self.children.clone() {
                    debug_print!("{}", &format!("{}{}", _indent, tree.display_node(c)));
                    tree.nodes[c].layer_p(depth + 1, max_tree_print_depth, tree);
                }
            }
        }
    }

    /// prints the entire tree with all node information (debug)
    pub fn display_full_tree(&self, tree: &Tree) {
        if cfg!(debug_assertions) {
            debug_print!("{}", &"        root node:".to_string());
            debug_print!("{}", &format!("            {}", tree.display_node(0)));
            debug_print!("{}", &"        children:".to_string());
            let max_tree_print_depth: u8 = 3;
            debug_print!("{}", &format!("    {}", tree.display_node(0)));
            self.layer_p(0, max_tree_print_depth, tree);
        }
    }
}

#[derive(PartialEq, Clone, Debug, Copy, Default)]
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
    /// utility function that returns a `Wdl` struct with all values initialised as `f32::NAN`
    pub fn nan() -> Self {
        Wdl {
            w: f32::NAN,
            d: f32::NAN,
            l: f32::NAN,
        }
    }
    /// utility function that returns a `Wdl` struct with a win/loss value of 0.
    pub fn zeros() -> Self {
        Wdl {
            w: 0.0,
            d: 0.0,
            l: 0.0,
        }
    }
    pub fn get_average(self, visits: u32) -> Self {
        Self {
            w: self.w / visits as f32,
            d: self.d / visits as f32,
            l: self.l / visits as f32,
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

/// computes the best move using MCTS with nn given the number of total MCTS iteration. (handles tree creation as well).
/// to reuse cache simply pass a mutable reference `&mut LruCache<CacheEntryKey, ZeroEvaluationAbs>` while repeatedly calling the function
pub async fn get_move(
    bs: BoardStack,
    tensor_exe_send: &Sender<Packet>,
    settings: SearchSettings,
    id: usize,
    cache: &mut LruCache<CacheEntryKey, ZeroEvaluationAbs>,
) -> (
    Move,
    ZeroEvaluationAbs,
    Option<Vec<usize>>,
    ZeroEvaluationAbs,
    u32,
) {
    let sw_uci = Instant::now();
    let thread_name = format!("generator-{}", id);
    let mut tree = Tree::new(bs, settings);
    if tree.board.is_terminal() {
        panic!("No valid move!/Board is already game over!");
    }
    match settings.max_nodes {
        Some(max_nodes) => {
            while tree.nodes[0].visits < max_nodes as u32 {
                debug_print!("{}", &format!("step {}", tree.nodes[0].visits));
                debug_print!(
                    "{}",
                    &format!("thread {}, step {}", thread_name, tree.nodes[0].visits)
                );

                let get_move_debugger = TimeStampDebugger::create_debug();

                tree.step(tensor_exe_send, sw_uci, id, cache).await;

                if id % 512 == 0 {
                    get_move_debugger.record("mcts_step", &thread_name);
                }
            }
        }
        None => {
            panic!("`max_nodes` is None! Must have a non-zero Some(u128) for data generation!")
        }
    }

    let mut child_visits: Vec<u32> = Vec::new();

    for child in tree.nodes[0].children.clone() {
        child_visits.push(tree.nodes[child].visits);
    }

    let all_same_visits = child_visits.iter().all(|&x| x == child_visits[0]);

    let best_move_node = if !all_same_visits {
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
        debug_print!("{}", &tree.display_node(child));
    }

    debug_print!("{}", &tree.display_node(0));
    let total_visits: u32 = total_visits_list.iter().sum();

    let mut pi: Vec<f32> = Vec::new();

    debug_print!("{}", &format!("{:?}", &total_visits_list));

    for &t in &total_visits_list {
        let prob = t as f32 / total_visits as f32;
        pi.push(prob);
    }

    debug_print!("{}", &format!("{:?}", &pi));
    debug_print!("{}", &format!("{}", best_move.expect("Error")));
    debug_print!("{}", &format!("best move: {}", best_move.expect("Error")));

    for _child in tree.nodes[0].children.clone() {
        debug_print!("{}", tree.display_node(_child));
    }

    // tree.nodes[0].display_full_tree(&tree);

    let mut all_tree_pol = Vec::new();

    for child in tree.nodes[0].clone().children {
        all_tree_pol.push(tree.nodes[child].policy);
    }

    let net_evaluation = ZeroEvaluationAbs {
        // network evaluation, NOT search/empirical data
        values: tree.nodes[0].net_evaluation,
        policy: all_tree_pol,
    };

    let search_data = ZeroEvaluationAbs {
        // empirical search data
        values: tree.nodes[0].total_evaluation,
        policy: pi,
    }
    .get_average(tree.nodes[0].visits);

    (
        best_move.expect("Error"),
        net_evaluation,
        tree.nodes[0].clone().move_idx,
        search_data,
        tree.nodes[0].visits,
    )
}
