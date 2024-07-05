use crate::{
    boardmanager::BoardStack,
    cache::{CacheEntryKey, CacheEntryValue},
    mcts_trainer::{Net, Node, Tree, Wdl},
    mvs::get_contents,
    utils::debug_print,
};
use cozy_chess::{Color, Move, Piece, Rank, Square};
use lru::LruCache;
use tch::{Cuda, Device, IValue, Kind, Tensor};

pub fn eval_state(board: Tensor, net: &Net) -> anyhow::Result<(Tensor, Tensor)> {
    let b = board.reshape([-1, 21, 8, 8]);
    let b: Tensor = b.to(net.device);
    let board = IValue::Tensor(b);

    let output = net.net.forward_is(&[board])?;

    let output_tensor = match output {
        IValue::Tuple(b) => b,
        a => panic!("the output is not an IValue {:?}", a),
    };

    let (board_eval, policy) = (&output_tensor[0], &output_tensor[1]);
    let board_eval = match board_eval {
        IValue::Tensor(b) => b,
        a => panic!("the output is not a Tensor {:?}", a),
    };
    let policy = match policy {
        IValue::Tensor(b) => b,
        a => panic!("the output is not a Tensor {:?}", a),
    };

    // Move the results to CPU to reduce GPU memory usage
    let board_eval = board_eval.to(Device::Cpu);
    let policy = policy.to(Device::Cpu);

    // Drop the original GPU tensors to free memory
    drop(output_tensor);

    Ok((board_eval, policy))
}

pub fn board_data(bs: &BoardStack) -> (Vec<f32>, Vec<bool>) {
    let us = bs.board().side_to_move();
    let mut scalar_data = vec![0.0; 8];
    if bs.board().side_to_move() == Color::Black {
        scalar_data[1] = 1.0;
    } else {
        scalar_data[0] = 1.0;
    }

    let li;
    if us == Color::White {
        // can't use Colour::ALL since the order of Colour::ALL is always going to be [white, black]
        li = [Color::White, Color::Black];
    } else {
        li = [Color::Black, Color::White];
    }

    let mut c = 2;

    for color in li {
        let l_rights = bs.board().castle_rights(color).long;
        let s_rights = bs.board().castle_rights(color).short;

        scalar_data[c] = if l_rights.is_some() { 1.0 } else { 0.0 };
        scalar_data[c + 1] = if s_rights.is_some() { 1.0 } else { 0.0 };
        c += 2
    }

    scalar_data[6] = bs.get_reps() as f32;
    scalar_data[7] = bs.board().halfmove_clock() as f32;

    let mut counter = 0;
    let mut pieces_sqs: Vec<bool> = vec![false; 64 * 12];
    for colour in li {
        for piece in Piece::ALL {
            for tile in bs.board().colored_pieces(colour, piece) {
                if li[0] == Color::Black {
                    pieces_sqs[(63 - (tile.rank() as usize * 8 + (7 - tile.file() as usize)))
                        + (64 * counter)] = true;
                } else {
                    pieces_sqs[(tile.rank() as usize * 8 + tile.file() as usize) + 64 * counter] =
                        true;
                }
            }
            counter += 1
        }
    }

    let is_ep = bs.board().en_passant();
    let fenstr = format!("{}", bs.board());
    debug_print(&format!("    board FEN: {}", fenstr));
    debug_print(&format!("En passant status: {:?}", is_ep));
    let mut sq21: Vec<bool> = vec![false; 64];
    match is_ep {
        Some(is_ep) => {
            if us == Color::White {
                // 4 for white and 5 for black for victim
                let row = Rank::Fourth;
                let ep_sq = Square::new(is_ep, row);
                sq21[ep_sq.rank() as usize * 8 + ep_sq.file() as usize] = true;
            } else {
                let row = Rank::Fifth;
                let ep_sq = Square::new(is_ep, row);
                sq21[63 - (ep_sq.rank() as usize * 8 + (7 - ep_sq.file() as usize))] = true;
            }
        }
        None => {}
    };
    pieces_sqs.extend(sq21);
    (scalar_data, pieces_sqs)
}

pub fn convert_board(bs: &BoardStack) -> Tensor {
    // FULL LIST HERE:
    // sq1 - white's turn
    // sq2 - black's turn
    // sq3, sq4 - castling pos l + r (us)
    // sq5, sq6 - castling pos l + r (opponent)
    // sql7, sql8 -  sqs for bits for the repetition counter
    // sq9 - sq20 - sqs for turn to move + non-turn to move's pieces
    // sq21 - en passant square if any

    // sq1 - white's turn
    // sq2 - black's turn

    // it seems that creating a Vec, processing everything first is faster than doing Tensor::zeros() and then stacking them
    // so i instead work with Vecs, get all of them together and convert them into a single Tensor at the end

    let mut all_data: Vec<f32> = Vec::new();

    let (scalar_data, pieces_sqs) = board_data(bs);

    for v in &scalar_data {
        for _ in 0..64 {
            all_data.push(*v);
        }
    }

    all_data.extend(pieces_sqs.iter().map(|&x| x as u8 as f32));

    let all_data = Tensor::from_slice(&all_data);
    all_data // all_data is 1d
}

pub fn get_evaluation(bs: &BoardStack, net: &Net) -> (Tensor, Tensor) {
    let b = convert_board(bs);

    eval_state(b, &net).expect("Error")
}

pub fn process_board_output(
    output: (&Tensor, &Tensor),
    selected_node_idx: &usize,
    tree: &mut Tree,
    bs: &BoardStack,
    mut cache: &mut LruCache<CacheEntryKey, CacheEntryValue>,
) -> Vec<usize> {
    let contents = get_contents();
    let (board_eval, policy) = output; // check policy, eval ordering!
    let board_eval = board_eval.squeeze();

    let board_evals: Vec<f32> = Vec::try_from(board_eval).expect("Error");

    let value: f32 = board_evals[0].tanh();

    let wdl_logits: Tensor = Tensor::from_slice(&board_evals[1..4]);

    let wdl = Tensor::softmax(&wdl_logits, 0, Kind::Float);

    let wdl: Vec<f32> = Vec::try_from(wdl).expect("Error");
    let wdl = Wdl {
        w: wdl[0],
        d: wdl[1],
        l: wdl[2],
    };
    let moves_left = board_evals[4];

    let policy = policy.squeeze();
    let policy: Vec<f32> = Vec::try_from(policy).expect("Error");
    let value = f32::try_from(value).expect("Error");

    let (value, wdl) = match bs.board().side_to_move() {
        Color::Black => (-value, wdl.flip()),
        Color::White => (value, wdl),
    };

    // step 1 - get the corresponding idx for legal moves

    let (legal_moves, idx_li) = extract_policy(bs, contents);

    // step 2 - using the idx in step 1, index all the policies involved
    let mut pol_list: Vec<f32> = Vec::new();
    for id in &idx_li {
        pol_list.push(policy[*id]);
    }

    debug_print(&format!("{:?}", pol_list));

    // step 3 - softmax

    let sm = Tensor::from_slice(&pol_list);

    let sm = Tensor::softmax(&sm, 0, Kind::Float);

    let pol_list: Vec<f32> = Vec::try_from(sm).expect("Error");

    debug_print(&format!("{:?}", pol_list));

    debug_print(&format!("        V={}", &value));

    debug_print(&format!("        Value={}", &value));

    // step 4 - iteratively append nodes into class
    let mut counter = 0;

    tree.nodes[*selected_node_idx].moves_left = moves_left;

    tree.nodes[*selected_node_idx].value = value;

    tree.nodes[*selected_node_idx].wdl = wdl;

    let ct = tree.nodes.len();
    for (mv, pol) in legal_moves.iter().zip(pol_list.iter()) {
        debug_print(&format!("value {}", value));
        let fm: Move;
        if bs.board().side_to_move() == Color::Black {
            // flip move
            fm = Move {
                from: mv.from.flip_rank(),
                to: mv.to.flip_rank(),
                promotion: mv.promotion,
            };
        } else {
            fm = *mv;
        }
        let child = Node::new(*pol, Some(*selected_node_idx), Some(fm));
        debug_print(&format!("{:?}, {:?}, {:?}", mv, child.policy, child.value));
        tree.nodes.push(child); // push child to the tree Vec<Node>
        counter += 1
    }
    tree.nodes[*selected_node_idx].children = ct..ct + counter; // push numbers
    debug_print(&format!("{:?}", tree.nodes.len()));

    cache.put(
        CacheEntryKey {
            hash: bs.board().hash(),
            halfmove_clock: bs.board().halfmove_clock(),
        },
        CacheEntryValue {
            eval_score: value,
            policy: pol_list,
            moves_left,
            wdl,
            mv: tree.nodes[*selected_node_idx].mv,
        },
    );

    idx_li
}

pub fn extract_policy(bs: &BoardStack, contents: &'static [Move]) -> (Vec<Move>, Vec<usize>) {
    let mut legal_moves: Vec<Move> = Vec::new();
    bs.board().generate_moves(|moves| {
        // Unpack dense move set into move list
        legal_moves.extend(moves);
        false
    });

    let mut fm: Vec<Move> = Vec::new();
    if bs.board().side_to_move() == Color::Black {
        // flip move
        for mv in &legal_moves {
            fm.push(Move {
                from: mv.from.flip_rank(),
                to: mv.to.flip_rank(),
                promotion: mv.promotion,
            })
        }
    } else {
        fm = legal_moves.clone();
    }

    legal_moves = fm;

    let mut idx_li: Vec<usize> = Vec::new();

    for mov in &legal_moves {
        if let Some(idx) = contents.iter().position(|x| mov == x) {
            idx_li.push(idx as usize);
        }
    }

    (legal_moves, idx_li)
}
