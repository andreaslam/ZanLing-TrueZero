use crate::boardmanager::BoardStack;
use crate::mcts_trainer::{Node, Tree};
use crate::{mcts_trainer::Net, mvs::get_contents};
use cozy_chess::{Color, Move, Piece, Rank, Square};
use tch::{IValue, Kind, Tensor};

pub fn eval_state(board: Tensor, net: &Net) -> anyhow::Result<(Tensor, Tensor)> {
    // reshape the model (originally from 1D)
    let b = board;
    let b = b.unsqueeze(0);
    let b = b.reshape([-1, 21, 8, 8]);
    // println!("{:?}", b.size());
    let b: Tensor = b.to(net.device);
    let board = IValue::Tensor(b);
    let output = net.net.forward_is(&[board])?;
    let output_tensor = match output {
        IValue::Tuple(b) => b,
        a => panic!("the output is not a TensorList {:?}", a),
    };
    let (board_eval, policy) = (&output_tensor[0], &output_tensor[1]);
    let board_eval = match board_eval {
        IValue::Tensor(b) => b,
        a => panic!("the output is not a TensorList {:?}", a),
    };
    let policy = match policy {
        IValue::Tensor(b) => b,
        a => panic!("the output is not a TensorList {:?}", a),
    };
    Ok((board_eval.clone(board_eval), policy.clone(policy)))
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

    // flatten to 1d

    let mut counter = 0;
    let mut pieces_sqs: Vec<f32> = vec![0.0; 64 * 12];
    for colour in li {
        for piece in Piece::ALL {
            for tile in bs.board().colored_pieces(colour, piece) {
                if li[0] == Color::Black {
                    pieces_sqs[(63 - (tile.rank() as usize * 8 + (7 - tile.file() as usize)))
                        + (64 * counter)] = 1.0;
                } else {
                    pieces_sqs[(tile.rank() as usize * 8 + tile.file() as usize) + 64 * counter] =
                        1.0;
                }
            }
            counter += 1
        }
    }

    let is_ep = bs.board().en_passant();
    // let fenstr = format!("{}", bs.board());
    // println!("    board FEN: {}", fenstr);
    // println!("En passant status: {:?}", is_ep);
    let mut sq21: Vec<f32> = vec![0.0; 64];
    match is_ep {
        Some(is_ep) => {
            if us == Color::White {
                // 4 for white and 5 for black for victim
                let row = Rank::Fourth;
                let ep_sq = Square::new(is_ep, row);
                sq21[ep_sq.rank() as usize * 8 + ep_sq.file() as usize] = 1.0;
            } else {
                let row = Rank::Fifth;
                let ep_sq = Square::new(is_ep, row);
                sq21[63 - (ep_sq.rank() as usize * 8 + (7 - ep_sq.file() as usize))] = 1.0;
            }
        }
        None => {}
    };

    let mut all_data: Vec<f32> = Vec::new();

    for v in &scalar_data {
        for _ in 0..64 {
            all_data.push(*v);
        }
    }

    all_data.extend(pieces_sqs);
    all_data.extend(sq21);

    let all_data = Tensor::from_slice(&all_data);
    all_data // all_data is 1d
}

pub fn eval_board(
    bs: &BoardStack,
    net: &Net,
    tree: &mut Tree,
    selected_node_idx: &usize,
) -> Vec<usize> {
    let output = get_evaluation(bs, net);
    process_board_output(output, selected_node_idx, tree, &bs)
}

fn get_evaluation(bs: &BoardStack, net: &Net) -> (Tensor, Tensor) {
    let b = convert_board(bs);

    eval_state(b, &net).expect("Error")
}

pub fn process_board_output(
    output: (Tensor, Tensor),
    selected_node_idx: &usize,
    tree: &mut Tree,
    bs: &BoardStack,
) -> Vec<usize> {
    let contents = get_contents();
    let (board_eval, policy) = output; // check policy, eval ordering!

    let board_eval = board_eval.squeeze();

    let board_eval: Vec<f32> = Vec::try_from(board_eval).expect("Error");

    let board_eval = Tensor::from_slice(&vec![board_eval[0]]);

    let value = Tensor::tanh(&board_eval);

    let policy = policy.squeeze();
    let policy: Vec<f32> = Vec::try_from(policy).expect("Error");
    let value = f32::try_from(value).expect("Error");

    let value = match bs.board().side_to_move() {
        Color::Black => -value,
        Color::White => value,
    };

    // step 1 - get the corresponding idx for legal moves

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
        // let mov = format!("{}", mov);
        if let Some(idx) = contents.iter().position(|x| mov == x) {
            idx_li.push(idx as usize);
        }
    }

    // step 2 - using the idx in step 1, index all the policies involved
    let mut pol_list: Vec<f32> = Vec::new();
    for id in &idx_li {
        pol_list.push(policy[*id]);
    }

    // println!("{:?}", pol_list);

    // step 3 - softmax

    let sm = Tensor::from_slice(&pol_list);

    let sm = Tensor::softmax(&sm, 0, Kind::Float);

    let pol_list: Vec<f32> = Vec::try_from(sm).expect("Error");

    // println!("{:?}", pol_list);

    // println!("        V={}", &value);

    // step 4 - iteratively append nodes into class
    let mut counter = 0;
    tree.nodes[*selected_node_idx].eval_score = value;
    // tree.nodes[*selected_node_idx].eval_score = 0.0;
    let ct = tree.nodes.len();
    for (mv, pol) in legal_moves.iter().zip(pol_list.iter()) {
        // println!("VAL {}", value);
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
        // FLAT POLICY VER
        // let child = Node::new(1.0/legal_moves.len() as f32, Some(*selected_node_idx), Some(fm));
        let child = Node::new(*pol, Some(*selected_node_idx), Some(fm));
        // println!("{:?}, {:?}, {:?}", mv, child.policy, child.eval_score);
        tree.nodes.push(child); // push child to the tree Vec<Node>
        counter += 1
    }
    tree.nodes[*selected_node_idx].children = ct..ct + counter; // push numbers
                                                                // println!("{:?}", tree.nodes.len());
    idx_li
}
