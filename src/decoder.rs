use crate::boardmanager::BoardStack;
use crate::mvs::get_contents;
use cozy_chess::*;
use std::collections::HashMap;
use std::vec;
use tch::*;

fn eval_state(board: Tensor) -> anyhow::Result<(Tensor, Tensor)> {
    let mut model = tch::CModule::load("chess_16x128_gen3634.pt")?;
    model.set_eval(); // set to eval!
    model.to(Device::cuda_if_available(), Kind::Float, true);
    // reshape the model (originally from 1D)
    let b = board;
    let b = b.unsqueeze(0);
    let b = b.reshape([-1, 21, 8, 8]);
    b.print();
    let b: Tensor = b.to(Device::cuda_if_available());
    let board = IValue::Tensor(b);
    let output = model.forward_is(&[board])?;
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
    // ignore error for bs now
    // FULL LIST HERE:
    // sq1 - white's turn
    // sq2 - black's turn
    // sq3, sq4 - castling pos l + r (us)
    // sq5, sq6 - castling pos l + r (opponent)
    // sql7, sql8 -  sqs for binary digits for the repetition counter
    // sq9 - sq20 - sqs for turn to move + non-turn to move's pieces
    // sq21 - en passant square if any

    // sq1 - white's turn
    // sq2 - black's turn

    // it seems that creating a Vec, processing everything first is faster than doing Tensor::zeros() and then stacking them
    // so i instead work with Vecs, get all of them together and convert them into a single Tensor at the end

    let sq1: Vec<f32>;
    let sq2: Vec<f32>;
    let us = bs.board().side_to_move();
    if bs.board().side_to_move() == Color::Black {
        sq1 = vec![0.0; 64];
        sq2 = vec![1.0; 64];
        // bs = mirror(bs.clone());
    } else {
        sq1 = vec![1.0; 64];
        sq2 = vec![0.0; 64];
    }

    println!("1{:?} \n 2{:?}", sq1, sq2);

    let li;
    if us == Color::White {
        // can't use Colour::ALL since the order of Colour::ALL is always going to be [white, black]
        li = [Color::White, Color::Black];
    } else {
        li = [Color::Black, Color::White];
    }

    let mut scalars: Vec<f32> = Vec::new();

    for color in li {
        let l_rights = bs.board().castle_rights(color).long;
        let s_rights = bs.board().castle_rights(color).short;

        scalars.push(if l_rights.is_some() { 1.0 } else { 0.0 });
        scalars.push(if s_rights.is_some() { 1.0 } else { 0.0 });
    }

    let sq3: Vec<f32> = vec![scalars[0]; 64];
    let sq4: Vec<f32> = vec![scalars[1]; 64];
    let sq5: Vec<f32> = vec![scalars[2]; 64];
    let sq6: Vec<f32> = vec![scalars[3]; 64];

    println!("3{:?} \n 4{:?}", sq3, sq4);

    println!("5{:?} \n 6{:?}", sq5, sq6);
    let num_reps = bs.get_reps() as f32;

    let sq7: Vec<f32> = vec![num_reps; 64];
    let sq8: Vec<f32> = vec![bs.board().halfmove_clock() as f32; 64];
    println!("7{:?} \n 8{:?}", sq7, sq8);
    // flatten to 1d
    let pieces = [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ];

    let mut counter = 0;
    let mut pieces_sqs = vec![0.0; 64 * 12];
    for colour in li {
        for piece in pieces {
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

    // still have to flatten sq
    println!("{:?}", pieces_sqs);
    // println!("{:?}", sq_1d);

    let sq21 = vec![0.0; 64];
    println!("21 {:?}", sq21);
    let all_data = [sq1, sq2, sq3, sq4, sq5, sq6, sq7, sq8, pieces_sqs, sq21];

    let mut sq_1d: Vec<f32> = Vec::new();

    for row in all_data {
        for element in row {
            sq_1d.push(element);
        }
    }

    let all_data = sq_1d;
    let all_data = all_data.to_vec();
    println!("{:?}", all_data);
    let all_data = Tensor::from_slice(&all_data);
    all_data // all_data is 1d
}

pub fn eval_board(bs: &BoardStack) -> (f32, HashMap<cozy_chess::Move, f32>, Vec<usize>) {
    // ignore bigl and model for now, model is the custom net class

    let contents = get_contents();
    let b = convert_board(bs);
    // convert b into [B,21,8,8] first!
    let output = eval_state(b).expect("Error");

    // let output: Vec<Vec<f32>> = Vec::try_from(output).expect("Error");

    let (board_eval, policy) = output; // check policy, eval ordering!

    let board_eval = board_eval.squeeze();

    let board_eval: Vec<f32> = Vec::try_from(board_eval).expect("Error");

    let board_eval = Tensor::from_slice(&vec![board_eval[0]]);

    println!("       raw policy:{}", policy);
    println!("       raw value:{}", board_eval);

    // let board_eval = Tensor::from_slice(board_eval);

    let value = Tensor::tanh(&board_eval);

    println!("    value after Tanh {}", value);

    let policy = policy.squeeze();
    println!("{}", policy);
    let policy: Vec<f32> = Vec::try_from(policy).expect("Error");
    let value = f32::try_from(value).expect("Error");
    let mut lookup: HashMap<String, f32> = HashMap::new();
    for (c, p) in contents.iter().zip(policy.iter()) {
        // full lookup, with str
        println!("{} {}", c, p);
        lookup.insert(c.to_string(), *p);
    }

    println!("{:?}", lookup);

    let mut legal_lookup: HashMap<String, f32> = HashMap::new();

    let mut legal_moves = Vec::new();
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

    for m in &fm {
        let idx_name = format!("{}", m);
        // println!("{}", idx_name);
        let m = lookup.get(&idx_name).expect("Error");
        legal_lookup.insert(idx_name, *m);
    }

    println!("{:?} ", legal_lookup);
    let mut sm: Vec<f32> = Vec::new();
    // TODO: check for performance optimisations here
    for (l, _) in &legal_lookup {
        let l = match &legal_lookup.get(l) {
            Some(&value) => value,
            None => 0.0, // default value in case of None
        };
        sm.push(l);
    }

    let sm = Tensor::from_slice(&sm);

    let sm = Tensor::softmax(&sm, 0, Kind::Float);

    let sm: Vec<f32> = Vec::try_from(sm).expect("Error");

    // println!("{:?}",sm);

    // let's try something new
    // refer back to the legal moves generator and redo the formatting, it's easier that way

    // attempt to turn Tensor back to vecs
    let mut ll: HashMap<String, f32> = HashMap::new();
    for (l, v) in legal_lookup.iter().zip(&sm) {
        let (idx_name, _) = l;
        ll.insert(idx_name.to_string(), *v);
    }

    let mut legal_lookup = ll;

    if bs.board().side_to_move() == Color::Black {
        // // println!("YOOOOOO");
        let mut n = std::collections::HashMap::new();

        for (move_key, value) in legal_lookup.clone() {
            let new_key = format!(
                "{}{}{}{}",
                &move_key.chars().nth(0).unwrap(),
                9 - move_key.chars().nth(1).unwrap().to_digit(10).unwrap(),
                &move_key.chars().nth(2).unwrap(),
                9 - move_key.chars().nth(3).unwrap().to_digit(10).unwrap()
            );
            n.insert(new_key, value);
        }

        legal_lookup = n;
    }

    let mut idx_li: Vec<usize> = Vec::new();

    for mov in contents.iter() {
        if let Some(&idx) = legal_lookup.get(*mov) {
            idx_li.push(idx as usize);
        }
    }
    // convert moves into Move object
    let mut l = HashMap::new();
    for (m, v) in legal_lookup {
        println!("{},{}", m, v);
        l.insert(m.parse().unwrap(), v);
    }
    let legal_lookup = l;
    // println!("value {}",value);
    (value, legal_lookup, idx_li)
}
