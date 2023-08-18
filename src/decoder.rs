use tch::*;
use cozy_chess::*;
use std::collections::HashMap;
use crate::boardmanager::BoardStack;
use crate::mvs::get_contents;

fn eval_state(board:Tensor) -> anyhow::Result<Vec<Tensor>> {
    let model = tch::CModule::load("chess_16x128_gen3634.pt")?;
    let b = board.to(Device::cuda_if_available());
    board.unsqueeze(0);
    let board = IValue::TensorList(vec![board]);
    let board = [board];
    let output = model.forward_is(&board)?;
    let output_tensor_list = match output {
    IValue::TensorList(tensor_list) => tensor_list,
    _ => panic!("the output is not a TensorList"),
};
    Ok(output_tensor_list)
}


pub fn convert_board(board:Board, bs:BoardStack) -> Tensor{ // not include bigl for now
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

    if board.side_to_move() == Color::Black {
        let sq1 = vec![0.0; 64];
        let sq2 = vec![1.0; 64];
        let fen_str = format!("{:?}", board);
        let fen_str:&str = &fen_str;
        let reversed_str = mirror(fen_str); // TODO: fix this
        let reversed_str:&str = &reversed_str;
        let board = Board::from_fen(reversed_str, false).unwrap();
    } else {
        let sq1 = vec![1.0; 64];
        let sq2 = vec![0.0; 64];
        let fen_str = format!("{:?}", board);
        let fen_str:&str = &fen_str;
        let reversed_str = mirror(fen_str); // TODO: fix this
        let reversed_str:&str = &reversed_str;
        let board = Board::from_fen(reversed_str, false).unwrap();
    }
    let us = board.side_to_move();

    let w_rights = board.castle_rights(Color::White);
    let b_rights = board.castle_rights(Color::Black);

    let wl = w_rights.long; // white left, long castling
    let wr = w_rights.short; // white right, short castling
    let sq3: Vec<f32>;
    let sq4: Vec<f32>;
    if wl != None { // still have castling
        let sq3 = vec![1.0; 64];
    } else {
        let sq3 = vec![0.0; 64];
    }
    if wr != None { // still have castling
        let sq4 = vec![1.0; 64];
    } else {
        let sq4  = vec![0.0; 64];
    }

    let bl = b_rights.long; // white left, long castling
    let br = b_rights.short; // white right, short castling

    let sq5: Vec<f32>;
    let sq6: Vec<f32>;


    if bl != None { // still have castling
        let sq5 = vec![1.0; 64];
    } else {
        let sq5 = vec![0.0; 64];
    }
    if br != None { // still have castling
        let sq6 = vec![1.0; 64];
    } else {
        let sq6 = vec![0.0; 64];
    }

    // skip sq7 and 8 for reps
    // create b or whatevah
    let num_reps = bs.get_reps() as f32; 

    let sq7 = vec![num_reps];

    let sq8 = vec![board.halfmove_clock() as f32;64];

    let pieces = [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King];
    let mut pieces_sqs = Vec::new();
    for colour in [Color::White, Color::Black] {
        for piece in pieces {
            let sq = vec![vec![0.0; 8]; 8];
            // let sq = Tensor::empty([8,8], (Kind::Float, Device::Cpu));
            for tile in board.colored_pieces(colour, piece) {
                let tile = format!("{:?}",tile);
                let tile = tile.parse::<i32>().unwrap();
                let (quotient, remainder) = (tile / 8 as i32, tile % 8);
                let quotient = quotient as usize;
                let remainder = remainder as usize;
                sq[quotient][remainder] = 1.0;
            pieces_sqs.extend(sq);
            }
        }
    }
    // still have to flatten sq

    let sq_1d = Vec::new();

    for row in pieces_sqs {
        for element in row {
            sq_1d.push(element);
        }
    }

    let pieces_sqs = sq_1d;

    let sq21 = vec![0.0; 64];
    
    let all_data = [
        sq1,
        sq2, 
        sq3,
        sq4,
        sq5, 
        sq6, 
        // skip 7 for now
        sq8,
        pieces_sqs,
        sq21
    ];
    
    let sq_1d: Vec<f32> = Vec::new();
    
    for row in all_data {
        for element in row {
            sq_1d.push(element);
        }
    }
    let all_data = sq_1d;
    let all_data = all_data.to_vec();
    let all_data = Tensor::from_slice(&all_data);
    all_data // all_data is 1d
}


// func to mirror board

fn mirror(fen: &str) -> String {
    let parts: Vec<&str> = fen.split_whitespace().collect();
    let board = parts[0];
    let turn = parts[1];
    let castling = parts[2];
    let en_passant = parts[3];
    let halfmove_clock = parts[4];
    let fullmove_number = parts[5];

    // Flip the board, reverse ranks <-->
    let flipped_board: String = board
        .split('/')
        .map(|rank| rank.chars().rev().collect::<String>())
        .rev()
        .collect::<Vec<String>>()
        .join("/");

    // turn
    let flipped_turn = if turn == "b" { "w" } else { "b" };

    // castling
    let flipped_castling: String = castling
        .chars()
        .map(|c| match c {
            'K' => 'k',
            'Q' => 'q',
            'k' => 'K',
            'q' => 'Q',
            _ => c,
        })
        .collect();

    // en passant
    let flipped_en_passant = if en_passant != "-" {
        en_passant.chars().rev().collect::<String>()
    } else {
        "-".to_string()
    };

    let flipped_fen = format!(
        "{} {} {} {} {} {}",
        flipped_board, flipped_turn, flipped_castling, flipped_en_passant, halfmove_clock, fullmove_number
    );

    flipped_fen
}

pub fn eval_board(board:Board, bs:BoardStack) { // ignore bigl and model for now, model is the custom net class
    let contents = get_contents();
    let b = convert_board(board, bs);
    // convert b into [B,21,8,8] first!
    match eval_state(b) {
        Ok(output) => {
            // reshape and view
            
            let (board_eval, policy) = (output[0], output[1]);
            let value = Tensor::tanh(&board_eval);
            // ignore getting .item()
            let mirrored = false;
            if board.side_to_move() == Color::Black {
                let fen_str = format!("{:?}", board);
                let fen_str:&str = &fen_str;
                let reversed_str = mirror(fen_str); // TODO: fix this
                let reversed_str:&str = &reversed_str;
                let board = Board::from_fen(reversed_str, false).unwrap();
                mirrored = true;
            }
            
            let mut lookup = HashMap::new();
            for (c,p) in contents.iter().zip(policy.iter()) { // contents is from the .txt
                lookup.insert(c, p);

            }
            let mut legal_lookup = HashMap::new();

            let mut legal_moves = Vec::new();
            board.generate_moves(|moves| {
                // Unpack dense move set into move list
                legal_moves.extend(moves);
                false
            });

            for m in legal_moves {
                let idx_name = format!("{}",m).as_str();
                legal_lookup.insert(idx_name, lookup.get(&idx_name));
            }

            let sm: Vec<f32> = Vec::new();

            for (l,_) in legal_lookup {
                sm.push(legal_lookup.get(l));
            }
            
            let sm = Tensor::from_slice(&sm); 

            let sm = Tensor::softmax(&sm,0, Kind::Float);
            

            // let's try something new
            // refer back to the legal moves generator and redo the formatting, it's easier that way
            
            // attempt to turn Tensor back to vecs
            for (l, v) in legal_moves.iter().zip(sm) {
                let idx_name = format!("{}",l).as_str();
                legal_lookup.insert(legal_lookup.get(&idx_name), v);
            }

            if mirrored {
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

                let legal_lookup = n;

            }
            let mut idx_li: Vec<usize> = Vec::new();
    
            for mov in contents.iter() {
                if let Some(&idx) = legal_lookup.get(mov) {
                    idx_li.push(idx);
                }
            }
        
        (value, legal_lookup, idx_li)
        }
        Err(err_msg) => {
            println!("Error: {}", err_msg);
        }
    }

}
