use tch::*;
use cozy_chess::*;
use std::collections::HashMap;

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


pub fn convert_board(board:Board) -> Tensor{ // not include bigl for now
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

    let sq1: Tensor;
    let sq2: Tensor;

    if board.side_to_move() == Color::Black {
        sq1 = Tensor::zeros([8,8], (Kind::Float, Device::Cpu));
        sq2 = Tensor::ones([8,8], (Kind::Float, Device::Cpu));
        let fen_str = format!("{:?}", board);
        let fen_str:&str = &fen_str;
        let reversed_str = mirror(fen_str); // TODO: fix this
        let reversed_str:&str = &reversed_str;
        let board = Board::from_fen(reversed_str, false).unwrap();
    } else {
        sq1 = Tensor::ones([8,8], (Kind::Float, Device::Cpu));
        sq2 = Tensor::zeros([8,8], (Kind::Float, Device::Cpu));
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
    let sq3: Tensor;
    let sq4: Tensor;
    if wl != None { // still have castling
        sq3  = Tensor::full([8,8], 1,(Kind::Float, Device::Cpu));   
    } else {
        sq3  = Tensor::full([8,8], 0,(Kind::Float, Device::Cpu));   
    }
    if wr != None { // still have castling
        sq4  = Tensor::full([8,8], 1,(Kind::Float, Device::Cpu));   
    } else {
        sq4  = Tensor::full([8,8], 0,(Kind::Float, Device::Cpu));   
    }

    let bl = b_rights.long; // white left, long castling
    let br = b_rights.short; // white right, short castling

    let sq5: Tensor;
    let sq6: Tensor;


    if bl != None { // still have castling
        sq5  = Tensor::full([8,8], 1,(Kind::Float, Device::Cpu));  
    } else {
        sq5  = Tensor::full([8,8], 0,(Kind::Float, Device::Cpu));   
    }
    if br != None { // still have castling
        sq6  = Tensor::full([8,8], 1,(Kind::Float, Device::Cpu));   
    } else {
        sq6  = Tensor::full([8,8], 0,(Kind::Float, Device::Cpu));   
    }

    // skip sq7 and 8 for reps

    let pieces = [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King];
    let mut pieces_sqs = Vec::new();
    for colour in [Color::White, Color::Black] {
        for piece in pieces {
            const ROWS: usize = 8;
            const COLS: usize = 8;
            // let mut sq: Vec<Vec<i32>> = vec![vec![0; COLS]; ROWS];
            let mut sq = Tensor::empty([8,8], (Kind::Float, Device::Cpu));
            for tile in board.colored_pieces(colour, piece) {
                let tile = format!("{:?}",tile);
                let tile = tile.parse::<i32>().unwrap();
                let (quotient, remainder) = (tile / 8 as i32, tile % 8);
                let quotient = quotient as usize;
                let remainder = remainder as usize;
                sq[quotient][remainder] = 1;
            pieces_sqs.extend(sq);
            }
        }
    }

    let mut result = Vec::new();

    for inner_vec in pieces_sqs {
        let inner_tensor = Tensor::from_slice(&inner_vec);
        result.push(inner_tensor);
    }

    let (turn,opp) = (result[0], result[1]);

    

    let sq21 = Tensor::zeros([8,8],(Kind::Float, Device::Cpu)); // 0 for now
    let all_data = [
        sq1,
        sq2, 
        sq3,
        sq4,
        sq5, 
        sq6, 
        // skip 7 and 8
        *pieces_sqs,
        sq21
    ];
    all_data = Tensor::stack(&all_data, 0);
    all_data 
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

pub fn eval_board(board:Board) { // ignore bigl and model for now, model is the custom net class
    let contents = get_contents();
    let b = convert_board(board);
    match eval_state(b) {
        Ok(output) => {
            let (board_eval, policy) = (output[0], output[1]);
            // hmm trying to get them to TensorLists 
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
                legal_lookup.insert(m, lookup.get(&idx_name));
            }

            let sm = Vec::new();

            for (l,_) in legal_lookup {
                sm.extend(legal_lookup.get(&l));
            }
            

            sm = Tensor::softmax(sm,0, Kind::Float);
            // skip tolist
            let sm = sm.to_vec();
            for (l, v) in legal_lookup.iter().zip(sm.iter()) {
                let idx_name = format!("{}",l).as_str();
                legal_lookup.insert(legal_lookup.get(&idx_name), v);
            }

            if mirrored {
                let n = {};
                let s = 0;
                for (m, key) in legal_lookup.iter().zip() { // find .items()
                    // oh no type conversion shit
                    s += key[-1]; // ????
                legal_lookup = n;
            }

            }
            let idx_li  = Vec::new();

            for m in legal_lookup {
                idx_li.extend(contents.index(m)); // contents is opening .txt
                
            }
        (value, legal_lookup, idx_li)
        }
        Err(err_msg) => {
            println!("Error: {}", err_msg);
        }
    }

}
