use tch::*;
use cozy_chess::*;
use std::collections::HashMap;
use std::vec;
use crate::mvs::get_contents;
use crate::selfplay::DataGen;

fn eval_state(board:Tensor) -> anyhow::Result<(Tensor, Tensor)> {
    let mut model = tch::CModule::load("tz.pt")?;
    model.to(Device::cuda_if_available(), Kind::Float, true);
    // reshape the model (originally from 1D)
    let b = board;
    let b = b.unsqueeze(0);
    let b = b.resize([1,21,8,8]); 
    let b = b.to(Device::cuda_if_available());
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


pub fn convert_board(board:&Board, bs:&DataGen) -> Tensor{ // ignore error for bs now
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
    let b: Board;
    let fen_str = format!("{}", board);
    let reversed_str: String;
    // println!("{}",board.side_to_move());
    if board.side_to_move() == Color::Black {
        sq1  = vec![0.0; 64];
        sq2  = vec![1.0; 64];
        let fen_str:&str = &fen_str;
        reversed_str = mirror(fen_str); // TODO: fix this
        let reversed_str:&str = &reversed_str;
        b = Board::from_fen(reversed_str, false).expect("Error");
    } else {
        // println!("fen string {}", fen_str);
        b = Board::from_fen(&fen_str, false).expect("Error");
        sq1  = vec![1.0; 64];
        sq2  = vec![0.0; 64];
    }
    let w_rights = b.castle_rights(Color::White);
    let b_rights = b.castle_rights(Color::Black);

    let wl = w_rights.long; // white left, long castling
    let wr = w_rights.short; // white right, short castling
    let sq3: Vec<f32>;
    let sq4: Vec<f32>;
    if wl != None { // still have castling
        sq3 = vec![1.0; 64];
    } else {
        sq3 = vec![0.0; 64];
    }
    if wr != None { // still have castling
        sq4 = vec![1.0; 64];
    } else {
        sq4 = vec![0.0; 64];
    }

    let bl = b_rights.long; // white left, long castling
    let br = b_rights.short; // white right, short castling

    let sq5: Vec<f32>;
    let sq6: Vec<f32>;


    if bl != None { // still have castling
        sq5 = vec![1.0; 64];
    } else {
        sq5 = vec![0.0; 64];
    }
    if br != None { // still have castling
        sq6 = vec![1.0; 64];
    } else {
        sq6 = vec![0.0; 64];
    }

    let num_reps = bs.stack_manager.get_reps() as f32; 

    let sq7 :Vec<f32>= vec![num_reps;64];
    // let sq7 :Vec<f32>= vec![0.0;64]; // hardcode 0.0 for now
    let sq8 :Vec<f32> = vec![b.halfmove_clock() as f32;64];
    // flatten to 1d
    let pieces = [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King];
    let mut pieces_sqs = Vec::new();
    for colour in Color::ALL {
        for piece in pieces {
            let mut sq: Vec<Vec<f32>> = vec![vec![0.0; 8]; 8];
            for tile in b.colored_pieces(colour, piece) {
                sq[tile.rank() as usize][tile.file() as usize] = 1.0;
            }
        pieces_sqs.extend(sq);
        }
    }
    // still have to flatten sq

    let mut sq_1d = Vec::new();

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
        sq7,
        sq8,
        pieces_sqs,
        sq21
    ];
    
    let mut sq_1d: Vec<f32> = Vec::new();
    
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

pub fn eval_board(board:&Board, bs:&DataGen) -> (f32, HashMap<cozy_chess::Move, f32>, Vec<usize>){ // ignore bigl and model for now, model is the custom net class
    let contents = get_contents();
    let b = convert_board(&board, bs);
    // convert b into [B,21,8,8] first!
    let output = eval_state(b).expect("ERROR output expect statement");
    
    // let output: Vec<Vec<f32>> = Vec::try_from(output).expect("Error");

    let (board_eval, policy) = output; // check policy, eval ordering!
    

    // let board_eval = Tensor::from_slice(board_eval);

    let value = Tensor::tanh(&board_eval);

    let mut mirrored = false;
    let b: Board;
    let fen_str = format!("{}", board);
    if board.side_to_move() == Color::Black {
        let fen_str = format!("{:?}", board);
        let fen_str:&str = &fen_str;
        let reversed_str = mirror(fen_str); // TODO: fix this
        let reversed_str:&str = &reversed_str;
        b = Board::from_fen(reversed_str, false).expect("Error");
        mirrored = true;
    } else {
        b = Board::from_fen(&fen_str, false).expect("ERROR");
    }
    let policy = policy.squeeze();
    let policy: Vec<f32> = Vec::try_from(policy).expect("Error");
    let value = f32::try_from(value).expect("Error");
    let mut lookup: HashMap<String, f32> = HashMap::new();
    for (c,p) in contents.iter().zip(policy.iter()) { // contents is from the .txt
        lookup.insert(c.to_string(),*p);

    }
    let mut legal_lookup: HashMap<String, f32> = HashMap::new();

    let mut legal_moves = Vec::new();
    b.generate_moves(|moves| {
        // Unpack dense move set into move list
        legal_moves.extend(moves);
        false
    });

    for m in &legal_moves {
        let idx_name = format!("{}",m);
        let m = match legal_lookup.get(&idx_name) {
            Some(&value) => value,
            None => 0.0, // Default value in case of None
        };
        legal_lookup.insert(idx_name, m);
    }

    let mut sm: Vec<f32> = Vec::new();
    // TODO: check for performance optimisations here
    for (l,_) in &legal_lookup {
        let l = match &legal_lookup.get(l) {
            Some(&value) => value,
            None => 0.0, // default value in case of None
        };
        sm.push(l);
    }
    
    let sm = Tensor::from_slice(&sm); 

    let sm = Tensor::softmax(&sm,0, Kind::Float);
    
    let sm:Vec<f32> = Vec::try_from(sm).expect("Error");

    // let's try something new
    // refer back to the legal moves generator and redo the formatting, it's easier that way
    
    // attempt to turn Tensor back to vecs
    for (l, v) in legal_moves.iter().zip(sm) {
        let idx_name = format!("{}",l);
        legal_lookup.insert(idx_name, v);
    }
    // let legal_lookup: HashMap<String,f32>;
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
        l.insert(m.parse().unwrap(), v);
    }
    let legal_lookup = l;
    (value, legal_lookup, idx_li)
}
