use cozy_chess::{Board, GameStatus, Move};
use std::{env, io, str::FromStr};
use tz_rust::{boardmanager::BoardStack, mcts_trainer::get_move};

fn get_input(bs: &BoardStack) -> Move {
    let mut mv;
    // get all legal moves and print them

    let mut move_list = Vec::new();

    bs.board().generate_moves(|moves| {
        // Unpack dense move set into move list
        move_list.extend(moves);
        false
    });

    let mut result = String::new();

        for (index, item) in move_list.iter().enumerate() {

            result.push_str(&format!("{}", item));

            if index < move_list.len() - 1 {
                result.push_str(", ");
            }
        }
    loop {
        let mut input = String::new();
        println!("Legal moves available: {}", result);
        println!("Enter move:");
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");
        let tb = bs.clone(); // test legality, play move
        let input = input.replace("\r\n", "");
        mv = Move::from_str(&input);
        match mv {
            Ok(valid_move) => {
                let result = tb.board().clone().try_play(valid_move); // TODO: use try_play instead
                match result {
                    Ok(_) => break,
                    Err(_) => continue,
                }
            }
            Err(_) => continue,
        }
    }

    mv.unwrap()
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let board = Board::default();
    let mut bs = BoardStack::new(board);
    let mut input = String::new();
    println!("Player (p) or bot (b) first: ");
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");
    let input = input.replace("\r\n", "");
    if input == "p".to_string() {
        while bs.status() == GameStatus::Ongoing {
            let mv = get_input(&bs);
            bs.play(mv);
            let (mv, _, _, _, _) = get_move(bs.clone());
            println!("{:#}", mv);
            bs.play(mv);
        }
    } else if input == "b".to_string() { // bot plays first
        while bs.status() == GameStatus::Ongoing {
            let (mv, _, _, _, _) = get_move(bs.clone());
            bs.play(mv);
            println!("{:#}", mv);
            let mv = get_input(&bs);
            bs.play(mv);
        }
    }
}
