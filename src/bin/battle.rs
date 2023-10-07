use cozy_chess::{Board, GameStatus, Move};
use std::{env, io, str::FromStr};
use tz_rust::{boardmanager::BoardStack, mcts_trainer::get_move};

fn get_input(bs: &BoardStack) -> Move {
    let mut mv;
    loop {
        let mut input = String::new();
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
    while bs.status() == GameStatus::Ongoing {
        let mv = get_input(&bs);
        bs.play(mv);
        let (mv, _, _, _, _) = get_move(bs.clone());
        println!("{:#}", mv);
        bs.play(mv);
    }
}
