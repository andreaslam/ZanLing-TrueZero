use anyhow::Result;
use cozy_chess::{Board, Color, Move};
use tch::{Kind, Tensor};

use tzrust::{
    boardmanager::BoardStack,
    decoder::{convert_board, eval_state},
    mcts_trainer::Net,
    mvs::get_contents,
};

fn main() -> Result<()> {
    // configuration

    let network_path =
        "/Users/andreas/Desktop/Code/RemoteFolder/TrueZero/hidden/chess_16x128_gen3634.pt";

    // construct the starting position

    // let board = Board::default();
    let board = Board::from_fen(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        false,
    )
    .unwrap();

    println!("=== BOARD ===");
    println!("FEN: {}", board);
    println!("Side to move: {:?}", board.side_to_move());

    let bs = BoardStack::new(board.clone());

    // encode exactly one position

    let input = convert_board(&bs);

    println!();
    println!("=== INPUT ===");
    println!("shape before reshape: {:?}", input.size());
    println!("num elements: {}", input.numel());

    // inspect the scalar channels

    let input_vec: Vec<f32> = Vec::try_from(input.shallow_clone())?;

    println!("scalar channels:");

    for i in 0..8 {
        println!("  channel {:2}: {}", i + 1, input_vec[i * 64]);
    }

    // inspect piece plane occupancy

    println!("piece planes:");

    for channel in 8..20 {
        let offset = channel * 64;

        let squares: Vec<usize> = (0..64).filter(|&i| input_vec[offset + i] != 0.0).collect();

        println!("  channel {:2}: {:?}", channel + 1, squares);
    }

    // run a forward pass without mcts

    let net = Net::new(network_path);

    let (board_eval, policy) = eval_state(input, &net)?;

    // decode the board head

    let board_eval = board_eval.squeeze();

    let board_evals: Vec<f32> = Vec::try_from(board_eval)?;

    println!();
    println!("=== BOARD OUTPUT ===");

    println!("raw board head:");

    for (i, value) in board_evals.iter().enumerate() {
        println!("  [{}] {:.10}", i, value);
    }

    let value = board_evals[0].tanh();

    let wdl_logits = Tensor::from_slice(&board_evals[1..4]);

    let wdl = Tensor::softmax(&wdl_logits, 0, Kind::Float);

    let wdl: Vec<f32> = Vec::try_from(wdl)?;

    let moves_left = board_evals[4].max(0.0);

    println!();
    println!("decoded:");
    println!("  value      = {:.10}", value);
    println!("  W          = {:.10}", wdl[0]);
    println!("  D          = {:.10}", wdl[1]);
    println!("  L          = {:.10}", wdl[2]);
    println!("  moves_left = {:.10}", moves_left);

    // dump all policy outputs

    let policy = policy.squeeze();

    let policy: Vec<f32> = Vec::try_from(policy)?;

    let contents = get_contents();

    assert_eq!(
        policy.len(),
        contents.len(),
        "policy output has {} entries but move list has {} entries",
        policy.len(),
        contents.len()
    );

    println!();
    println!("=== POLICY ===");
    println!("policy entries: {}", policy.len());

    // print the raw policy logits

    println!();
    println!("raw policy logits:");

    for (i, (&logit, mv)) in policy.iter().zip(contents.iter()).enumerate() {
        println!("{:4} {} {:.10}", i, mv, logit);
    }

    // decode legal moves without modifying the raw network output

    let mut legal_moves = Vec::<Move>::new();

    board.generate_moves(|moves| {
        legal_moves.extend(moves);
        false
    });

    println!();
    println!("=== LEGAL MOVES ===");
    println!("legal moves: {}", legal_moves.len());

    println!();
    println!("legal move policy:");

    let side = board.side_to_move();

    let mut legal_logits = Vec::new();

    for mv in legal_moves {
        let canonical = if side == Color::Black {
            Move {
                from: mv.from.flip_rank().flip_file(),
                to: mv.to.flip_rank().flip_file(),
                promotion: mv.promotion,
            }
        } else {
            mv
        };

        let idx = contents
            .iter()
            .position(|x| *x == canonical)
            .expect("legal move missing from 1880 move list");

        let logit = policy[idx];

        legal_logits.push((mv, canonical, idx, logit));
    }

    for (mv, canonical, idx, logit) in &legal_logits {
        println!(
            "{:6} canonical={:6} idx={:4} logit={:.10}",
            mv, canonical, idx, logit
        );
    }

    // apply softmax only over legal moves

    let logits = Tensor::from_slice(&legal_logits.iter().map(|x| x.3).collect::<Vec<f32>>());

    let probs = Tensor::softmax(&logits, 0, Kind::Float);

    let probs: Vec<f32> = Vec::try_from(probs)?;

    println!();
    println!("legal-move probabilities:");

    let mut ranked = legal_logits
        .iter()
        .zip(probs.iter())
        .map(|((mv, canonical, idx, logit), prob)| (*mv, *canonical, *idx, *logit, *prob))
        .collect::<Vec<_>>();

    ranked.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));

    for (rank, (mv, canonical, idx, logit, prob)) in ranked.iter().enumerate() {
        println!(
            "{:2}. {:6} canonical={:6} idx={:4} logit={:12.8} p={:.10}",
            rank + 1,
            mv,
            canonical,
            idx,
            logit,
            prob
        );
    }

    Ok(())
}
