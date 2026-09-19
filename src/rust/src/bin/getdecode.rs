use cozy_chess::Board;
use std::{env, time::Instant};
use tzrust::{boardmanager::BoardStack, decoder::convert_board};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let board = Board::from_fen(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        false,
    )
    .unwrap();

    let bs = BoardStack::new(board);

    let sw = Instant::now();
    let converted_tensor = convert_board(&bs);

    println!(
        "# Rust conversion: {:.3} ms",
        sw.elapsed().as_nanos() as f64 / 1e6
    );

    let flat = converted_tensor.reshape([-1]);

    let values: Vec<f64> = Vec::<f64>::try_from(&flat).unwrap();

    println!("bigl = torch.tensor(");
    println!("    [");

    for chunk in values.chunks(16) {
        print!("        ");
        for (i, value) in chunk.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }

            // Integers are emitted as 0/1 rather than 0.0/1.0,
            // making the resulting Python tensor visually cleaner.
            if value.fract() == 0.0 {
                print!("{}", *value as i64);
            } else {
                print!("{:.10}", value);
            }
        }
        println!(",");
    }

    println!("    ]");
    println!(",dtype=torch.float64,");
    println!(")");
}
