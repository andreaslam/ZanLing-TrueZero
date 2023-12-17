# ZanLing-TrueZero
A Python and Rust chess engine that starts from Zero. This project is still very much work-in-progress.


## About the Engine 
The name of the Engine is 真零 (TrueZero), which is Chinese for "True Zero" and romanised using [Jyutping](https://en.wikipedia.org/wiki/Jyutping) for Cantonese.

Instead of being hard-coded by humans, this AI learns how to play through playing games from itself and learning from randomly generated games. 


The chess Engine will then play games against itself using the evaluation to evaluate chess positions, done using [Monte Carlo Tree Search (MCTS)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)).

## Engine setup

Firstly, download this repository onto your computer. 

```
git clone https://github.com/andreaslam/ZanLing-TrueZero
```

Make sure you have Rust installed. If not, follow the instructions [here](https://doc.rust-lang.org/book/ch01-01-installation.html). 
Make sure you have Python installed. If not, download the latest version [here]([https://doc.rust-lang.org/book/ch01-01-installation.html](https://www.python.org/downloads/)). 
Configure `tch-rs` from the instructions [here](https://github.com/LaurentMazare/tch-rs/blob/main/README.md). For now, the neural net for this project is not provided but the NN architecture is available [here](https://github.com/andreaslam/ZanLing-TrueZero/blob/main/network.py) for reference.

Navigate to `ZanLing-TrueZero`:

```
cd ZanLing-TrueZero
```

Then, build using `cargo`:

```
cargo build
```

Then choose a binary to run!

## Running Data Generation

To run data generation, simply run the python training client `client.py`, `main` binary and the `server` binary as follows. Open a new terminal window for each.

```
python client.py
```

```
cargo run --bin main 
```

```
cargo run --bin server
```

## Features in progress
Rewrite in progress! After careful consideration, TrueZero will be written in Rust. More details can be found on the [roadmap here](https://github.com/andreaslam/ZanLing-TrueZero/issues/1)

## What each file does

### Internal testing (non-UCI compliant)
- `getdecode.rs` - Used for obtaining the encoded NN inputs.
- `getmove.rs` - Used for obtaining a single tree search.
- `getgame.rs` - Used for obtaining a game.
- `getinferencetime.rs` - Used for benchmarking inference times and batching effectiveness through calculating the nodes/s. 
### Source code for AI
- `decoder.rs` - used to decode and encode inputs for AI. Also handles the creation of child nodes. This is where NN inference happens.
- `mcts_trainer.rs` - used for MCTS tree search. Initialises the NN and manages the entire tree search. Adds Dirichlet noise .
- `boardmanager.rs` - a wrapper for the cozy-chess library. Manages and handles draw conditions, such as fifty-move repetition, threefold repetition and must-draw scenarios.
- `dirichlet.rs` - Dirichlet noise generator.
- `mvs.rs` - a large array that contains all possible moves in chess. Used for indexing and storing (legal) move order. Statically loads and stored during programme execution.
- `selfplay.rs` - facilitates selfplay. This is where search is initialised. Contains temperature management.
- `fileformat.rs` - contains the code for binary encoding.
- `dataformat.rs` - contains necessary abstractions for `fileformat.rs`.
- `message_types.rs` - contains the message protocols for processes (such as the Generator and the training loop) to communicate with the server and vice vera.

## Libraries/technologies used 

### Python 

This Python and Rust Engine uses the following:
- **Pytorch** - used for creating NN
- **Numpy** - used for processing data (chess board representation after one-hot encoding, handling final outcome and final game  result

### Rust

- **cozy-chess** - chess move generation library. There is a [simple wrapper](https://github.com/andreaslam/ZanLing-TrueZero/blob/main/src/boardmanager.rs) of this library that TrueZero uses that covers draws, repetitions and serves as an interface between cozy-chess and the rest of the code.
- **flume** - multi-sender, multi-producer channels used to send data between channels for data generation.
- **tch-rs** - Rust wrapper of libtorch.
- **crossbeam** - multithreading


## Credits and Acknowledgements

I would like to extend my heartfelt thanks to **[Karel Peeters](https://github.com/KarelPeeters)** for his persistent help and guidance. Without him this project would not been possible. 

Portions/entire files of code from [KZero](https://github.com/KarelPeeters/kZero) are being used in this Repository with express permission from Karel, which include:

- `src/dirichlet.rs`
- `src/fileformat.rs`
- `src/dataformat.rs`
- the `lib` folder used for reading KZero's custom data format and training



