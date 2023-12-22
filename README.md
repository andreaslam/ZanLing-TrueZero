# ZanLing-TrueZero
A Python and Rust chess engine that starts from Zero. This project is still very much work-in-progress.


## About the Engine 
The name of the Engine is 真零 (TrueZero), which is Chinese for "True Zero" and romanised using [Jyutping](https://en.wikipedia.org/wiki/Jyutping) for Cantonese.

Instead of being hard-coded by humans, this AI learns how to play through playing games from itself and learning from randomly generated games. 


The chess Engine will then play games against itself using the evaluation to evaluate chess positions, done using [Monte Carlo Tree Search (MCTS)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search).

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

Alternately, you can use Docker to set up this project for training. This assumes that you are using Linux Ubuntu to train since the base Docker Image uses Ubuntu. 

firstly, run

```
https://hub.docker.com/repository/docker/andreaslam/tz/general
```

Next, set the following environment variables.
```

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1 
PYTORCH_PATH=$(python3 -c "import torch; print(torch.__path__[0])")
export LD_LIBRARY_PATH="$PYTORCH_PATH/lib:$LD_LIBRARY_PATH"
```

## Features in progress
Rewrite in progress! After careful consideration, TrueZero will be written in Rust. More details can be found on the [roadmap here](https://github.com/andreaslam/ZanLing-TrueZero/issues/1)

## What each file does

### Internal testing (non-UCI compliant)
- `getdecode.rs` - used for obtaining the encoded NN inputs.
- `getmove.rs` - used for obtaining a single tree search.
- `getgame.rs` - used for obtaining a game.
- `getinferencetime.rs` - used for benchmarking inference times and batching effectiveness through calculating the nodes/s.


### Source code for AI
- `decoder.rs` - used to decode and encode inputs for AI. Also handles the creation of child nodes. This is where NN inference happens.
- `mcts_trainer.rs` - used for MCTS tree search. Initialises the NN and manages the entire tree search. Adds Dirichlet noise to search results.
- `boardmanager.rs` - a wrapper for the cozy-chess library. Manages and handles draw conditions, such as fifty-move repetition, threefold repetition and must-draw scenarios.
- `dirichlet.rs` - Dirichlet noise generator.
- `mvs.rs` - a large array that contains all possible moves in chess. Used for indexing and storing (legal) move order. Statically loads and stored during programme execution.
- `selfplay.rs` - facilitates selfplay. This is where search is initialised. Contains temperature management.
- `fileformat.rs` - contains the code for binary encoding.
- `dataformat.rs` - contains necessary abstractions for `fileformat.rs`.
- `message_types.rs` - contains the message protocols for processes (such as the Generator and the training loop) to communicate with the server and vice vera.

### Data Generation and Training components

#### Rust binaries
- `main.rs` - runs multi-threaded data generation code, where each thread runs an independent game. It needs to be connected to `server.rs` via TCP in order to get the latest Neural Net. It also sends key statistics for live telemetry.
- `server.rs` - a TCP server that co-ordinates Rust data generation and Python training. It sends each connected instance a unique identifier, broadcasts key information to different processes, which include statistics, Neural Network information and training settings.

#### Python code
- `client.py` - runs training and manages neural network training. Connects to `server.rs` via TCP to receive file paths for neural network training.

## Libraries/technologies used 

### Python 

This Python and Rust Engine uses the following:
- **Pytorch** - used for creating and training the Neural Network 
- **Numpy** - used for processing data (chess board representation after one-hot encoding, handling final outcome and final game result


### Rust

- **cozy-chess** - chess move generation library. There is a [simple wrapper](https://github.com/andreaslam/ZanLing-TrueZero/blob/main/src/boardmanager.rs) of this library that TrueZero uses that covers draws, repetitions and serves as an interface between cozy-chess and the rest of the code.
- **flume** - multi-sender, multi-producer channels used to send data between channels for data generation.
- **tch-rs** - Rust wrapper of libtorch. Used for Neural Network inference.
- **crossbeam** - enables multithreading data generation.
- **serde** - serialises messages to send across TCP server


## Credits and Acknowledgements

I would like to extend my heartfelt thanks to **[Karel Peeters](https://github.com/KarelPeeters)** for his persistent help and guidance. Without him this project would not been possible. 

Portions/entire files of code from [KZero](https://github.com/KarelPeeters/kZero) are being used in this Repository with express permission from Karel, which include:

- `src/dirichlet.rs`
- `src/fileformat.rs`
- `src/dataformat.rs`
- the `lib` folder used for reading KZero's custom data format and training



