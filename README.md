
<div align="center">

<img src="https://github.com/andreaslam/ZanLing-TrueZero/blob/main/TrueZero.png" alt="TrueZero logo">

### A Python and Rust chess engine that starts from Zero.
<img src="https://img.shields.io/badge/Powered%20by-Rust-b7410e" alt="Powered by Rust">
<img src="https://img.shields.io/badge/Powered%20by-Python-306998" alt="Powered by Python">
<img src="https://badgen.net/github/commits/andreaslam/ZanLing-TrueZero/main" alt="Total commits">
</div>

## About the Engine 
The name of the Engine is 真零 (TrueZero), which is Chinese for "True Zero" and romanised using [Jyutping](https://en.wikipedia.org/wiki/Jyutping) for Cantonese (Zan1 Ling4).

Instead of using hand-crafted evaluations (HCE), this AI learns how to play through playing against itself, starting with zero prior knowledge except for the rules of chess.

The chess engine will then play games against itself using the evaluation to evaluate chess positions, done using [Monte Carlo Tree Search (MCTS)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search).

This project is still very much work-in-progress.

## Engine setup

### Using git

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

### Using Docker

Alternately, you can use Docker to set up this project for training. This assumes that you are using Linux Ubuntu to train since the base Docker Image uses Ubuntu. 

Firstly, run

```
docker pull andreaslam/tz
```
You may need to change your directory and locate this project.
```
cd ..
cd app
```

## Setting environment variables


Set the following environment variables to configure `tch-rs` before running anything. This assumes you use PyTorch version 2.1 to use and set up. Use a [virtual environment if neeeded](https://docs.python.org/3/library/venv.html). 

```
export LIBTORCH_USE_PYTORCH=1
PYTORCH_PATH=$(python3 -c "import torch; print(torch.__path__[0])")
export LD_LIBRARY_PATH="$PYTORCH_PATH/lib:$LD_LIBRARY_PATH"
```

On Windows, it's
```
$env:LIBTORCH_USE_PYTORCH=1
$PYTORCH_PATH = python -c "import torch; print(torch.__path__[0])"
$env:PATH = "$PYTORCH_PATH\lib;$env:PATH"
$env:LIBTORCH_BYPASS_VERSION_CHECK=1
```

## Running Data Generation and Training

Before data generation, ensure that the loaded copy of the Repository is up-to-date.

```
git pull
```

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

## Running the Engine in UCI

To use TrueZero through UCI, simply run the following:

```
cargo run --bin ucimain
```

## What each file does

### Internal Engine testing (non-UCI compliant)
- `getdecode.rs` - used for obtaining the encoded NN inputs.
- `getmove.rs` - used for obtaining a single tree search.
- `getgame.rs` - used for obtaining a game.
- `getinferencetime.rs` - used for benchmarking inference times and batching effectiveness through calculating the nodes/s.

### External Engine testing (UCI compliant)
- `uci.rs` - contains code for UCI implementation.
- `ucimain.rs` - used for running games using UCI.

### Source code for the Engine
- `decoder.rs` - used to decode and encode inputs for the Engine. Also handles the creation of child nodes. This is where NN inference happens.
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

### Utility and visualisation code

- `visualiser.py` - visualises training data and monitoring key performance indicators logged in `client.py`, where code from `lib/plotter.py`. For more details on the `lib` folder see [here](https://github.com/andreaslam/ZanLing-TrueZero?tab=readme-ov-file#credits-and-acknowledgements).
- `visualisenet.py` - code that allows visualisation of neural network activations. Generates a `.gif` file for an animation of a policy-only game.
- `scheduler.py` - code for TrueScheduler, an experiment scheduler for scheduling experiments and monitoring server controls. This GUI also supports remote SSH logins to schedule experiments on external devices. 
- `gui.py` - GUI code for `scheduler.py`. Code is currently work in progress. Running this file directly launches a demo version without server backend.
- `experiment.py` - code that can convert data generated from `client.py` into TensorBoard-readable format. This is an alternative experiment visualiser. Additionally, the code supports remote visualisation if SSH port forwarding is enabled on your device.
- `exec_plotter.py` - debugging code that shows the thread schedule. Useful for debugging async tasks (such as data generation code for the Engine)
- `onnx_exporter.py` - contains code to convert `.pt` model weights to `.onnx`

## Libraries/technologies used 
This Python and Rust Engine uses the following:
### Python 

- **PyTorch** - used for creating and training the Neural Network. Also used for visualising experiments through its TensorBoard API.
- **Matplotlib** - used for plotting and creating animations and visualisations in `visualisenet.py`. 
- **NumPy** - used for processing data (chess board representation after one-hot encoding, handling final outcome and final game result

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


