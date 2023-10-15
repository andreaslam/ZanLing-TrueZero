# ZanLing-TrueZero [V2 IN PROGRESS]
A Python and Rust chess engine that starts from Zero. This project is still very much work-in-progress.


## About the Engine 
The name of the Engine is 真零 (TrueZero), which is Chinese for "True Zero" and romanised using [Jyutping](https://en.wikipedia.org/wiki/Jyutping) for Cantonese.

Instead of being hard-coded by humans, this AI learns how to play through playing games from itself and learning from randomly generated games. 

The evaluation learns from randomly generated games with outcomes and move turns and based on move turn and final outcome, assign a score between 1/0/-1 for games that were won/lost/drawn respectively, accounting for move turn.

The chess Engine will then play games against itself using the evaluation to evaluate chess positions, done using [Monte Carlo Tree Search (MCTS)]([https://en.wikipedia.org/wiki/Negamax#Negamax_with_alpha_beta_pruning](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)).

## How all this works in more detail 


## Engine setup

## Features in progress
Rewrite in progress! After careful consideration, TrueZero will be written in Rust. 

## Long term goals and vision
- Implement reinforcement learning (specifically genetic algorithms) for the AI in order to allow for quicker realisation of chess concepts through gameplay.
- V2 will feature chess games that are randomly generated instead of taking from games that are played by humans before, to fully achieve TrueZero.
- Create a [Data Engine](https://www.youtube.com/watch?v=zPH5O8hRfMA) where games are taken automatically and put to training DB and newly trained AIs can play against each other 24/7

## What each file does
### Evaluation Engines (NN files)

### Internal testing (non-UCI compliant)
- `getdecode.rs` - used for internal testing. Used for obtaining the encoded NN inputs.
- `getmove.rs` - used for internal testing. Used for obtaining a single tree search.
- `getgame.rs` - used for internal testing. Used for obtaining a game.
### Source code for AI
- `decoder.rs` - used to decode and encode inputs for AI. Also handles the creation of child nodes. This is where NN inference happens.
- `mcts_trainer.rs` - used for MCTS tree search. Initialises the NN and manages the entire tree search. Adds Dirichlet noise .
- `boardmanager.rs` - a wrapper for the cozy-chess library. Manages and handles draw conditions, such as fifty-move repetition, threefold repetition and must-draw scenarios.
- `dirichlet.rs` - Dirichlet noise generator.
- `mvs.rs` - a large array that contains all possible moves in chess. Used for indexing and storing (legal) move order. Statically loads and stored during programme execution.
- `selfplay.rs` - facilitates selfplay. This is where search is initialised. Contains temperature management.
- `fileformat.rs` - contains the code for binary encoding.
- `dataformat.rs` - contains necessary abstractions for `fileformat.rs`.

## Libraries/technologies used 
This Python and Rust Engine uses the following:
- **Pytorch** - used for creating NN
- **Numpy** - used for processing data (chess board representation after one-hot encoding, handling final outcome and final game  result
- **Scikitlearn** - used minimally for splitting data into train/validation sets (will be replaced with Pytorch DataLoader in the future)
- **Python Chess** - used for handling board-related code
- **Cython** - used for running files at faster speeds instead of running on Vanilla Python 
- **Setuptools** - used in tandem with Cython to Cythonise the Python code
- **SQLite3** - used for writing/accessing data to the SQL database
- **tqdm** - used as progress bar 
- **multiprocessing** - used for parallelisation of code

*database not uploaded to this GitHub repository


