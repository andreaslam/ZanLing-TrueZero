# ZanLing-TrueZero
A Python 🐍 chess ♟️ engine that starts 🆕 from Zero 0️⃣.  This project is still very much work-in-progress.

This repository contains latest and greatest code for the chess AI but previous-generation code can be accessed through the `archives` folder.

## About the Engine 
The name of the Engine is Zan1Ling4, which is taken from 真零, which is Chinese for  "True Zero" and romanised using [Jyutping](https://en.wikipedia.org/wiki/Jyutping) for Cantonese.

Instead of being hard-coded by humans, this AI learns 🎓 how to play ⏯️ through playing games 🎮 from itself and learning 📖 from randomly generated games. 

The evaluation learns 📕 from randomly generated games with outcomes and move turn and based on move turn and final outcome, assign a score 💯 between 1/0/-1 for games that were won/lost/drew respectively, accounting 🧾 for move turn.

The chess ♟️ Engine will then play ▶️ games against itself using the evaluation to evalutate chess ♟️ positions, done using  [Negamax with Alpha Beta pruning](https://en.wikipedia.org/wiki/Negamax#Negamax_with_alpha_beta_pruning).

## How all this works in more detail 🔎
The Evaluation AI is trained on a SQL database* containing games 🎮 that were previously played 👾 on [Lichess](lichess.com) (taken [here](https://database.lichess.org/)). The AI is then trained on 20 randomly selected board states of each game, given nothing (no prerequisite knowledge) but the board  state (after one-hot  encoding) and turn to move (and nothing else) to predict final outcome, with 1,0 and -1 denoting a winning, 🎖️ an equal outcome (draw) and a losing 🏳️ game for relative to side to move. This regression model is then fed to the Engine, where it looks 👀 for optimal moves using Negamax with Alpha Beta pruning, giving an evaluation for each position.

## Engine setup
Run ```pip install -r requirements.txt``` in Terminal. This `requirements.txt` works with most versions of the AI as the packages used are mostly the same.

## Features in progress
- Implement finding best hyperparameters using genetic 🧬 algorithm 🔍 and evolutionary algorithms
- Implement massive speed boosts (Cython/Numba) and improving memory efficiency (`multiprocessing`, `subprocesses`) 
- More UCI features and more complete support (showing best moves in each depth) for UCI

## Current todo and goals 
- Train V1 Evaluation Engine (with at least 3+ million games 🎮 analysed) (ETA - End 🔚 of May)
- Have fully functional UCI compliance ✅ (ETA - End 🔚 of June/start of July)
- Implement Move Ordering (ETA - mid-July)
- Implement Transposition tables, killer🔪 moves (ETA - August)
- TrueZero V1 (ETA - End 🔚 of August)

## Long term goals and vision
- Implement reinforcement learning 📘 (specifically genetic 🧬 algorithms) for the AI in order to allow for quicker realisation of chess ♟️ conepts through gameplay.
- V2 will feature chess ♟️ games 🎮 that are randomly generated instead of taking from games that are played by humans before, to fully achieve TrueZero.
- Create a [Data Engine](https://www.youtube.com/watch?v=zPH5O8hRfMA) where games are taken automatically and put to training DB and newly trained AIs can play against each other 24/7

## What each file 📁 does
### Internal testing (non-UCI compliant)
- `ai5.pyx` - used for internal testing, 🧪📝 not a UCI compliant way of running the Engine
- `aimatchup.pyx` - used for internal testing, 🧪📝 used as companion code to `aieval8` to faciliate playing games against other agents
### Prototype UCI compliant code
- `main.py` - run this file 📁 for the "work-in-progress" 🏗️ experience of the UCI Engine 
- `aiucieval.py` - UCI compliant version of the code that handles move search 🔍 and evaluation
- `aiuci.py` - UCI compliant, contains the logic for handling UCI commands
### Source code for AI
- `a1NAB2.pyx` - prototype code using reinforcement learning and genetic algorithm for the chess AI to learn to play chess using Negamax using Alpha-Beta pruning. 
- `aidata.py` - the companion file for `aieval7m.pyx`. Handles SQL data accessing and one-hot ♨️ encoding seperately in this file to improve speed and memory performance
- `aieval8.pyx` - used for training the evaluation AI. Handles SQL data accessing and one-hot ♨️ encoding as well. Includes ranking agents by playing games agianst other agents
- `aieval7sc.pyx` - used for training the evaluation AI. Handles SQL data accessing and one-hot ♨️ encoding as well. 
- `aieval7sc.pyx` - used for training the evaluation AI. Handles SQL data accessing and one-hot ♨️ encoding as well. Includes support for GPU.
- `aieval8t.pyx` - used for training the evaluation AI by using state-of-the-art [Transformer architecture](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)). Handles SQL data accessing and one-hot ♨️ encoding as well.
### Training data code and sample dataset
- `aigamesdb.pyx` - used for turning PGNs from Lichess to UCI notation and storing it to the SQL database (with parameterised inputs)
- `aigamesrand.pyx` - turning random 🎲 game PGNs to UCI notation and storing 🏬 it to the SQL database (with parameterised inputs)
- `randmovegen.pyx` - used for generating random 🔀 games 👾 in PGN notation. Saves 📑 games 🎮🏏 as .pgn file
- `aitraitgen.pyx` - used for generating and storing inital traits for the agents for genetic algorithm
- `fracchess.db` - a sample, smaller database that mirrors 🪞 the actual database used in training containing a smaller number of games 🎲 from Lichess

## Libraries/technologies used 🔨
This Python 🐍 Engine uses the following:
- **Pytorch** - used for creating 🔨 NN
- **Numpy** - used for processing data 💽 (chess board representation after one-hot encoding, handling final outcome and final game 🎮 👾 result
- **Scikitlearn** - used minimally for splitting data 💽 into train/validation sets (will be replaced with Pytorch DataLoader in the  future)
- **Python Chess** - used for handling board-related code
- **Cython** - used for running files 📁🗃️📂 at faster speeds 🚅 instead of running on Vanilla Python 🐍
- **Setuptools** - used in tandem with Cython to Cythonise the Python 🐍 code
- **SQLite3** - used for writing/accessing data 💽 to theSQL database
- **tqdm** - used as progress 🚧 bar 📊
- **multiprocessing** - used for parallelisation of code

*database not uploaded to this GitHub repository
