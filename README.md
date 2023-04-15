# ZanLing-TrueZero
A Python 🐍 chess ♟️ engine that starts 🆕 from Zero 0️⃣.  This project is still very much work-in-progress.

## About the Engine 
The name of the Engine is Zan1Ling4, which is taken from 真零, which is Chinese for  "True Zero" and romanised using [Jyutping](https://en.wikipedia.org/wiki/Jyutping) for Cantonese.

Instead of being hard-coded by humans, this AI learns 🎓 how to play ⏯️ through playing games 🎮 from itself and learning 📖 from randomly generated games. 

The evaluation learns 📕 from randomly generated games with outcomes and move turn and based on move turn and final outcome, assign a score 💯 between 1/0/-1 for games that were won/lost/drew respectively, accounting 🧾 for move turn.

The chess ♟️ Engine will then play ▶️ games against itself using the evaluation to evalutate chess ♟️ positions, done using  [Negamax with Alpha Beta pruning](https://en.wikipedia.org/wiki/Negamax#Negamax_with_alpha_beta_pruning).

## How all this works in more detail 🔎
The Evaluation AI is trained on a SQL database* containing games 🎮 that were previously played 👾 on [Lichess](lichess.com) (taken [here](https://database.lichess.org/)). The AI is then trained on 20 randomly selected board states 🇺🇸 of each game, given nothing (no prerequisite knowledge) but the board  state (after one-hot  encoding) and turn to move (and nothing else) to predict final outcome, with 1,0 and -1 denoting a winning, 🎖️ an equal outcome (draw) and a losing 🏳️ game for relative to side to move. This regression model is then fed to the Engine, 🚒 where it looks 👀 for optimal moves using 🇻🇮 Negamax with 
Beta Pruning, giving an evaluation for each position.

## Current todo and goals 
- Train V1 Evaluation Engine (with at least 2+ million games 🎮 analysed) (ETA - End 🔚 of May)
- Have fully functional UCI compliance (ETA - End 🔚 of June/start of July)
- Implement Move Ordering (ETA - mid-July)
- Implement Transposition tables, killer moves 🚶 (ETA - August)
- TrueZero V1 (ETA - End 🔚 of August)

## What each file 📁 does
- `ai5.pyx` - used for internal testing, 🧪📝 not a UCI compliant way of running the Engine
- `aiuci.py` - UCI compliant, contains the logic for handling UCI commands
- `main.py` - run this file 📁 for the "work-in-progress" 🏗️ experience of the UCI Engine 
- `aiucieval.py` - UCI compliant version of the code that handles move search 🔍 and evaluation
- `aieval6.pyx` - used for training the evaluation AI. Handles SQL data accessing and one-hot encoding as well

## Libraries/technologies used 🔨
This Python 🐍 Engine uses the following:
- **Pytorch** - used for creating 🔨 NN
- **Numpy** - used for processing data (chess board representation after one-hot encoding, handling final outcome and final game 👾 result
- **Scikitlearn** - used minimally for splitting data into train/validation sets (will be replaced with Pytorch DataLoader in the future)
- **Python Chess** - used for handling 🙋‍♂️ board-related code
- **Cython** - used🇮 for running files at faster speeds 🚅 instead of running 🏃‍♂️ on Vanilla Python 🐍
- **Setuptools** - used in tandem with Cython to Cythonise the Python 🐍 code
- **SQLite3** - used for writing/accessing data to the SQL database

*database not uploaded to this GitHub repository
