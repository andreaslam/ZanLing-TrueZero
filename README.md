# ZanLing-TrueZero
A python chess engine that starts from Zero. This project is still very much work-in-progress.

## About the Engine
The name of the Engine is Zan1Ling4, which is taken from 真零, which is Chinese for "True Zero" and romanised using [Jyutping](https://en.wikipedia.org/wiki/Jyutping) for Cantonese. 

Instead of being hard-coded by humans, this AI learns how to play through playing games from itself and learning from randomly generated games. 

The evaluation learns from randomly generated games with outcomes and move turn and based on move turn and final outcome, assign a score between 1/0/-1 for games that were won/lost/drew respectively, accounting for move turn. 

The chess Engine will then play games against itself using the evaluation to evalutate chess positions, done using [Negamax with Alpha Beta pruning](https://en.wikipedia.org/wiki/Negamax#Negamax_with_alpha_beta_pruning).
