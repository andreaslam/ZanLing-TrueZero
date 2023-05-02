# zan1ling4 | 真零 | (pronounced Jun Ling)
# imports
import numpy as np
import chess
from chess import Move
import torch
from torch import nn
from torch import optim
import torch.nn.init as init
import tqdm
import copy

def negamax_ab(board, alpha, beta, colour, model, depth=2):
    if depth == 0 or board.is_game_over():  # check if the depth is 0 or "terminal node"
        if colour == 1:
            move_turn = 0  # my eval accepts 0 for white and black for 1 :/
        else:
            move_turn = 1
        matrix_game = np.array([board_data(board)])  # game after one hot encoding
        matrix_game = np.concatenate(
            (matrix_game, np.array([[move_turn]])), axis=1
        )  # have to append the move turn - the AI needs to know this
        matrix_game = torch.tensor(matrix_game, dtype=torch.float32)
        score = model(
            matrix_game
        )  # EVALUTATION - high score for winning (if white/black wins, high score, vice versa)
        score = float(score)
        if board.is_game_over():
            score = 2
        return score * colour

    child_nodes = list(board.legal_moves)
    # child_nodes = order_moves(child_nodes) # make an ordermove function
    best_score = -np.inf
    for child in child_nodes:
        board.push(child)  # Push the current child move on the board
        score = -negamax_ab(board, -beta, -alpha, -colour, model, depth - 1)
        board.pop()  # Pop the current child move from the board

        best_score = max(best_score, score)
        alpha = max(alpha, best_score)
        if alpha >= beta:
            break

    return best_score


class Tanh200(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 200)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(833, 512)
        self.dropout1 = nn.Dropout(p=0.25)
        self.relu = nn.LeakyReLU(0.05)
        self.layer2 = nn.Linear(512, 1)
        self.dropout2 = nn.Dropout(p=0.25)
        self.tanh200 = Tanh200()
        self.hidden_layers = nn.ModuleList()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.5
        )
        init.uniform_(self.fc1.weight, -1, 1)
        init.uniform_(self.layer2.weight, -1, 1)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.tanh200(x)
        return x

    def generate_move(self, board, depth=3):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        if board.turn == chess.WHITE:
            colour = 1
        else:
            colour = -1
        best_move = None
        m_dict = {}
        for move in legal_moves:
            board.push(move)
            move_score = negamax_ab(board, np.inf, np.inf, colour, self, depth)
            m_dict[str(move)] = move_score
            board.pop()
        m_dict = {
            k: v
            for k, v in sorted(
                m_dict.items(), key=lambda item: item[1], reverse=True
            )  # reverse=False to find the best move with highest score
        }
        best_move = list(m_dict.keys())[0]  # best move, first key
        del m_dict
        return best_move


def board_data(board):
    board_array = np.zeros((8, 8, 13), dtype=np.int8)
    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            color = int(piece.color)
            piece_type = piece.piece_type - 1
            board_array[i // 8][i % 8][piece_type + 6 * color] = 1
        else:
            board_array[i // 8][i % 8][-1] = 1
    board_array = board_array.flatten()
    return board_array


def play_game(agent1, agent2):
    board = chess.Board()
    batch_inputs = []
    batch_targets = []
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = agent1.generate_move(board)
        else:
            move = agent2.generate_move(board)
        move = Move.from_uci(move)
        board.push(move)
        if board.turn == chess.WHITE:
            move_turn = 0
            batch_inputs.append(
                np.concatenate(
                    (np.array([board_data(board)]), np.array([[move_turn]])),
                    axis=1,
                )
            )
            batch_targets.append(
                agent1.forward(
                    torch.tensor(
                        np.concatenate(
                            (
                                np.array([board_data(board)]),
                                np.array([[move_turn]]),
                            ),
                            axis=1,
                        ),
                        dtype=torch.float,
                    )
                ).item()
            )
        else:
            move_turn = 1
            batch_inputs.append(
                np.concatenate(
                    (np.array([board_data(board)]), np.array([[move_turn]])),
                    axis=1,
                )
            )
            batch_targets.append(
                agent2.forward(
                    torch.tensor(
                        np.concatenate(
                            (
                                np.array([board_data(board)]),
                                np.array([[move_turn]]),
                            ),
                            axis=1,
                        ),
                        dtype=torch.float,
                    )
                ).item()
            )
    # Append final target value when game is over
    if board.result() == "1-0":
        score = 1
    elif board.result() == "0-1":
        score = -1
    else:
        score = 0
    batch_targets[-1] = score  # set the last target value to score
    # Train the agents on the batch
    if len(batch_inputs) > 0:
        batch_inputs = np.array(batch_inputs)
        batch_targets = np.array([[batch_targets]])
        inputs = torch.tensor(batch_inputs, dtype=torch.float)
        targets = torch.tensor(batch_targets, dtype=torch.float)
        targets = torch.reshape(targets, (-1, 1, 1))
        outputs = agent1.forward(inputs)
        loss = agent1.loss(outputs, targets)
        loss.backward()
        # NOTE: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        agent1.optimizer.zero_grad()
        agent1.optimizer.step()
        agent1.scheduler.step()
        outputs = agent2.forward(inputs)
        loss = agent2.loss(outputs, targets)
        loss.backward()
        agent2.optimizer.zero_grad()
        agent2.optimizer.step()
        agent2.scheduler.step()
    board = chess.Board()
    batch_inputs = []
    batch_targets = []
    return score


def play_game_tournament(population):
    POP_SIZE = len(population)
    p = [Agent() for _ in range(POP_SIZE)]
    for agent in range(POP_SIZE): # load weights
        best_weights = copy.deepcopy(population[agent].state_dict())
        p[agent].load_state_dict(best_weights)
    population = p
    s_table = {}  # TODO: implement fairer logic for evaluation
    for x in range(POP_SIZE):
        s_table[x] = 0
    for i in tqdm.tqdm(range(POP_SIZE), desc="tournament"):
        for j in range(POP_SIZE):
            if i != j:
                raw_score = play_game(population[i], population[j])
                if raw_score == 1:
                    s_table[i] += 1
                elif raw_score == -1:
                    s_table[j] += raw_score * -1
    for x in s_table:
        s_table[x] = s_table[x]/((POP_SIZE)**2)
    print(s_table)
    return s_table