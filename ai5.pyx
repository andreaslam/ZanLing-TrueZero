# imports
import numpy as np
import torch
import torch.nn as nn
import chess
from chess import Move
import torch.nn.init as init

# set up AI (Sequential NN)

class Tanh200(nn.Module):
    def __init__(self):
        super(Tanh200, self).__init__()

    def forward(self, x):
        return torch.tanh(x / 200)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(833, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(p=0.35)
        self.relu = nn.LeakyReLU(0.05)
        self.layer2 = nn.Linear(2048, 1)
        self.dropout2 = nn.Dropout(p=0.35)
        self.tanh200 = Tanh200()
        self.hidden_layers = nn.ModuleList()

        # Initialize weights of Linear layers using Xavier initialization
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.layer2.weight)

        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.tanh200(x)

        return x


model = Agent()
model.eval()
weights_path = "./zlv7_3m.pt"
state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)


# set up important functions
# board_data - this function will turn a python-chess board into a matrix for the AI to evaluate
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


def negamax_ab(board, alpha, beta, colour, depth=2):
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
        with torch.no_grad():
            score = model(
                matrix_game
            )  # EVALUTATION - high score for winning (if white/black wins, high score, vice versa)
        score = float(score)
        if board.is_game_over():
            return 2 * colour
        return score * colour

    child_nodes = list(board.legal_moves)
    # child_nodes = order_moves(child_nodes) # make an ordermove function
    best_score = -np.inf
    for child in child_nodes:
        board.push(child)  # Push the current child move on the board
        score = -negamax_ab(board, -beta, -alpha, -colour, depth - 1)
        board.pop()  # Pop the current child move from the board

        best_score = max(best_score, score)
        alpha = max(alpha, best_score)
        if alpha >= beta:
            break
    return best_score


# set up chess game

NUMBER_OF_GAMES = 10


def play_game(NUMBER_OF_GAMES):
    for _ in range(NUMBER_OF_GAMES):
        board = chess.Board()
        legal_moves = board.legal_moves
        colour = 1  # white starts first
        for __ in range(1):  # random opening, 1 move
            legal_moves = list(board.legal_moves)
            chosen_move = np.random.randint(0, len(legal_moves))
            board.push(legal_moves[chosen_move])
            with open("ai_gamesNN.txt", "a+") as f:
                f.write(str(legal_moves[chosen_move]))
                f.write(" ")
            colour = colour * -1
            print(str(legal_moves[chosen_move]))
        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            m_dict = {}
            for move in legal_moves:
                test_board = board.copy()
                test_board.push(move)
                move_score = negamax_ab(test_board, -np.inf, np.inf, -colour, 3)
                m_dict[str(move)] = move_score
            if colour == -1:
                m_dict = {
                    k: v
                    for k, v in sorted(
                        m_dict.items(), key=lambda item: item[1], reverse=False
                    )  # reverse=False to find the best move with highest score
                }
            else:
                m_dict = {
                    k: v
                    for k, v in sorted(
                        m_dict.items(), key=lambda item: item[1], reverse=True
                    )  # reverse=True to find the best move with highest score
                }
            best_move = list(m_dict.keys())[0]  # best move, first key
            print(m_dict)
            print(best_move)
            with open("ai_gamesNN.txt", "a+") as f:
                f.write(best_move)
                f.write(" ")
            best_move = Move.from_uci(best_move)
            board.push(best_move)
            colour = colour * -1
        with open("ai_gamesNN.txt", "a+") as f:
            f.write(board.result())
            f.write("\n")


play_game(NUMBER_OF_GAMES)
