# imports
import numpy as np
import torch
import torch.nn as nn
import chess
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
        self.dropout1 = nn.Dropout(p=0.45)
        self.relu = nn.LeakyReLU(0.05)
        self.layer2 = nn.Linear(2048, 1)
        self.dropout2 = nn.Dropout(p=0.45)
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
weights_path = "./zlv7_pro.pt"
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


def static_eval(board):
    matrix_game = np.array([board_data(board)])
    matrix_game = np.concatenate(
        (matrix_game, np.array([[0]])), axis=1
    )  # 0 because static evals typically returns evaluation from white's POV
    matrix_game = torch.tensor(matrix_game, dtype=torch.float32)
    score = model(matrix_game)
    score = float(score)
    return score


def negamax_ab(board, alpha, beta, colour, model, depth=2):
    if depth == 0 or board.is_game_over():  # check if the depth is 0 or "terminal node"
        if colour == 1:
            move_turn = 0  # my eval accepts 0 for white and black for 1 :/
        else:
            move_turn = 1
        # game after one hot encoding
        matrix_game = np.array([board_data(board)])
        matrix_game = np.concatenate(
            (matrix_game, np.array([[move_turn]])), axis=1
        )  # have to append the move turn - the AI needs to know this
        matrix_game = torch.tensor(matrix_game, dtype=torch.float32)
        model.eval()
        score = model(
            matrix_game
        )  # EVALUTATION - high score for winning (if white/black wins, high score, vice versa)
        score = float(score)
        if board.is_game_over():
            return score * colour * (1 if board.turn == chess.WHITE else -1)
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

# set up chess game

NUMBER_OF_GAMES = 10


# NOTE: this function analyses moves by NUMBER OF NODES SEARCHED
def analyse_move_nodes(board, ready, max_nodes):
    if ready:
        legal_moves = list(board.legal_moves)
        turn = board.turn
        colour = 1
        if turn == chess.WHITE:
            colour = 1
        else:
            colour = -1
        m_dict = {}
        n_count = 1
        max_nodes = min(len(legal_moves), max_nodes)
        for move in legal_moves:
            if n_count < max_nodes:
                board.push(move)
                move_score = negamax_ab(
                    board, -np.inf, np.inf, -colour, 2)
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
                        )  # reverse=False to find the best move with highest score
                    }
                best_move = list(m_dict.keys())[0]  # best move, first key
                board.pop()
                n_count += 1
        m = iter(m_dict)
        best_move = next(m)  # first move after sorting, best move
        colour = colour * -1  # TODO: check if this line is needed
        return best_move


# NOTE: this function analyses moves by DEPTH
def analyse_move(board, ready, depth=2):
    if ready:
        legal_moves = list(board.legal_moves)
        turn = board.turn
        colour = 0
        if turn == chess.WHITE:
            colour = 1
        else:
            colour = -1
        m_dict = {}
        for move in legal_moves:
            board.push(move)
            move_score = negamax_ab(board, -np.inf, np.inf, colour, model,  depth)
            m_dict[str(move)] = move_score
            board.pop()
        m_dict = {
            k: v
            for k, v in sorted(m_dict.items(), key=lambda item: item[1], reverse=False)
        }  # reverse=True to find best move with highest score
        m = iter(m_dict)
        best_move = next(m)  # first move after sorting, best move
        colour = colour * -1  # TODO: check if this line is needed
        return best_move
