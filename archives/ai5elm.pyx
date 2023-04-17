# imports
import numpy as np
import torch
import torch.nn as nn
import chess
from chess import Move

# set up AI (Extreme Learning Machine)


class Tanh200(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 200)

class ELM(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(ELM, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.weight_input = torch.randn(n_input, n_hidden)
        self.bias_hidden = torch.randn(1, n_hidden)
        self.weight_output = nn.Parameter(torch.randn(n_hidden, n_output).squeeze())
        self.bias_output = torch.randn(1, n_output)

    def forward(self, x):
        # Compute hidden layer output
        hidden = torch.matmul(x, self.weight_input) + self.bias_hidden
        hidden = torch.sigmoid(hidden)

        # Compute output layer output
        output = torch.matmul(hidden, self.weight_output) + self.bias_output
        output = torch.sigmoid(output)

        return output

    def fit(self, x, y):
        # Convert data to PyTorch tensors

        # Compute hidden layer output
        hidden = torch.matmul(x, self.weight_input) + self.bias_hidden
        hidden = torch.sigmoid(hidden)

        # Compute output layer weights using pseudoinverse
        pinv = torch.pinverse(hidden)
        self.weight_output = nn.Parameter(torch.matmul(pinv, y))

    def predict(self, X):
        # Compute the output of the network for the given input
        output = self.forward(X)
        return output

model = ELM(833, 5000, 1)

model.eval()
weights_path = "./elm_model_weights1.pth"
state_dict = torch.load(weights_path)
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


def negamax_ab(
    board,
    alpha,
    beta,
    move_count,
    depth=2,
):
    if depth == 0:
        if move_count % 2 == 0: # white
            move_count = 0
        else:
            move_count = 1
        matrix_game = np.array([board_data(board)])  # game after one hot encoding
        matrix_game = np.concatenate(
            (matrix_game, np.array([[move_count]])), axis=1
        )  # have to append the move turn - the AI needs to know this
        matrix_game = torch.tensor(matrix_game, dtype=torch.float32)
        best_val = model(matrix_game)
        best_val = float(best_val)
        return best_val
    child_nodes = list(board.legal_moves)
    # child_nodes = ordermoves(child_nodes) # make an ordermove function
    best_score = -np.inf
    indexer = 0
    for child in child_nodes:
        test_board = board.copy()  # Create a copy of the board
        test_board.push(child)  # Push the current child move on the test_board
        score = negamax_ab(
            test_board, -beta, -alpha, move_count + 1, depth - 1
        )
        score = -float(score)
        best_score = max(best_score, score)
        alpha = max(alpha, best_score)
        test_board.pop()  # Pop the current child move from the test_board
        if alpha >= beta:
            break

        indexer += 1
    return best_score


# set up chess game

NUMBER_OF_GAMES = 10


def play_game(NUMBER_OF_GAMES):
    for _ in range(NUMBER_OF_GAMES):
        board = chess.Board()
        legal_moves = board.legal_moves
        move_count = 0
        for __ in range(1):  # random opening, 1 move
            legal_moves = list(board.legal_moves)
            chosen_move = np.random.randint(0, len(legal_moves))
            board.push(legal_moves[chosen_move])
            with open("ai_games.txt", "a+") as f:
                f.write(str(legal_moves[chosen_move]))
                f.write(" ")
            move_count += 1
        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            m_dict = {}
            for move in legal_moves:
                test_board = board.copy()
                test_board.push(move)
                move_score = negamax_ab(
                    test_board, -np.inf, np.inf, move_count, 2
                )
                m_dict[str(move)] = move_score
                m_dict = {
                    k: v
                    for k, v in sorted(
                        m_dict.items(), key=lambda item: item[1], reverse=True
                    )
                }  # reverse=True to find best move with highest score
                m = iter(m_dict)
                best_move = next(m)
            print(m_dict)
            print(best_move)
            with open("ai_games.txt", "a+") as f:
                f.write(best_move)
                f.write(" ")
            best_move = Move.from_uci(best_move)
            board.push(best_move)
            move_count += 1
        with open("ai_games.txt", "a+") as f:
            f.write(board.result())
            f.write("\n")


play_game(NUMBER_OF_GAMES)
