# imports
import numpy as np
import torch
import torch.nn as nn
import chess
from chess import Move
import random

# set up AI (Sequential NN)


class Tanh200(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 200)


model = nn.Sequential(
    nn.Linear(833, 512),
    nn.Dropout(p=0.25),
    nn.ReLU(),
    nn.Linear(512, 1),
    nn.Dropout(p=0.25),
    Tanh200(),
)

model.eval()
weights_path = "./zlv6_1.pt"
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


# set up chess game

NUMBER_OF_GAMES = 10


# MCTS Node class
class MCTSNode:
    def __init__(self, move=None, parent=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0


# MCTS function
def mcts(board, model, n_simulations):
    root = MCTSNode()

    def select(node):  # pick move
        while node.children:
            node = max(
                node.children,
                key=lambda x: x.score
                + 1.4 * np.sqrt(np.log(node.visits + 1) / (x.visits + 1e-6)),
            )
        return node

    def expand(node):
        legal_moves = list(board.legal_moves)
        random.shuffle(legal_moves)
        for move in legal_moves:
            child = MCTSNode(move, node)
            node.children.append(child)

    def simulate(node):
        if node.move is None:
            return 0
        test_board = board.copy()
        test_board.push(node.move)
        turn = board.turn
        move_count = 0
        colour = 1
        # get the current turn as 0 or 1
        if turn == chess.WHITE:
            move_count = 0
            colour = 1
        elif turn == chess.BLACK:
            move_count = 1
            colour = -1
        matrix_game = np.array([board_data(board)])  # game after one hot encoding
        matrix_game = np.concatenate(
            (matrix_game, np.array([[move_count]])), axis=1
        )  # have to append the move turn - the AI needs to know this
        matrix_game = torch.tensor(matrix_game, dtype=torch.float32)
        best_val = model(matrix_game)
        return float(best_val)

    def backpropagate(node, score):
        while node is not None:
            node.visits += 1
            node.score += score
            node = node.parent

    for _ in range(n_simulations):
        selected_node = select(root)
        expand(selected_node)
        score = simulate(selected_node)
        backpropagate(selected_node, score)

    best_move = max(root.children, key=lambda x: x.visits).move
    return best_move


def play_game(NUMBER_OF_GAMES):
    for _ in range(NUMBER_OF_GAMES):
        board = chess.Board()
        legal_moves = board.legal_moves
        move_count = 0
        for __ in range(1):  # random opening, 1 move
            legal_moves = list(board.legal_moves)
            chosen_move = np.random.randint(0, len(legal_moves))
            board.push(legal_moves[chosen_move])
            with open("ai_gamesmcts.txt", "a+") as f:
                f.write(str(legal_moves[chosen_move]))
                f.write(" ")
            print(str(legal_moves[chosen_move]))
            move_count += 1
        while not board.is_game_over():
            best_move = mcts(board, model, n_simulations=5000)
            print(str(best_move))
            with open("ai_gamesmcts.txt", "a+") as f:
                f.write(str(best_move))
                f.write(" ")
            board.push(best_move)
            move_count += 1
        with open("ai_gamesmcts.txt", "a+") as f:
            f.write(board.result())
            f.write("\n")


play_game(NUMBER_OF_GAMES)
