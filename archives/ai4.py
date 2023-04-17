# imports
import numpy as np
import torch
import torch.nn as nn
# import tqdm
import chess

# set up AI (Sequential NN)

class Tanh200(nn.Module):
    def forward(self, x):
        return torch.tanh(x/200)

model = nn.Sequential(
    nn.Linear(833, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.Dropout(p=0.25),
    nn.BatchNorm1d(512),
    nn.LeakyReLU(0.075),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.Dropout(p=0.25),
    nn.BatchNorm1d(128),
    nn.LeakyReLU(0.075),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Dropout(p=0.25),
    Tanh200()
)

model.eval()
weights_path = "./zl.pt"
state_dict = torch.load(weights_path)
model.load_state_dict(state_dict)


# set up important functions
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
    node,
    alpha,
    beta,
    board,
    move_count,
    move,
    colour,
    depth=2,
):
    if depth == 0:
        with torch.no_grad():
            move_turn = colour
            matrix_game = np.array([board_data(board)])  # game after one hot encoding
            matrix_game = np.concatenate((matrix_game, np.array([[move_turn]])), axis=1)
            matrix_game = torch.tensor(matrix_game, dtype=torch.float32)
            best_val = model(matrix_game)
        best_val = float(best_val)
        return best_val, move
    child_nodes = list(board.legal_moves)
    # child_nodes = ordermoves(child_nodes) # make an ordermove function
    value = -np.inf
    indexer = 0
    best_move = ""
    for child in child_nodes:
        test_board = board.copy()
        test_board.push(child)
        str_move = str(child)
        child = board_data(test_board)
        x, y = negamax_ab(
            child,
            -beta,
            -alpha,
            board,
            move_count,
            str_move,
            -colour,
            depth - 1,
        )
        v = float(x)
        if -v > value:
            value = -v
            best_move = y
        alpha = max(alpha, value)
        test_board.pop()
        if alpha >= beta:
            break
        indexer += 1
    return value, best_move  # best_move is the best move


# set up chess game

NUMBER_OF_GAMES = 10


def play_game(is_vs_humans=False, *args):
    if is_vs_humans == False:
        NUMBER_OF_GAMES = int(args[0])
        for _ in range(NUMBER_OF_GAMES):
            board = chess.Board()
            legal_moves = board.legal_moves
            raw_board = board_data(board)
            move_count = 0
            for __ in range(3): # random opening, 3 moves
                legal_moves = list(board.legal_moves)
                chosen_move = np.random.randint(0, len(legal_moves))
                board.push(legal_moves[chosen_move])
                with open("ai_games.txt", "a+") as f:
                    f.write(str(legal_moves[chosen_move]))
                    f.write(" ")
            while not board.is_game_over():
                if move_count % 2 == 0:
                    colour = 1  # white
                else:
                    colour = -1  # black
                _, move = negamax_ab(
                    raw_board, -np.inf, np.inf, board, move_count, "", colour, 2
                )
                with open("ai_games.txt", "a+") as f:
                    f.write(move)
                    f.write(" ")
                board.push(chess.Move.from_uci(move))
                move_count += 1
            with open("ai_games.txt", "a+") as f:
                f.write(board.result())
                f.write("\n")
    else:
        board = chess.Board()
        legal_moves = list(board.legal_moves)
        raw_board = board_data(board)

        def call():
            non_zl_colour = input("Play as White (0) or Black (1): ")
            return non_zl_colour

        def move_call(board, non_zl_colour):
            if int(non_zl_colour) % 2 == 1:
                print(board.mirror())
            else:
                print(board)
            legal_moves = list(board.legal_moves)
            formatted = ""
            for x in legal_moves:
                formatted = formatted + " " + str(x)
            non_zl_move = input("Non-ZL, moves to choose from:" + formatted + ": ")
            try:
                return chess.Move.from_uci(non_zl_move)
            except chess.InvalidMoveError:
                move_call(board, non_zl_colour)

        non_zl_colour = call()
        while non_zl_colour not in ["0", "1"]:
            non_zl_colour = call()
        zl_colour = "0"
        if non_zl_colour == "0":
            zl_colour = "1"
        else:
            zl_colour = "0"
        move_count = 0
        while not board.is_game_over():
            if move_count % 2 == 0:  # white
                if int(non_zl_colour) % 2 == 0:  # who's white?
                    move = move_call(board, non_zl_colour)
                    while move not in list(board.legal_moves):
                        move = move_call(board, non_zl_colour)
                    board.push(move)
                elif int(zl_colour) % 2 == 0:
                    _, move = negamax_ab(
                        raw_board,
                        -np.inf,
                        np.inf,
                        board,
                        move_count,
                        "",
                        int(zl_colour),
                        2,
                    )
                    board.push(chess.Move.from_uci(move))
            else:  # black
                if int(non_zl_colour) % 2 == 1:  # who's black?
                    move = move_call(board, non_zl_colour)
                    while move not in legal_moves:
                        move = move_call(board, non_zl_colour)
                    board.push(move)
                elif int(zl_colour) % 2 == 1:
                    _, move = negamax_ab(
                        raw_board,
                        -np.inf,
                        np.inf,
                        board,
                        move_count,
                        "",
                        int(zl_colour),
                        2,
                    )
                    board.push(chess.Move.from_uci(move))
            move_count += 1


if __name__ == "__main__":
    play_game(False, NUMBER_OF_GAMES)
