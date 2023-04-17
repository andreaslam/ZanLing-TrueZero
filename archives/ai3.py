# PROTOTYPE GEN 3.0
# zan1ling4 真零 | TrueZero
# imports
import numpy as np
import chess
import pandas as pd
import joblib
import boardnotation

# set up AI (random forest)

rf = joblib.load("forest.joblib")


# set up important functions
def board_data(board):
    board_array = np.zeros((8, 8, 13), dtype=np.int64)
    move_data = {
        "p": 0,
        "k": 0,
        "q": 0,
        "r": 0,
        "n": 0,
        "b": 0,
        "empty": 0,
    }
    for i in range(64):
        piece = board.piece_at(i)
        if piece != None:
            x = str(piece).lower()
            move_data[x] = move_data[x] + 1
        else:
            move_data["empty"] = move_data["empty"] + 1
        if piece is not None:
            color = int(piece.color)
            piece_type = piece.piece_type - 1
            board_array[i // 8][i % 8][piece_type + 6 * color] = 1
        else:
            board_array[i // 8][i % 8][-1] = 1
    return board_array.flatten(), move_data


def board_format(board):
    ai_moves = list(board)
    counter = 0
    final_li = []
    temp_li = []
    while counter < len(ai_moves):
        if counter % 13 == 0 and counter > 0:
            final_li.append(temp_li)
            temp_li = []
        temp_li.append(ai_moves[counter])
        counter += 1
    store = []
    for x in final_li:
        x = [str(i) for i in x]
        x = "".join(x)
        store.append(int(x, 2))
    final_li = store
    final_li = [str(i) for i in final_li]
    final_li = "".join(final_li)
    final_data = float(final_li)
    # Define the integer value to rescale
    x = final_data

    # Define the range of values for the logarithmic scale
    start = 1
    stop = 1.1  # 2

    # Define the base of the logarithm (10 for base-10 scaling)
    base = 10

    # Rescale the integer logarithmically
    rescaled = (
        (np.log10(x) - np.log10(start))
        / (np.log10(stop) - np.log10(start))
        * (np.log10(base) ** 2)
    )
    # Return the rescaled value
    return rescaled


def append_data(node, move_count, piece_stat):
    df = pd.DataFrame(
        columns=[
            "board_data",
            "num_of_moves",
            "p",
            "k",
            "q",
            "r",
            "n",
            "b",
            "empty",
        ]
    )
    d = {
        "board_data": node,
        "num_of_moves": move_count,
        "p": piece_stat["p"],
        "k": piece_stat["k"],
        "q": piece_stat["q"],
        "r": piece_stat["r"],
        "n": piece_stat["n"],
        "b": piece_stat["b"],
        "empty": piece_stat["empty"],
    }
    df.loc[len(df)] = d
    return df


def negamax_ab(
    node,
    piece_stat,
    alpha,
    beta,
    board,
    move_count,
    move,
    colour,
    depth=2,
):
    if depth == 0:
        node = board_format(node)  # make board readable for randomforest
        df = append_data(
            node, move_count, piece_stat
        )  # function that makes data understandable for the random forest
        best_val = rf.predict(df) * colour
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
        child, piece_stat = board_data(test_board)
        x, y = negamax_ab(
            child,
            piece_stat,
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


for _ in range(NUMBER_OF_GAMES):
    board = chess.Board()
    legal_moves = board.legal_moves
    raw_board, piece_stat = board_data(board)
    move_count = 0
    df = pd.read_csv("all_openings.csv")
    df.sample(frac=1)
    opening = df.iloc[np.random.randint(0, len(df))]
    opening = str(opening["pgn"])
    right_moves = boardnotation.convert_pgn_to_lan(str(opening))
    right_moves = right_moves.split()
    final_str = ""
    for x in right_moves:
        final_str = final_str + x + " "
    final_str = final_str[:-1]
    final_str = final_str.split()
    for move in final_str:
        board.push(chess.Move.from_uci(move))
        print((move))
    while not board.is_game_over():
        if move_count % 2 == 0:
            colour = 1  # white
        else:
            colour = -1  # black
        score, move = negamax_ab(
            raw_board, piece_stat, -np.inf, np.inf, board, move_count, "", colour, 2
        )
        print(move)
        board.push(chess.Move.from_uci(move))
        move_count += 1
    print(board.result())
