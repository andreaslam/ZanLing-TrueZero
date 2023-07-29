# zan1ling4 | 真零 | (pronounced Jun Ling)
# imports
import numpy as np
import chess
import torch
import sqlite3
from chess import Move
from sklearn.model_selection import train_test_split
import sys
import tqdm
import pickle
import psutil
import gc

gc.disable()
# import torch.cuda
# from memory_profiler import profile

# from torch.cuda.amp import GradScaler  # NEED GPU


class DataManager:
    def __init__(self, training_size, completed):
        self.training_size = training_size
        self.completed = completed

    def get_status(self, move_turn, result):
        status = 0
        if (move_turn == 0 and result == "1") or (move_turn == 1 and result == "-1"):
            status = 1
        elif (move_turn == 0 and result == "-1") or (move_turn == 1 and result == "1"):
            status = -1
        elif result == "0":
            status = 0
        return status

    def board_data(self, board):
        board_array = np.zeros((8, 8, 13), dtype=np.float32)
        for i in range(64):
            piece = board.piece_at(i)
            if piece is not None:
                color = int(piece.color)
                piece_type = piece.piece_type - 1
                board_array[i // 8][i % 8][piece_type + 6 * color] = 1
            else:
                board_array[i // 8][i % 8][-1] = 1
        board_array = board_array.flatten()
        # board_array.shape = 832 (1D)
        yield board_array

    def load(self, completed, size, DB_LOCATION):
        conn = sqlite3.connect(DB_LOCATION)
        cursor = conn.cursor()
        # select all rows from the table
        cursor.execute("SELECT * FROM games LIMIT ? OFFSET ?", (size, completed))
        # Execute the SQL query to select the next 1000 rows after skipping the first 1000
        # replace reading from file with reading from SQLite
        games = cursor.fetchall()

        for g in tqdm.tqdm(games, desc="each game"):
            game = g[2]
            MAX_MOMENTS = min(2, len(game) - 10)
            try:
                unsampled_idx = [
                    np.random.randint(20, len(game)) for _ in range(MAX_MOMENTS - 1)
                ]
            except ValueError:
                unsampled_idx = [
                    np.random.randint(len(game), 20) for _ in range(MAX_MOMENTS - 1)
                ]
            game = game.split(" ")
            for move in range(MAX_MOMENTS - 1):
                board = chess.Board()
                up_to = unsampled_idx[move]
                moment = game[:up_to]
                for counter, move in enumerate(moment):
                    move = Move.from_uci(move)
                    board.push(move)
                matrix_game = np.array(
                    [self.board_data(board).__next__()]
                )  # game after one hot encoding
                # Add move_turn as a feature to matrix_game
                move_turn = (
                    counter % 2
                )  # TODO: change this so that it outputs 1/-1, which is the same as the status instead of 0,1
                matrix_game = np.concatenate(
                    (matrix_game, np.array([[move_turn]])), axis=1
                )
                yield matrix_game.flatten(), self.get_status(move_turn, g[1])

        conn.close()
        del games

    def loading(self, train_data, train_target):
        train_data, train_target = np.array(train_data), np.array(train_target)
        X_train, X_val, y_train, y_val = train_test_split(
            train_data, train_target, test_size=0.2, shuffle=True
        )
        del train_data
        del train_target
        gc.enable()
        gc.collect()
        gc.disable()
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        return X_train, y_train, X_val, y_val


DB_LOCATION = "./all_data.db"
completed, size = int(sys.argv[1]), int(sys.argv[2])
d = DataManager(completed, size)
train_data, train_target = zip(*d.load(completed, size, DB_LOCATION))
X_train, y_train, X_val, y_val = d.loading(train_data, train_target)
data = [X_train, y_train, X_val, y_val]
files = ["X_train", "y_train", "X_val", "y_val"]

for d, file in zip(data, files):
    storagefile = open(file,"ab")
    pickle.dump(d, storagefile)

test = sys.argv[3]
if test == "True":
    memory_available_after = psutil.virtual_memory().available/ psutil.virtual_memory().total # in percent
    print(memory_available_after)
    for file in files:
        f = open(file, "w+")
        f.write("")
    del memory_available_after
del X_train
del y_train
del X_val
del y_val
del data
del files
del storagefile
del test
del DataManager

gc.enable()
gc.collect()
