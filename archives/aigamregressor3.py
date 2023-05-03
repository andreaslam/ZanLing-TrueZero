# RANDOM FOREST NETWORK THAT IS USED TO CLASSIFY GAMES
# zan1ling4 真零
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import graphviz
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import chess
import joblib
import sqlite3
from chess import Move
import tqdm
import pandas as pd

# puzzle presets
MAX_MOMENTS = 20
TOTAL_GAMES = 4
board = chess.Board()
# pre-training loop
completed = 0
# find number of lines in a database

DB_LOCATION = "./fracchess_games.db"

# Connect to the database
conn = sqlite3.connect(
    DB_LOCATION
)  # TODO: implement a variable to replace manually entering DB address

# Create a cursor object
cursor = conn.cursor()

# # Execute the query to get the length of the table
cursor.execute("SELECT COUNT(*) FROM games")

# # Fetch the result
result = cursor.fetchone()[0]
# print(result)
conn.close()
size = 20


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

    def load(self, completed, size):
        conn = sqlite3.connect(DB_LOCATION)
        cursor = conn.cursor()
        # select all rows from the table
        cursor.execute("SELECT * FROM games LIMIT ? OFFSET ?", (size, completed))
        # Execute the SQL query to select the next 1000 rows after skipping the first 1000
        # replace reading from file with reading from SQLite
        games = cursor.fetchall()

        for g in tqdm.tqdm(games, desc="each game"):
            game = g[2]
            MAX_MOMENTS = min(20, len(game) - 10)
            unsampled_idx = [
                np.random.randint(10, len(game)) for _ in range(MAX_MOMENTS - 1)
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
        return X_train, y_train, X_val, y_val


# create/load random forest
class Train:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def cycle(self,X_train, y_train, X_val, y_val):
        try:
            rf = joblib.load("forest.joblib")
            print("Forest loaded")
        except FileNotFoundError:
            rf = RandomForestRegressor(n_estimators=1000)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        print("PREDICTIONS", y_pred)

        # Export the first three decision trees from the forest
        param_dist = {"n_estimators": randint(50, 500), "max_depth": randint(1, 20)}

        # Use random search to find the best hyperparameters
        rand_search = RandomizedSearchCV(
            rf, param_distributions=param_dist, n_iter=5, cv=5
        )

        # Fit the random search object to the data
        rand_search.fit(X_train, y_train)

        # Create a variable for the best model
        best_rf = rand_search.best_estimator_

        # Print the best hyperparameters
        print("Best hyperparameters:", rand_search.best_params_)
        # Generate predictions with the best model
        y_pred = best_rf.predict(X_val)

        # Create a series containing feature importances from the model and feature names from the training data
        joblib.dump(best_rf, "forest.joblib")

all_completed = False
counter = 0

try:
    with open("progressY.txt", "r") as f:
        contents = f.read()
except FileNotFoundError:
    with open("progressY.txt", "w+") as f:  # create the file if it does not exist
        f.write(
            "0 " + str(size)
        )  # 0 means 0 games processed; starting from scratch, size is number of games to process in one cycle

while all_completed == False:
    with open("progressY.txt", "r+") as f:
        contents = f.read()
    contents = contents.split(" ")
    completed, size = int(contents[0]), int(contents[1])
    not_done = result - completed
    if not_done == 0:
        all_completed = True
        break
    if not_done < size:  # size is the number of chess games processed/cycle in total
        size = not_done  # this is for the final cycle if there are any remainders
        all_completed = True
    print("SIZE", size)
    print("COMPLETED", completed)
    d = DataManager(size, 0)
    train_data, train_target = None, None  # served for clearing variable in loops
    train_data, train_target = zip(*d.load(completed, size))
    X_train, y_train, X_val, y_val = d.loading(train_data, train_target)
    t = Train(X_train, y_train, X_val, y_val)
    t.cycle(X_train, y_train, X_val, y_val)
    completed = completed + size
    with open("progressY.txt", "w") as f:  # overwrite file contents
        f.write(str(completed) + " " + str(size))
    completed = counter * size
    del d
    del t
    del X_train
    del y_train
    del X_val
    del y_val
    counter += 1
