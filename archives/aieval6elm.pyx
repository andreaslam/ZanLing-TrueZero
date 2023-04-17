# EXTREME LEARNING MACHINE
import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import chess
from chess import Move
import torch.nn.init as init
import sqlite3

# puzzle presets
board = chess.Board()
completed = 0
# find number of lines in a database

# Connect to the database
conn = sqlite3.connect(
    "chess_games.db"
)  # TODO: implement a variable to replace manually entering DB address

# Create a cursor object
cursor = conn.cursor()

# # Execute the query to get the length of the table
cursor.execute("SELECT COUNT(*) FROM games")

# # Fetch the result
result = cursor.fetchone()[0]
print(result)
conn.close()
size = 1000


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
        # board_array.shape = 832 (1D)
        yield board_array

    def load(self, completed, size):
        conn = sqlite3.connect("chess_games.db")
        cursor = conn.cursor()
        # select all rows from the table
        cursor.execute("SELECT * FROM games LIMIT ? OFFSET ?", (size, completed))
        # Execute the SQL query to select the next 1000 rows after skipping the first 1000
        # replace reading from file with reading from SQLite
        games = cursor.fetchall()

        for g in tqdm.tqdm(games, desc="each game"):
            game = g[2]
            MAX_MOMENTS = min(20, len(game))
            unsampled_idx = [
                np.random.randint(1, len(game)) for _ in range(MAX_MOMENTS - 1)
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
                move_turn = counter % 2
                matrix_game = np.concatenate(
                    (matrix_game, np.array([[move_turn]])), axis=1
                )
                yield matrix_game.flatten(), self.get_status(move_turn, g[1])

        conn.close()

    def loading(self, train_data, train_target):
        train_data, train_target = np.array(train_data), np.array(train_target)

        X_train, X_val, y_train, y_val = train_test_split(
            train_data, train_target, test_size=0.5, shuffle=True
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        return X_train, y_train, X_val, y_val


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
        output = torch.tanh(output)

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


# Training loop
completed = 0
counter = 1
all_completed = False
machine = ELM(833, 5000, 1)
loss_fn = nn.MSELoss()
while all_completed == False:
    not_done = result - completed
    if not_done == 0:
        all_completed = True
    if not_done < size:  # size is the number of chess games processed/cycle in total
        size = not_done  # this is for the final cycle if there are any remainders
        all_completed = True
    d = DataManager(size, 0)
    print(completed, size)
    train_data, train_target = zip(*d.load(completed, size))
    X_train, y_train, X_val, y_val = d.loading(train_data, train_target)
    # torch.Size([15200, 833]) torch.Size([15200]) torch.Size([3800, 833]) torch.Size([3800])
    machine.fit(X_train, y_train)
    preds = machine.predict(X_val)
    # accuracy = metrics.pairwise.cosine_similarity(preds.detach().numpy().reshape(-1,1), Y=y_val.detach().numpy().reshape(-1,1), dense_output=True)
    preds = preds.detach().numpy().reshape(-1, 1)
    y_val = y_val.detach().numpy().reshape(-1, 1)
    # loss = loss_fn(preds, y_val)
    # print(loss)
    torch.save(machine.state_dict(), "elm_model_weights" + str(counter) + ".pth")
    completed = counter * size
    counter += 1
