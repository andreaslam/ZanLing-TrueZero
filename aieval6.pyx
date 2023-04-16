# zan1ling4 | 真零 | (pronounced Jun Ling)
# imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import chess

# from torch.cuda.amp import autocast, GradScaler # NEED GPU
from chess import Move
import torch.nn.init as init

# from memory_profiler import profile
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
# print(result)
conn.close()
size = 50000


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

    def loading(self, train_data, train_target):
        train_data, train_target = np.array(train_data), np.array(train_target)

        X_train, X_val, y_train, y_val = train_test_split(
            train_data, train_target, test_size=0.2, shuffle=True
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        return X_train, y_train, X_val, y_val


class Tanh200(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 200)


class Train(Tanh200):
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def cycle(self, X_train, y_train, X_val, y_val):
        model = nn.Sequential(
            nn.Linear(833, 512),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Dropout(p=0.25),
            Tanh200(),
        )

        # Weight initialization
        try:
            weights_path = "./zlv6.pt"
            state_dict = torch.load(weights_path)
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

        # scaler = GradScaler()

        # loss function and optimizer
        loss_fn = nn.MSELoss()  # mean square error
        optimizer = optim.AdamW(model.parameters(), lr=1.25e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.9999999, patience=2, verbose=True
        )
        n_epochs = 200
        batch_size = 256  # size of each batch
        batch_start = torch.arange(0, len(X_train), batch_size)
        # Hold the best model
        best_mse = np.inf  # initialise value as infinite
        best_weights = None
        history = []
        accumulation_steps = 4  # accumulate gradients over 4 batches
        for epoch in tqdm.tqdm(range(n_epochs), desc="Epochs"):
            model.train()
            epoch_loss = 0.0
            for i, batch_idx in enumerate(batch_start):
                batch_X, batch_y = (
                    X_train[batch_idx : batch_idx + batch_size],
                    y_train[batch_idx : batch_idx + batch_size],
                )
                optimizer.zero_grad()
                y_pred = model(batch_X)
                loss = loss_fn(y_pred, batch_y.view(-1, 1))
                # scaler.scale(loss).backward() # NEED GPU

                # accumulate gradients over several batches
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(batch_start):
                    # scaler.step(optimizer) # NEED GPU
                    # scaler.update() # NEED GPU
                    model.zero_grad()
                y_pred = model(batch_X)
                loss = loss_fn(y_pred, batch_y.view(-1, 1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.shape[0]
            epoch_loss /= len(X_train)
            print(epoch_loss)
            scheduler.step(epoch_loss)
            history.append(epoch_loss)
            if epoch_loss < best_mse:
                best_mse = epoch_loss
                best_weights = copy.deepcopy(model.state_dict())

        # load the best weights into the model
        model.load_state_dict(best_weights)

        print("MSE: %.2f" % best_mse)
        print("RMSE: %.2f" % np.sqrt(best_mse))
        plt.plot(history)
        plt.title("Epoch loss for ZL")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Epoch Loss")
        plt.draw()
        plt.savefig("ai-eval-losses.jpg")
        model.eval()
        with torch.no_grad():
            # Test out inference with 5 samples
            for i in range(5):
                X_sample = X_val[i : i + 1]
                X_sample = X_sample.clone().detach()
                y_pred = model(X_sample)
                print(y_pred)
        torch.save(best_weights, "zlv6.pt")
        # return scheduler.optimizer.param_groups[0][
        #     "lr"
        # ]  # get learning rate of training


# Training loop
completed = 0
counter = 1
all_completed = False
while all_completed == False:
    with open("progress.txt", "r+") as f:
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
    train_data, train_target = None, None # served for clearing variable in loops 
    train_data, train_target = zip(*d.load(completed, size))
    X_train, y_train, X_val, y_val = d.loading(train_data, train_target)
    t = Train(X_train, y_train, X_val, y_val)
    t.cycle(X_train, y_train, X_val, y_val)
    completed = counter * size
    with open("progress.txt", "w") as f:  # overwrite file contents
        f.write(str(completed) + " " + str(size))
    counter += 1
