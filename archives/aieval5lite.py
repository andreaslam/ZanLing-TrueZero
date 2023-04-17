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


def get_status(move_turn, result):
    status = 0
    if (move_turn == 0 and result == "1") or (move_turn == 1 and result == "-1"):
        status = 1
    elif (move_turn == 0 and result == "-1") or (move_turn == 1 and result == "1"):
        status = -1
    elif result == "0":
        status = 0
    return status


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
    # board_array.shape = 832 (1D)
    yield board_array


def load():
    conn = sqlite3.connect("fracchess_games.db")
    c = conn.cursor()
    # select all rows from the table
    c.execute("SELECT * FROM games")
    # replace reading from file with reading from SQLite
    games = c.fetchall()

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
                [board_data(board).__next__()]
            )  # game after one hot encoding
            # Add move_turn as a feature to matrix_game
            move_turn = counter % 2
            matrix_game = np.concatenate((matrix_game, np.array([[move_turn]])), axis=1)
            yield matrix_game.flatten(), get_status(move_turn, g[1])

    conn.close()


train_data, train_target = zip(*load())


def loading(train_data, train_target):
    train_data, train_target = np.array(train_data), np.array(train_target)

    X_train, X_val, y_train, y_val = train_test_split(
        train_data, train_target, test_size=0.1, shuffle=True
    )

    X_train = torch.tensor(X_train, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64)
    X_val = torch.tensor(X_val, dtype=torch.float64)
    y_val = torch.tensor(y_val, dtype=torch.float64)
    return X_train, y_train, X_val, y_val


X_train, y_train, X_val, y_val = loading(train_data, train_target)
class Tanh200(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 200)


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
    Tanh200(),
)
# Weight initialization
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

# scaler = GradScaler()

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.AdamW(
    model.parameters(),
    lr=1.e-3,  # 1e-3
    betas=(0.99, 0.999),
    eps=1e-08,
    weight_decay=0,
    amsgrad=False,
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.075, patience=10, verbose=True
)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=500, verbose=True
)

n_epochs = 100  # number of epochs to run optimal = 40, 22040, 220
batch_size = 500  # size of each batch
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
    scheduler2.step()
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
    q = 0
    for i in range(5):
        X_sample = X_val[i : i + 1]
        X_sample = X_sample.clone().detach()
        y_pred = model(X_sample)
        print(y_pred)
        counter = 0
torch.save(best_weights, "zlv6.pt")
