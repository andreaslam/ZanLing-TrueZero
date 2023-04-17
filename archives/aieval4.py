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

# puzzle presets
board = chess.Board()
files = ["./ww.pgn", "./bw.pgn", "./drew.pgn"]


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
    return board_array


def process_output(y_pred, is_eval_output=True):
    score = np.mean(np.array(y_pred), axis=1, keepdims=False, dtype=np.float32)
    median = np.median(np.array(y_pred))
    temp = np.array(list(map(int, str(int(median)) * 832)))
    temp = np.split(temp, 832)
    temp = np.array(list(map(float, ["".join(map(str, x)) for x in temp])))
    temp = np.where(temp > 1, 1, temp)
    temp = np.where(temp < -1, -1, temp)
    temp = np.where((-1 <= temp) & (temp <= 1), temp, np.where(temp < -1, -1, 1))
    temp = np.array(list(map(float, temp)))
    temp = np.array(list(map(int, temp)))
    temp = [y for y in temp]
    big_l = [temp]
    if not is_eval_output:
        return big_l
    return score


# load the numpy arrays (if any), these arrays are generated once and saved locally so that time is not wasted reloading every single time
# the files are only there because during development of the code a lot of the time will be lost through loading games afresh
# getting the number of games needed to analyse
# set up training data
train_data = np.empty(
    (0, 832)
)  # 832 is the size of the flattened board in one hot encoding
train_target = []
for file in files:
    with open(file, "r") as f:
        contents = f.read()
    contents = contents.split(" \n")
    for game in tqdm.tqdm(contents, desc="each game"):
        game = game.split(" \n")
        g = "".join(game)
        game = g.split(" ")
        game = [x for x in game if x]  # remove empty strings
        MAX_MOMENTS = min(20, len(game))
        unsampled_idx = [x for x in range(1, len(game))]
        for move in range(MAX_MOMENTS - 1):
            board = chess.Board()
            up_to = np.random.choice(unsampled_idx)
            unsampled_idx.remove(up_to)
            moment = game[:up_to]
            counter = 0
            for move in moment:
                board.push(chess.Move.from_uci(move))
                counter += 1
            matrix_game = np.array([board_data(board)])  # game after one hot encoding
            train_data = np.append(train_data, matrix_game, axis=0)
            status = 0
            move_turn = counter % 2 
            if (move_turn == 0 and file == "./ww.pgn") or (
                move_turn == 1 and file == "./bw.pgn"
            ):
                status = 1
            elif (move_turn == 0 and file == "./bw.pgn") or (
                move_turn == 1 and file == "./ww.pgn"
            ):
                status = -1
            elif file == "./drew.pgn":
                status = 0
            train_target.append(status)
train_target = np.array(train_target, dtype=np.float32).reshape(-1, 1)


# save np arrays (if needed to reuse/redo training)
np.save("train_target.pckl", train_target)
np.save("train_data.pckl", train_data)

# convert all to pytorch tensor


X_train, X_val, y_train, y_val = train_test_split(
    train_data, train_target, test_size=0.1, shuffle=True
)


X_train = torch.tensor(X_train, dtype=torch.float32)  #
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)


# Define the model architecture
model = nn.Sequential(
    nn.Linear(832, 256), 
    nn.ReLU(), 
    nn.Dropout(p=0.1), 
    nn.Linear(256, 128), 
    nn.ReLU(), 
    nn.Dropout(p=0.1), 
    nn.Linear(128, 64), 
    nn.ReLU(), 
    nn.Dropout(p=0.1), 
    nn.Linear(64, 32), 
    nn.ReLU(), 
    nn.Dropout(p=0.1), 
    nn.Linear(32, 1), 
    nn.Tanh()
)
try:
    model.eval()
    weights_path = "./ai_zan1ling4_mini_eval.pt"
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
except FileNotFoundError:
    model.train()

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.999, 0.999),
    eps=1e-08,
    weight_decay=0,
    amsgrad=False,
)

n_epochs = 100  # number of epochs to run optimal = 40, 220
batch_size = 300  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf  # initialise value as infinite
best_weights = None
history = []

for epoch in tqdm.tqdm(range(n_epochs), desc="Epochs"):
    model.train()
    epoch_loss = 0.0
    for batch_idx in batch_start:
        batch_X, batch_y = (
            X_train[batch_idx : batch_idx + batch_size],
            y_train[batch_idx : batch_idx + batch_size],
        )
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = loss_fn(y_pred, batch_y.view(-1, 1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.shape[0]
    epoch_loss /= len(X_train)
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
torch.save(best_weights, "zl.pt")
