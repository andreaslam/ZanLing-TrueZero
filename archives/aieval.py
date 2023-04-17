# 真零 eval engine with NN
# zan1ling4 | 真零 | (pronounced Jun Ling)
# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import tqdm
from sklearn.model_selection import train_test_split
import chess

# puzzle presets
MAX_MOMENTS = 10
board = chess.Board()
files = ["./won.txt", "./lost.txt", "./drew.txt"]

# pre-training loop

final_df = pd.DataFrame()

# set up training data

for file in files:
    with open(file, "r") as f:
        contents = f.read()

    contents = contents.split(" \n")
    df_add = pd.DataFrame(columns=["moves", "status"])

    for game in contents:
        if file == "./won.txt" or file == "./a.txt":  # NOTE: change the filename
            d = {"moves": game, "status": "won"}
            df_add.loc[len(df_add)] = d
        elif file == "./lost.txt" or file == "./ab.txt":
            d = {"moves": game, "status": "lost"}
            df_add.loc[len(df_add)] = d
        elif file == "./drew.txt" or file == ".ac.txt":
            d = {"moves": game, "status": "drew"}
            df_add.loc[len(df_add)] = d
    final_df = pd.concat([final_df, df_add], ignore_index=True)

print("ALL GAMES LOADED")

# define function that takes chess board and turns into AI-understandable matrix


def process_output(y_pred, is_eval_output=True):
    total_sum = 0
    big_l = []
    c = 0
    small_l = []
    for x in y_pred:
        for y in x:
            total_sum += y
            small_l.append(y)
            c += 1
        median = 0
        if len(small_l) % 2 == 0:
            median = small_l[int(len(small_l) / 2)]
        else:
            median = small_l[int((len(small_l) + 1) / 2)]
        avg = total_sum / c
        score = (float(median) + float(avg)) / 2
        temp = ""
        big_l = []
        temp = (str(int(median)) + "2") * 832
        temp = temp.split("2")
        tt = []
        temp = list(filter(None, temp))
        for y in temp:
            tt.append(y)
        temp = [float(y) for y in tt]
        for y in temp:
            if y > 1:
                y = min(y, 1)
            elif y < -1:
                y = max(y, -1)
        temp = [y if -1 <= y <= 1 else max(-1, min(y, 1)) for y in temp]
        temp = [float(int(y)) for y in temp]
        big_l.append(temp)
    if is_eval_output == False:
        return big_l
    return score


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


game = 0
train_df = pd.DataFrame(
    columns=[
        "board_data",
        "status",
    ]
)

for index, row in final_df.iterrows():
    moves = row["moves"].split(" ")
    status = row["status"]
    moves = [x for x in moves if x]  # removes empty strings in list
    if len(moves) <= MAX_MOMENTS:
        MAX_MOMENTS = len(moves)
    unsampled_idx = [x for x in range(len(moves))]
    unsampled_idx.pop(0)
    for _ in range(MAX_MOMENTS - 1):
        board = chess.Board()
        up_to = np.random.choice(unsampled_idx)
        unsampled_idx.remove(up_to)
        moment = moves[:up_to]
        df_add = pd.DataFrame(
            columns=[
                "board_data",
                "status",
            ]
        )
        for move in moment:
            board.push(chess.Move.from_uci(move))
        ai_moves = board_data(board)

        counter = 0
        d = {
            "board_data": ai_moves,
            "status": status,
        }
        df_add.loc[len(df_add)] = d
        train_df = pd.concat([train_df, df_add], ignore_index=True)
    game += 1

# preprocessing data

train_df["status"] = train_df["status"].map(
    {"won": 1.0, "lost": -1.0, "drew": 0.0}
)  # target
X = np.array([x for x in train_df["board_data"]])
temp = ""
big_l = []
for x in train_df["status"]:
    temp = (str(x) + "2") * 832
    temp = temp.split("2")
    tt = []
    for y in temp:
        if y != "":
            tt.append(y)
    temp = [float(y) for y in tt]
    big_l.append(temp)

train_df["status"] = big_l

print("TRAIN", train_df)

y = np.array([x for x in train_df["status"]])
# train-test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
NON_LAST_LAYERS = 2
model = nn.Sequential(
    nn.Linear(832, 832),
)
for i in range(NON_LAST_LAYERS):
    model.add_module(f"linear_{i}", nn.ReLU())
    model.add_module(f"linear_{i}", nn.Linear(832, 832))

model.add_module(f"linear_{NON_LAST_LAYERS}", nn.Tanh())
try:
    model.eval()
    weights_path = "./ai_zan1ling4_mini_eval.pt"
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
except FileNotFoundError:
    model.train()

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

n_epochs = 40  # number of epochs to run optimal = 40, 220
batch_size = 512  # size of each batch
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
        loss = loss_fn(y_pred, batch_y)
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
        X_sample = X_test[i : i + 1]
        X_sample = X_sample.clone().detach()
        print(X_sample.shape)
        y_pred = model(X_sample)
        print("Y PRED", y_pred)
        print(y_pred.shape)
        y_pred = process_output(y_pred, False)
        y_pred = torch.tensor(big_l)
        counter = 0
torch.save(best_weights, "ai_zan1ling4_eval.pt")
