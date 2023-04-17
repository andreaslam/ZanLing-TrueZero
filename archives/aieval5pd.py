# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import copy
# import tqdm
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# import chess
# from chess import Move
# import torch.nn.init as init
# import sqlite3
# import pandas as pd

# # puzzle presets
# board = chess.Board()


# def get_status(move_turn, result):
#     status = 0
#     if (move_turn == 0 and result == "1") or (move_turn == 1 and result == "-1"):
#         status = 1
#     elif (move_turn == 0 and result == "-1") or (move_turn == 1 and result == "1"):
#         status = -1
#     elif result == "0":
#         status = 0
#     return status


# def board_data(board):
#     board_array = np.zeros((8, 8, 13), dtype=np.int8)
#     for i in range(64):
#         piece = board.piece_at(i)
#         if piece is not None:
#             color = int(piece.color)
#             piece_type = piece.piece_type - 1
#             board_array[i // 8][i % 8][piece_type + 6 * color] = 1
#         else:
#             board_array[i // 8][i % 8][-1] = 1
#     board_array = board_array.flatten()
#     # board_array.shape = 832 (1D)
#     yield board_array


# def load():
#     conn = sqlite3.connect("fracchess_games.db")
#     c = conn.cursor()
#     c.execute("SELECT * FROM games")
#     games = c.fetchall()
#     data = []
#     for g in tqdm.tqdm(games, desc="each game"):
#         game = g[2]
#         MAX_MOMENTS = min(20, len(game))
#         unsampled_idx = [
#             np.random.randint(1, len(game)) for _ in range(MAX_MOMENTS - 1)
#         ]
#         game = game.split(" ")
#         for move in range(MAX_MOMENTS - 1):
#             board = chess.Board()
#             up_to = unsampled_idx[move]
#             moment = game[:up_to]
#             for counter, move in enumerate(moment):
#                 move = Move.from_uci(move)
#                 board.push(move)
#             board_array = np.array(list(board_data(board))).flatten()
#             move_turn = counter % 2
#             data.append(
#                 {
#                     "board_array": board_array,
#                     "move_turn": move_turn,
#                     "status": get_status(move_turn, g[1]),
#                 }
#             )
#     conn.close()
#     df = pd.DataFrame(data)
#     return df


# X = load()
# y = X
# X = X.drop(["status"], axis=1)
# y = y.drop(["board_array", "move_turn"], axis=1)

# print(X)
# print(y)
# print(X.info)
# print(y.info)

# model = nn.Sequential(
#     nn.Linear(833, 1024),
#     nn.BatchNorm1d(1024),
#     nn.ReLU(),
#     nn.Linear(1024, 512),
#     nn.Dropout(p=0.25),
#     nn.BatchNorm1d(512),
#     nn.LeakyReLU(0.1),
#     nn.Linear(512, 256),
#     nn.BatchNorm1d(256),
#     nn.ReLU(),
#     nn.Linear(256, 128),
#     nn.Dropout(p=0.25),
#     nn.BatchNorm1d(128),
#     nn.LeakyReLU(0.01),
#     nn.Linear(128, 64),
#     nn.ReLU(),
#     nn.Linear(64, 1),
#     nn.Dropout(p=0.25),
#     nn.Tanh(),
# )
# # Weight initialization
# for m in model.modules():
#     if isinstance(m, nn.Linear):
#         init.kaiming_uniform_(m.weight, a=0, mode="fan_in")
#         if m.bias is not None:
#             init.constant_(m.bias, 0)

# # scaler = GradScaler()

# # loss function and optimizer
# loss_fn = nn.MSELoss()  # mean square error
# optimizer = optim.AdamW(
#     model.parameters(),
#     lr=1e-3,
#     betas=(0.999, 0.999),
#     eps=1e-08,
#     weight_decay=0,
#     amsgrad=False,
# )
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, factor=0.05, patience=40, verbose=True
# )

# n_epochs = 100  # number of epochs to run optimal = 40, 220
# batch_size = 500  # size of each batch
# batch_start = torch.arange(0, len(X_train), batch_size)

# # Hold the best model
# best_mse = np.inf  # initialise value as infinite
# best_weights = None
# history = []
# accumulation_steps = 4  # accumulate gradients over 4 batches
# for epoch in tqdm.tqdm(range(n_epochs), desc="Epochs"):
#     model.train()
#     epoch_loss = 0.0
#     for i, batch_idx in enumerate(batch_start):
#         batch_X, batch_y = (
#             X_train[batch_idx : batch_idx + batch_size],
#             y_train[batch_idx : batch_idx + batch_size],
#         )
#         optimizer.zero_grad()
#         y_pred = model(batch_X)
#         loss = loss_fn(y_pred, batch_y.view(-1, 1))
#         # scaler.scale(loss).backward() # NEED GPU

#         # accumulate gradients over several batches
#         if (i + 1) % accumulation_steps == 0 or (i + 1) == len(batch_start):
#             # scaler.step(optimizer) # NEED GPU
#             # scaler.update() # NEED GPU
#             model.zero_grad()
#         y_pred = model(batch_X)
#         loss = loss_fn(y_pred, batch_y.view(-1, 1))
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item() * batch_X.shape[0]
#     epoch_loss /= len(X_train)
#     print(epoch_loss)
#     # scheduler.step(epoch_loss)
#     history.append(epoch_loss)
#     if epoch_loss < best_mse:
#         best_mse = epoch_loss
#         best_weights = copy.deepcopy(model.state_dict())

# # load the best weights into the model
# model.load_state_dict(best_weights)

# print("MSE: %.2f" % best_mse)
# print("RMSE: %.2f" % np.sqrt(best_mse))
# plt.plot(history)
# plt.title("Epoch loss for ZL")
# plt.xlabel("Number of Epochs")
# plt.ylabel("Epoch Loss")
# plt.draw()
# plt.savefig("ai-eval-losses.jpg")
# model.eval()
# with torch.no_grad():
#     # Test out inference with 5 samples
#     q = 0
#     for i in range(5):
#         X_sample = X_val[i : i + 1]
#         X_sample = X_sample.clone().detach()
#         y_pred = model(X_sample)
#         print(y_pred)
#         counter = 0
# torch.save(best_weights, "zl.pt")
