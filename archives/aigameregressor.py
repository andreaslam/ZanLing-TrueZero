# RANDOM FOREST NETWORK THAT IS USED TO CLASSIFY GAMES
# zan1ling4 真零
import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import graphviz
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import chess
import joblib

# puzzle presets
MAX_MOMENTS = 20
TOTAL_GAMES = 4
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
        if file == "./won.txt":  # NOTE: change the filename
            d = {"moves": game, "status": "won"}
            df_add.loc[len(df_add)] = d
        elif file == "./lost.txt":
            d = {"moves": game, "status": "lost"}
            df_add.loc[len(df_add)] = d
        elif file == "./drew.txt":
            d = {"moves": game, "status": "drew"}
            df_add.loc[len(df_add)] = d
    final_df = pd.concat([final_df, df_add], ignore_index=True)
print(final_df)
# define function that takes chess board and turns into AI-understandable matrix


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


game = 0
train_df = pd.DataFrame(
    columns=[
        "board_data",
        "status",
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
        for move in moment:
            board.push(chess.Move.from_uci(move))
        ai_moves, piece_stat = board_data(board)
        ai_moves = list(ai_moves)
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
        # Print the rescaled value
        final_data = rescaled
        counter = 0
        d = {
            "board_data": final_data,
            "status": status,
            "num_of_moves": float(len(moment)),
            "p": piece_stat["p"],
            "k": piece_stat["k"],
            "q": piece_stat["q"],
            "r": piece_stat["r"],
            "n": piece_stat["n"],
            "b": piece_stat["b"],
            "empty": piece_stat["empty"],
        }
        df_add.loc[len(df_add)] = d
        train_df = pd.concat([train_df, df_add], ignore_index=True)
    game += 1
print(train_df)

# preprocessing data

train_df["status"] = train_df["status"].map(
    {"won": 1.0, "lost": -1.0, "drew": 0.0}
)  # target
X = train_df.drop("status", axis=1)
y = train_df["status"]
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create/load random forest

try:
    rf = joblib.load("forest.joblib")
    print("Forest loaded")
except FileNotFoundError:
    rf = RandomForestRegressor(n_estimators=1000)


rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("PREDICTIONS", y_pred)

# Export the first three decision trees from the forest

for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(
        tree,
        feature_names=X_train.columns,
        filled=True,
        max_depth=2,
        impurity=False,
        proportion=True,
    )
    graph = graphviz.Source(dot_data)
with open("trees.dot", "w+") as f:
    f.write(str(graph))
param_dist = {"n_estimators": randint(50, 500), "max_depth": randint(1, 20)}

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print("Best hyperparameters:", rand_search.best_params_)
# Generate predictions with the best model
y_pred = best_rf.predict(X_test)

# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(
    best_rf.feature_importances_, index=X_train.columns
).sort_values(ascending=False)

joblib.dump(rf, "forest.joblib")

# Plot a simple bar chart
ax = feature_importances.plot.bar()
ax.figure.savefig("results.png")
