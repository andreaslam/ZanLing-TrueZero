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
size = 10000


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

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        return X_train, y_train, X_val, y_val


class Tanh200(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 200)


class Train:
    def __init__(self, X_train, y_train, X_val, y_val, model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = model

    def cycle(self, X_train, y_train, X_val, y_val, model):
        # Weight initialization

        # scaler = GradScaler()

        # loss function and optimizer
        loss_fn = nn.MSELoss()  # mean square error
        optimizer = optim.AdamW(model.parameters(), lr=7.5e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.75, patience=3, verbose=True
        )
        n_epochs = 100
        batch_size = 2048  # size of each batch
        batch_start = torch.arange(0, len(X_train), batch_size)
        # Hold the best model
        best_mse = np.inf  # initialise value as infinite
        best_weights = None
        history = []
        accumulation_steps = 2  # accumulate gradients over 2 batches
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
        del X_train
        del X_val
        del y_train
        del y_val
        return history[-1]


# Training loop
completed = 0
counter = 1
all_completed = False

# define important functions


# define function for similarity
def similarity(population):
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            agent1 = population[i]
            agent2 = population[j]
            SIMILARITY_THRESHOLD = 0.85
            cosine_similarity = nn.CosineSimilarity(dim=0)
            euclidean_distance = nn.PairwiseDistance(p=2, keepdim=True)
            weights1 = [param.data.flatten() for param in agent1.parameters()]
            weights2 = [param.data.flatten() for param in agent2.parameters()]
            for w1, w2 in zip(weights1, weights2):
                distance = euclidean_distance(
                    w1.clone().detach(), w2.clone().detach()
                ).item()
                similarity = cosine_similarity(w1, w2).item()
                if similarity > SIMILARITY_THRESHOLD or distance > SIMILARITY_THRESHOLD:
                    # Mutate one of the agents
                    if np.random.rand() < 0.5:
                        agent1 = mutate(agent1, np.random.rand())
                    else:
                        agent2 = mutate(agent2, np.random.rand())


# define mutation
def mutate(agent, mutation_rate):
    # Calculate the new mutation rate based on the game score
    # Mutate the agent's parameters
    for param in agent.parameters():
        if np.random.rand() < mutation_rate:
            param.data += torch.randn(param.shape) * np.random.rand()
    return agent


# progress checking logic


try:
    with open("progress.txt", "r") as f:
        contents = f.read()
except FileNotFoundError:
    with open("progress.txt", "w+") as f:  # create the file if it does not exist
        f.write(
            "0 " + str(size)
        )  # 0 means 0 games processed; starting from scratch, size is number of games to process in one cycle

while all_completed == False:
    with open("progress.txt", "r+") as f:
        contents = f.read()
    NUM_AGENTS = 5
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
    model = nn.Sequential(
        nn.Linear(833, 512),
        nn.Dropout(p=0.25),
        nn.ReLU(),
        nn.Linear(512, 1),
        nn.Dropout(p=0.25),
        Tanh200(),
    )
    population = [model for _ in range(0, NUM_AGENTS)]
    try:
        weights_paths = ["./zlparent1.pt", "./zlparent2.pt"]
        idx = np.random.randint(0, 2)
        # NOTE: weights_paths and weights_path are NOT the same!
        weights_path = weights_paths[idx]
        state_dict = torch.load(weights_path)
        for model in population:
            model.load_state_dict(state_dict)
    except FileNotFoundError:
        for mod in population:
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
    # make sure every agent is unique
    similarity(population)
    results = {}
    # training the population
    c = 0
    for agent in population:
        t = Train(X_train, y_train, X_val, y_val, agent)
        score = t.cycle(X_train, y_train, X_val, y_val, agent)
        results[str(c)] = score
        c += 1
    # save the best Agents and breed them
    results = {
        k: v
        for k, v in sorted(results.items(), key=lambda item: item[1], reverse=False)
    }  # reverse=False to find agents with the lowest MSE score
    p1 = list(results.values())[0]
    p2 = list(results.values())[1]
    # indexing the find the parent among the population
    parent1 = population[int(p1)]
    parent2 = population[int(p2)]
    # save parents' weights
    torch.save(parent1.state_dict(), "./zlparent1.pt")
    torch.save(parent2.state_dict(), "./zlparent2.pt")
    num_of_children = len(population) - 2
    new_population = []
    new_population.append(parent1)
    new_population.append(parent2)
    for _ in range(num_of_children):
        child = model
        for name, param in child.named_parameters():
            if np.random.rand() > 0.5:
                param.data.copy_(parent1.state_dict()[name].data)
            else:
                param.data.copy_(parent2.state_dict()[name].data)
        # Perform mutation on the child
        child = mutate(child, np.random.rand())
        new_population.append(child)
    population = new_population
    completed = completed + size
    with open("progress.txt", "w") as f:  # overwrite file contents
        f.write(str(completed) + " " + str(size))
    completed = counter * size
    del c
    del d
    del new_population
    del X_train
    del y_train
    del X_val
    del y_val
    del population
    counter += 1
