# zan1ling4 | 真零 | (pronounced Jun Ling)
# imports
import numpy as np
import chess
import torch
from torch import nn
from torch import optim
import sqlite3
import matplotlib.pyplot as plt
import tqdm
import torch.nn.init as init
import copy
import pickle
import gc
import subprocess
import psutil
import torch.cuda
from torch.cuda.amp import GradScaler  # NEED GPU
import subprocess
gc.disable()

scaler = GradScaler()

TEST_PRECISION = 100000  # use 100 games in the test
RAM_USAGE = 50  # RAM usage in %


class Tanh200(nn.Module):
    def __init__(self):
        super(Tanh200, self).__init__()

    def forward(self, x):
        return torch.tanh(x / 200).to("cuda")


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(833, 2048).to("cuda")
        self.dropout1 = nn.Dropout(p=0.45).to("cuda")
        self.relu = nn.LeakyReLU(0.05).to("cuda")
        self.layer2 = nn.Linear(2048, 1).to("cuda")
        self.dropout2 = nn.Dropout(p=0.45).to("cuda")
        self.tanh200 = Tanh200().to("cuda")
        self.hidden_layers = nn.ModuleList().to("cuda")

        # Initialize weights of Linear layers using Xavier initialization
        init.xavier_uniform_(self.fc1.weight).to("cuda")
        init.xavier_uniform_(self.layer2.weight).to("cuda")

        self.loss = nn.MSELoss().to("cuda")

    def forward(self, x):
        x = self.fc1(x).to("cuda")
        x = self.dropout1(x).to("cuda")
        x = self.relu(x).to("cuda")
        x = self.layer2(x).to("cuda")
        x = self.dropout2(x).to("cuda")
        x = self.tanh200(x).to("cuda")
        return x


class MemoryEstimator:
    def __init__(self, threshold_percent):
        self.threshold_percent = threshold_percent

    def estimate_memory(self):  # used for getting total RAM in percent
        return psutil.virtual_memory().available * 100 / psutil.virtual_memory().total

    def estimate_count(
        self, threshold_percent
    ):  # used for estimating how many games to analyse before reaching threshold
        before = psutil.virtual_memory().available / psutil.virtual_memory().total
        print("BEFORE", before)
        after = subprocess.run(
            ["python3", "aidata.py", str(0), str(TEST_PRECISION), "True"],
            capture_output=True,
        )  # do test run with 10 games
        after = float(str(after.stdout.decode("utf-8").strip()))
        print("AFTER", after)
        # find memory reduction
        memory_reduction = (
            before - after
        ) / TEST_PRECISION  # memory in percent that each test game contributed
        # find the number of games based on threshold_percent
        total_samples = abs((threshold_percent / 100) / memory_reduction)
        print(total_samples)
        # try:
        #     with open("progressX.txt", "r+") as f:
        #         contents = f.read()
        #     contents = contents.split(" ")
        #     completed = int(contents[0])
        # except Exception:
        #     completed = 0
        with open("progressX.txt", "w") as f:  # overwrite file contents
            f.write(str(completed) + " " + str(int(total_samples)))
        return int(total_samples)


class Train(Tanh200):
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def cycle(self, X_train, y_train, X_val, y_val, best_score):
        model = Agent().to("cuda")
        X_train, y_train, X_val, y_val = X_train.to("cuda"), y_train.to("cuda"), X_val.to("cuda"), y_val.to("cuda")
        # Weight initialization
        try:
            weights_path = "./zlv7.pt"
            state_dict = torch.load(weights_path, map_location="cuda")
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
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.95, patience=50, verbose=True
        )
        n_epochs = 500
        batch_size = 2048  # size of each batch
        batch_start = torch.arange(0, len(X_train), batch_size)
        # Hold the best model
        best_mse = np.inf  # initialise value as infinite
        best_weights = None
        history = []
        accumulation_steps = 2  # accumulate gradients over 2 batches
        for _ in tqdm.tqdm(range(n_epochs), desc="Epochs"):
            model.train()
            epoch_loss = 0.0
            for i, batch_idx in enumerate(batch_start):
                batch_X, batch_y = (
                    X_train[batch_idx : batch_idx + batch_size],
                    y_train[batch_idx : batch_idx + batch_size],
                )
                optimizer.zero_grad()
                y_pred = model.forward(batch_X).to("cuda")
                loss = loss_fn(y_pred, batch_y.view(-1, 1)).to("cuda")
                scaler.scale(loss).backward() # NEED GPU

                # accumulate gradients over several batches
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(batch_start):
                    scaler.step(optimizer)  # NEED GPU
                    scaler.update()  # NEED GPU
                model.zero_grad()
                y_pred = model(batch_X).to("cuda")
                loss = loss_fn(y_pred, batch_y.view(-1, 1)).to("cuda")
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.shape[0]
            epoch_loss /= len(X_train)
            scheduler.step(epoch_loss)
            history.append(epoch_loss)
            if epoch_loss < best_mse:
                best_mse = epoch_loss
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights, "zlv7_t.pt")
            elif epoch_loss >= best_mse:
                weights_path = "./zlv7_t.pt"
                state_dict = torch.load(weights_path, map_location="cuda")
                model.load_state_dict(state_dict)

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
                X_sample = X_sample.clone().detach().to("cuda")
                y_pred = model.forward(X_sample).to("cuda")
        torch.save(best_weights, "zlv7.pt")
        if best_score > epoch_loss:
            torch.save(best_weights, "zlv7.pt")
        # return scheduler.optimizer.param_groups[0][
        #     "lr"
        # ]  # get learning rate of training
        torch.cuda.empty_cache()
        del X_train
        del X_val
        del y_train
        del y_val
        gc.enable()
        gc.collect()
        gc.disable()
        return epoch_loss



def manager(size,completed):
    subprocess.run(
        ["python3", "aidata.py", str(completed), str(size), "False"], shell=False
    )


if __name__ == "__main__":
    # Define genetic algorithm parameters
    # Training loop
    completed = 0
    counter = 1
    all_completed = False
    try:
        with open("progressX.txt", "r") as f:
            contents = f.read()
    except FileNotFoundError:
        m = MemoryEstimator(RAM_USAGE)
        size = m.estimate_count(
            RAM_USAGE
        )  # memory based allocation with arg as percentage usage of RAM per cycle

    # puzzle presets
    board = chess.Board()
    completed = 0
    # find number of lines in a database

    DB_LOCATION = "chess_games.db"

    # Connect to the database
    conn = sqlite3.connect(DB_LOCATION)

    # Create a cursor object
    cursor = conn.cursor()

    # # Execute the query to get the length of the table
    cursor.execute("SELECT COUNT(*) FROM games")

    # # Fetch the result
    result = cursor.fetchone()[0]
    # print(result)
    conn.close()
    # instantiate population
    count = 0  # used for indexing which agent it is to train now
    best_score = np.inf
    while all_completed == False:
        with open("progressX.txt", "r+") as f:
            contents = f.read()
        contents = contents.split(" ")
        completed, size = int(contents[0]), int(contents[1])
        if completed != 0:
            m = MemoryEstimator(RAM_USAGE)
            estimate = m.estimate_count(RAM_USAGE)
            size = estimate
        not_done = result - completed
        if not_done == 0:
            all_completed = True
            break
        if (
            not_done < size
        ):  # size is the number of chess games processed/cycle in total
            size = not_done  # this is for the final cycle if there are any remainders
            all_completed = True
        # repeat the training process for all agents in population
        # load weights onto AI
        manager(size,completed)
        ###############################################################################################
        files = ["X_train", "y_train", "X_val", "y_val"]
        data = []
        for file in files:
            storagefile = open(file, "rb")
            data.append(pickle.load(storagefile))
        X_train, y_train, X_val, y_val = (
            data[0],
            data[1],
            data[2],
            data[3],
        )
        gc.enable()
        gc.collect()
        gc.disable()
        print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
        #######################################################################################
        # print("ready")
        t = Train(X_train, y_train, X_val, y_val)
        score = t.cycle(X_train, y_train, X_val, y_val, best_score)
        best_score = min(best_score, score)
        completed = completed + size
        with open("progressX.txt", "w") as f:  # overwrite file contents
            f.write(str(completed) + " " + str(size))
        completed = counter * size
        del t
        del X_train
        del y_train
        del X_val
        del y_val
        gc.enable()
        gc.collect()
        gc.disable()
        counter += 1
