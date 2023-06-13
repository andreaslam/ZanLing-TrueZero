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
import aimatchup
import pickle
import gc
import subprocess
import psutil
import torch.cuda
from memory_profiler import profile

# from torch.cuda.amp import GradScaler  # NEED GPU
import subprocess

gc.disable()


# scaler = GradScaler()

TEST_PRECISION = 10 # number of games used for test
RAM_USAGE = 2  # RAM usage in %

class Tanh200(nn.Module):
    def __init__(self):
        super(Tanh200, self).__init__()

    def forward(self, x):
        return torch.tanh(x / 200)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(833, 2048)
        self.dropout1 = nn.Dropout(p=0.35)
        self.relu = nn.LeakyReLU(0.05)
        self.layer2 = nn.Linear(2048, 1)
        self.dropout2 = nn.Dropout(p=0.35)
        self.tanh200 = Tanh200()
        self.hidden_layers = nn.ModuleList()

        # Initialize weights of Linear layers using Xavier initialization
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.layer2.weight)

        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.tanh200(x)

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
        #     with open("progressY.txt", "r+") as f:
        #         contents = f.read()
        #     contents = contents.split(" ")
        #     completed = int(contents[0])
        # except Exception:
        #     completed = 0
        with open("progressY.txt", "w") as f:  # overwrite file contents
            f.write(str(completed) + " " + str(int(total_samples)))
        return int(total_samples)

class Train:
    def __init__(self, X_train, y_train, X_val, y_val, model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = model

    def cycle(self, X_train, y_train, X_val, y_val, model):
        # loss function and optimizer
        loss_fn = nn.MSELoss()  # mean square error
        optimizer = optim.AdamW(model.parameters(), lr=5e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.98, patience=5, verbose=True
        )
        n_epochs = 100
        batch_size = 4096  # size of each batch
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
                y_pred = model(batch_X)
                loss = loss_fn(y_pred, batch_y.view(-1, 1))
                # scaler.scale(loss).backward()  # NEED GPU
                # accumulate gradients over several batches
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(batch_start):
                    # scaler.step(optimizer)  # NEED GPU
                    # scaler.update()  # NEED GPU
                    model.zero_grad()
                # print("gradient_accumulation")

                y_pred = model(batch_X)
                loss = loss_fn(y_pred, batch_y.view(-1, 1))
                # print("loss_backwards")
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.shape[0]
            epoch_loss /= len(X_train)
            # print(epoch_loss)
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
                # print(y_pred)
        torch.save(best_weights, "zlv8.pt")
        # return scheduler.optimizer.param_groups[0][
        #     "lr"
        # ]  # get learning rate of training
        del X_train
        del X_val
        del y_train
        del y_val
        gc.enable()
        gc.collect()
        gc.disable()
        return epoch_loss


@profile
def manager(size, completed):
    subprocess.run(
        ["python3", "aidata.py", str(completed), str(size), "False"], shell=False
    )
    
def mutate(agent, mutation_rate):
        # Calculate the new mutation rate based on the game score
        # Mutate the agent's parameters
        for param in agent.parameters():
            if np.random.rand() < mutation_rate:
                param.data += torch.randn(param.shape) * np.random.rand()
        return agent


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
        return population

if __name__ == "__main__":
    # Define genetic algorithm parameters
    # Training loop
    completed = 0
    counter = 1
    all_completed = False



    try:
        with open("progressY.txt", "r") as f:
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
    POPULATION_SIZE = 5
    population = [Agent() for _ in range(POPULATION_SIZE)]
    count = 0  # used for indexing which agent it is to train now

    while all_completed == False:
        with open("progressY.txt", "r+") as f:
            contents = f.read()
        contents = contents.split(" ")
        completed, size = int(contents[0]), int(contents[1])
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
        for agent in population:
            decider = np.random.rand()
            if decider > 0.5:
                weights_path = "./best_agents0.pt"
            else:
                weights_path = "./best_agents1.pt"
            try:
                state_dict = torch.load(weights_path)
                agent.load_state_dict(state_dict)
                # print("loaded")
            except Exception:
                pass
            population = similarity(population)
        results = {}
        for agent in population:
            # print("SIZE", size)
            # print("COMPLETED", completed)
            manager(size, completed)
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
            t = Train(X_train, y_train, X_val, y_val, agent)
            score = t.cycle(X_train, y_train, X_val, y_val, agent)
            results[count] = 1 - score # make sure that the higher the better
            completed = completed + size
            with open("progressX.txt", "w") as f:  # overwrite file contents
                f.write(str(completed) + " " + str(size))
            del X_train
            del y_train
            del X_val
            del y_val
            del t
            gc.enable()
            gc.collect()
            gc.disable()
            count += 1
        if count == POPULATION_SIZE:  # reached end of list index
            # print(results)
            match_scores = aimatchup.play_game_tournament(population)
            # print(match_scores)
            r = [(x + y)/2 for x, y in zip(list(match_scores.values()), list(results.values()))]
            # print(r)
            indexer = 0
            for item in r:
                results[indexer] = item
                indexer += 1
            count = 0  # reset back to 0 for indexing
            # print(results)
            results = {
                k: v
                for k, v in sorted(
                    results.items(), key=lambda item: item[1], reverse=True
                )  # reverse=True to find the best move with highest score
            }
            # print(results)
            p1 = list(results.keys())[0]
            p2 = list(results.keys())[1]
            parent1 = population[p1]
            parent2 = population[p2]
            new_population = []
            new_population.append(parent1)
            new_population.append(parent2)
            torch.save(
                parent1.state_dict(),
                "./zlparent1.pt",
            )
            torch.save(
                parent2.state_dict(),
                "./zlparent2.pt",
            )
            for _ in range(POPULATION_SIZE - 2):  # exclude parents, already included
                child = Agent()
                for name, param in child.named_parameters():
                    if np.random.rand() > 0.5:
                        param.data.copy_(parent1.state_dict()[name].data)
                    else:
                        param.data.copy_(parent2.state_dict()[name].data)
                child = mutate(child, np.random.rand())
                new_population.append(child)
            population = new_population
            del new_population
        with open("progressY.txt", "w") as f:  # overwrite file contents
            f.write(str(completed) + " " + str(size))
        completed = counter * size
        counter += 1
