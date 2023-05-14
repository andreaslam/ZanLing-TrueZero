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
from chess import Move
import torch.nn.init as init
from sklearn.model_selection import train_test_split
import copy
from sklearn.cluster import KMeans
import multiprocessing


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
    def __init__(self):
        super(Tanh200, self).__init__()

    def forward(self, x):
        return torch.tanh(x / 200)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(833, 2048)
        self.dropout1 = nn.Dropout(p=0.25)
        self.relu = nn.LeakyReLU(0.05)
        self.layer2 = nn.Linear(2048, 1)
        self.dropout2 = nn.Dropout(p=0.25)
        self.tanh200 = Tanh200()
        self.hidden_layers = nn.ModuleList()

        # Initialize weights of Linear layers using Xavier initialization
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.layer2.weight)

        self.loss = nn.MSELoss()

    def forward(self, x):
        x
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.tanh200(x)
        return x


class Train:
    def __init__(self, X_train, y_train, X_val, y_val, model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = model

    def cycle(self, X_train, y_train, X_val, y_val, model):
        X_train = X_train
        y_train = y_train
        X_val = X_val
        y_val = y_val
        # loss function and optimizer
        loss_fn = nn.MSELoss()  # mean square error
        optimizer = optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.75, patience=5, verbose=False
        )
        n_epochs = 125
        batch_size = 512  # size of each batch
        batch_start = torch.arange(0, len(X_train), batch_size)
        # Hold the best model
        best_mse = np.inf  # initialise value as infinite
        best_weights = None
        history = []
        # accumulation_steps = 2  # accumulate gradients over 2 batches
        for _ in tqdm.tqdm(range(n_epochs), desc="Epochs"):
            epoch_loss = 0.0
            for i, batch_idx in enumerate(batch_start):
                batch_X, batch_y = (
                    X_train[batch_idx : batch_idx + batch_size],
                    y_train[batch_idx : batch_idx + batch_size],
                )
                optimizer.zero_grad()
                y_pred = model.forward(batch_X)
                loss = loss_fn(y_pred, batch_y.view(-1, 1))
                # scaler.scale(loss).backward() # NEED GPU

                # accumulate gradients over several batches
                # if (i + 1) % accumulation_steps == 0 or (i + 1) == len(batch_start):
                #     # scaler.step(optimizer) # NEED GPU
                #     # scaler.update() # NEED GPU
                model.zero_grad()
                y_pred = model(batch_X)
                loss = loss_fn(y_pred, batch_y.view(-1, 1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.shape[0]
            epoch_loss /= len(X_train)
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
        with torch.no_grad():
            # Test out inference with 5 samples
            for i in range(5):
                X_sample = X_val[i : i + 1]
                X_sample = X_sample.clone().detach()
                y_pred = model.forward(X_sample)
        torch.save(best_weights, "zlv7.pt")
        # return scheduler.optimizer.param_groups[0][
        #     "lr"
        # ]  # get learning rate of training
        del X_train
        del X_val
        del y_train
        del y_val
        return history[-1]


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


# Define genetic algorithm parameters
# Training loop
completed = 0
counter = 1
all_completed = False


def mutate(agent, mutation_rate):
    # Calculate the new mutation rate based on the game score
    # Mutate the agent's parameters
    for param in agent.parameters():
        if np.random.rand() < mutation_rate:
            x = torch.randn(param.shape)
            y = torch.tensor(np.random.rand())
            param.data += x * y
    return agent


try:
    with open("progressX.txt", "r") as f:
        contents = f.read()
except FileNotFoundError:
    with open("progressX.txt", "w+") as f:  # create the file if it does not exist
        f.write(
            "0 " + str(size)
        )  # 0 means 0 games processed; starting from scratch, size is number of games to process in one cycle


def similarity(population):
    SIMILARITY_THRESHOLD = 0.85
    cosine_similarity = nn.CosineSimilarity(dim=0)
    euclidean_distance = nn.PairwiseDistance(p=2, keepdim=True)
    # Flatten the weights of all agents
    weights = [
        np.concatenate(
            [param.data.flatten().to("cpu").numpy() for param in agent.parameters()]
        )
        for agent in population
    ]
    # Use KMeans clustering to group agents by similarity
    kmeans = KMeans(n_clusters=len(population), n_init="auto").fit(weights)
    for cluster in np.unique(kmeans.labels_):
        agents = [population[i] for i in np.where(kmeans.labels_ == cluster)[0]]
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1 = agents[i]
                agent2 = agents[j]
                weights1 = [param.data.flatten() for param in agent1.parameters()]
                weights2 = [param.data.flatten() for param in agent2.parameters()]
                for w1, w2 in zip(weights1, weights2):
                    distance = euclidean_distance(
                        w1.clone().detach(), w2.clone().detach()
                    ).item()
                    similarity = cosine_similarity(w1, w2).item()
                    if (
                        similarity > SIMILARITY_THRESHOLD
                        or distance > SIMILARITY_THRESHOLD
                    ):
                        # Mutate one of the agents
                        if np.random.rand() < 0.5:
                            agent1 = mutate(agent1, np.random.rand())
                        else:
                            agent2 = mutate(agent2, np.random.rand())
    return population


def manager(cpu):
    print("GOT HERE")
    size = cpu[1] - cpu[0]
    print(size, cpu[1])
    d = DataManager(size, cpu[1])
    train_data, train_target = zip(*d.load(completed, size))
    X_train, y_train, X_val, y_val = d.loading(train_data, train_target)
    del d
    del train_data
    del train_target
    return X_train, y_train, X_val, y_val


def split_tasks(cpu_count, size):
    non_last_cores = size // cpu_count
    li = []
    start_counter = 0
    stop_counter = non_last_cores
    for _ in range(cpu_count):
        li.append([start_counter, stop_counter])
        start_counter += non_last_cores
        stop_counter += non_last_cores
    ending = li[-1][-1]
    if ending < size:
        li[-1][-1] = ending + (size - ending)
    del non_last_cores
    del start_counter
    del stop_counter

    return li


if __name__ == "__main__":
    # puzzle presets
    board = chess.Board()
    completed = 0
    # find number of lines in a database

    DB_LOCATION = "fracchess_games.db"

    # Connect to the database
    conn = sqlite3.connect(DB_LOCATION)

    # Create a cursor object
    cursor = conn.cursor()

    # # Execute the query to get the length of the table
    cursor.execute("SELECT COUNT(*) FROM games")

    # # Fetch the result
    result = cursor.fetchone()[0]
    print("done!")
    # print(result)
    conn.close()
    size = 5
    # instantiate population
    POPULATION_SIZE = 50
    population = [Agent() for _ in range(POPULATION_SIZE)]
    num_elites = int(POPULATION_SIZE * 0.4)
    count = 0  # used for indexing which agent it is to train now
    print("done")
    while all_completed == False:
        with open("progressX.txt", "r+") as f:
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
        random_integers = np.random.randint(0, num_elites, size=POPULATION_SIZE)
        index = 0
        for agent in population:
            weights_path = "./best_agents" + str(random_integers[index]) + ".pt"
            try:
                state_dict = torch.load(weights_path)
                agent.load_state_dict(state_dict)

            except Exception:
                pass
            index += 1
        population = similarity(population)
        results = {}
        p = multiprocessing.cpu_count()
        load = split_tasks(p, size)
        # print("SIZE", size)
        # print("COMPLETED", completed)
        print("HERE WE GO AGAIN")
        u = 0
        for agent in population:
            with multiprocessing.Pool(p) as pool:
                print("HEY")
                r = pool.map(manager, load)
            X_train, y_train, X_val, y_val = zip(*r)
            xt = X_train[0]  # initialize the result with the first tensor
            for i in range(1, len(X_train)):
                xt = torch.cat(
                    (xt, X_train[i]), dim=0
                )  # concatenate each tensor to the result tensor along the first dimension
            yt = y_train[0]  # initialize the result with the first tensor
            for i in range(1, len(y_train)):
                yt = torch.cat(
                    (yt, y_train[i]), dim=0
                )  # concatenate each tensor to the result tensor along the first dimension
            xv = X_val[0]  # initialize the result with the first tensor
            for i in range(1, len(X_val)):
                xv = torch.cat(
                    (xv, X_val[i]), dim=0
                )  # concatenate each tensor to the result tensor along the first dimension
            yv = y_val[0]  # initialize the result with the first tensor
            for i in range(1, len(y_val)):
                yv = torch.cat(
                    (yv, y_val[i]), dim=0
                )  # concatenate each tensor to the result tensor along the first dimension
            X_train, y_train, X_val, y_val = xt, yt, xv, yv
            
            t = Train(X_train, y_train, X_val, y_val, agent)
            score = t.cycle(X_train, y_train, X_val, y_val, agent)
            results[count] = score  # make sure that the lower the better
            completed = completed + size
            count += 1
            with open("progress.txt", "w") as f:  # overwrite file contents
                f.write(str(completed) + " " + str(size))
            completed = counter * size
            counter += 1
            u += 1
        if count == POPULATION_SIZE:  # reached end of list index
            # print(results)
            results = {
                k: v
                for k, v in sorted(
                    results.items(), key=lambda item: item[1], reverse=False
                )  # reverse=False to find the best agents with lowest loss
            }
            # print(results)
            elites = list(results.keys())[:num_elites]
            q = 0  # used for indexing elites
            new_population = []

            for elite in elites:
                torch.save(population[elite].state_dict(), "zlparent" + str(q) + ".pt")
                new_population.append(population[elite])
                q += 1
            random_integers = np.random.randint(
                0, num_elites, size=(POPULATION_SIZE - num_elites) * 2
            )
            f = 0  # indexing rest of agents
            for _ in range(
                POPULATION_SIZE - num_elites
            ):  # exclude parents, already included
                child = Agent()
                parent1 = population[random_integers[f]]
                parent2 = population[random_integers[f + 1]]
                for name, param in child.named_parameters():
                    if np.random.rand() > 0.5:
                        param.data.copy_(parent1.state_dict()[name].data)
                    else:
                        param.data.copy_(parent2.state_dict()[name].data)
                child = mutate(child, np.random.rand())
                new_population.append(child)
                f += 2
            population = new_population
            del new_population
        with open("progress.txt", "w") as f:  # overwrite file contents
            f.write(str(completed) + " " + str(size))
        completed = counter * size
        counter += 1
