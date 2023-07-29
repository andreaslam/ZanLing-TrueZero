# FIRST GEN WITH ALPHA BETA PRUNING
# zan1ling4 真零 | TrueZero
# imports
import numpy as np
import chess 
from chess import Move
import torch
from torch import nn
from torch import optim
import torch.nn.init as init
import tqdm


def negamax_ab(board, alpha, beta, colour, model, depth=2):
    if depth == 0 or board.is_game_over():  # check if the depth is 0 or "terminal node"
        if colour == 1:
            move_turn = 0  # my eval accepts 0 for white and black for 1 :/
        else:
            move_turn = 1
        matrix_game = np.array([board_data(board)])  # game after one hot encoding
        matrix_game = np.concatenate(
            (matrix_game, np.array([[move_turn]])), axis=1
        )  # have to append the move turn - the AI needs to know this
        matrix_game = torch.tensor(matrix_game, dtype=torch.float32)
        score = model(
            matrix_game
        )  # EVALUTATION - high score for winning (if white/black wins, high score, vice versa)
        score = float(score)
        if board.is_game_over():
            score = 2
        return score * colour

    child_nodes = list(board.legal_moves)
    # child_nodes = order_moves(child_nodes) # make an ordermove function
    best_score = -np.inf
    for child in child_nodes:
        board.push(child)  # Push the current child move on the board
        score = -negamax_ab(board, -beta, -alpha, -colour, model, depth - 1)
        board.pop()  # Pop the current child move from the board

        best_score = max(best_score, score)
        alpha = max(alpha, best_score)
        if alpha >= beta:
            break

    return best_score


class Tanh200(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 200)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(833, 512)
        self.dropout1 = nn.Dropout(p=0.25)
        self.relu = nn.LeakyReLU(0.05)
        self.layer2 = nn.Linear(512, 1)
        self.dropout2 = nn.Dropout(p=0.25)
        self.tanh200 = Tanh200()
        self.hidden_layers = nn.ModuleList()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.5
        )
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

    def generate_move(self, board, depth=3):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        if board.turn == chess.WHITE:
            colour = 1
        else:
            colour = -1
        best_move = None
        m_dict = {}
        for move in legal_moves:
            board.push(move)
            move_score = negamax_ab(board, np.inf, np.inf, colour, self, depth)
            m_dict[str(move)] = move_score
            m_dict = {
                k: v
                for k, v in sorted(
                    m_dict.items(), key=lambda item: item[1], reverse=True
                )  # reverse=False to find the best move with highest score
            }
            if colour == 1:
                best_move = list(m_dict.keys())[0]  # best move, first key
            else:
                best_move = list(m_dict.keys())[-1]
            board.pop()
        with open("./games.txt", "a+") as f:
            f.write(str(best_move) + "\n")
        del m_dict
        return best_move

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


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


def play_game(agent1, agent2, population):
    board = chess.Board()
    batch_inputs = []
    batch_targets = []

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            agent1 = population[i]
            agent2 = population[j]

            similarity_threshold = 0.90
            cosine_similarity = nn.CosineSimilarity(dim=0)
            euclidean_distance = nn.PairwiseDistance(p=2, keepdim=True)
            weights1 = [param.data.flatten() for param in agent1.parameters()]
            weights2 = [param.data.flatten() for param in agent2.parameters()]
            for w1, w2 in zip(weights1, weights2):
                distance = euclidean_distance(
                    w1.clone().detach(), w2.clone().detach()
                ).item()
                similarity = cosine_similarity(w1, w2).item()
                if similarity > similarity_threshold or distance > similarity_threshold:
                    # Mutate one of the agents
                    if np.random.rand() < 0.5:
                        agent1 = mutate(agent1, np.random.rand(), np.random.rand())
                    else:
                        agent2 = mutate(agent2, np.random.rand(), np.random.rand())

            while not board.is_game_over():
                if board.turn == chess.WHITE:
                    move = agent1.generate_move(board)
                else:
                    move = agent2.generate_move(board)
                move = Move.from_uci(move)
                board.push(move)
                if board.turn == chess.WHITE:
                    move_turn = 0
                    batch_inputs.append(
                        np.concatenate(
                            (np.array([board_data(board)]), np.array([[move_turn]])),
                            axis=1,
                        )
                    )
                    batch_targets.append(
                        agent1.forward(
                            torch.tensor(
                                np.concatenate(
                                    (
                                        np.array([board_data(board)]),
                                        np.array([[move_turn]]),
                                    ),
                                    axis=1,
                                ),
                                dtype=torch.float,
                            )
                        ).item()
                    )
                else:
                    move_turn = 1
                    batch_inputs.append(
                        np.concatenate(
                            (np.array([board_data(board)]), np.array([[move_turn]])),
                            axis=1,
                        )
                    )
                    batch_targets.append(
                        agent2.forward(
                            torch.tensor(
                                np.concatenate(
                                    (
                                        np.array([board_data(board)]),
                                        np.array([[move_turn]]),
                                    ),
                                    axis=1,
                                ),
                                dtype=torch.float,
                            )
                        ).item()
                    )
            # Append final target value when game is over
            with open("games.txt", "a+") as f:
                f.write(board.result() + "\n" + "===" + "\n")
            if board.result() == "1-0":
                score = 10
            elif board.result() == "0-1":
                score = -10
            else:
                score = -5
            batch_targets[-1] = score  # set the last target value to score
            # Train the agents on the batch
            if len(batch_inputs) > 0:
                batch_inputs = np.array(batch_inputs)
                batch_targets = np.array([[batch_targets]])
                inputs = torch.tensor(batch_inputs, dtype=torch.float)
                targets = torch.tensor(batch_targets, dtype=torch.float)
                targets = torch.reshape(targets, (-1, 1, 1))
                outputs = agent1.forward(inputs)

                loss = agent1.loss(outputs, targets)  #
                loss.backward()
                # NOTE: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
                agent1.optimizer.zero_grad()
                agent1.optimizer.step()
                agent1.scheduler.step()
                outputs = agent2.forward(inputs)
                loss = agent2.loss(outputs, targets)
                loss.backward()
                agent2.optimizer.zero_grad()
                agent2.optimizer.step()
                agent2.scheduler.step()
            board = chess.Board()
            batch_inputs = []
            batch_targets = []
            return score


# Define genetic algorithm parameters
POP_SIZE = 5  # 10
NUM_EPOCHS = 100  # 100
MUTATION_RATE = 0.5

# Initialise documents

with open("games.txt", "a+") as f:
    f.write("\n")


def mutate(agent, mutation_rate, score):
    # Calculate the new mutation rate based on the game score
    if score < 0:
        mutation_rate = max(0.8, mutation_rate - 0.1)
    elif score > 0:
        mutation_rate = min(0.2, mutation_rate + 0.1)
    # Mutate the agent's parameters
    for param in agent.parameters():
        if np.random.rand() < mutation_rate:
            param.data += torch.randn(param.shape) * np.random.rand()
    return agent


# Initialize the population
population = [Agent() for _ in range(POP_SIZE)]
num_elites = int(POP_SIZE * 0.2)
# Use best agents


for epoch in tqdm.tqdm(range(NUM_EPOCHS), desc="each epoch"):
    for agent in population:
        decider = np.random.rand()
        if decider > 0.5:
            try:
                index_chooser = np.random.randint(0, num_elites)
                agent.load_weights("./best_agents" + str(index_chooser) + ".pt")
            except FileNotFoundError:
                pass

    with open("games.txt", "a+") as f:
        f.write("Epoch " + str(epoch) + "\n")
    # Play each agent against every other agent in the population
    scores = np.zeros((POP_SIZE, POP_SIZE))
    val_table = []
    for i in range(POP_SIZE):
        for j in range(POP_SIZE):
            if i != j:
                mutate(population[i], np.random.rand(), np.random.randint(-10, 10))
                mutate(population[j], np.random.rand(), np.random.randint(-10, 10))
                scores[i][j] = play_game(population[i], population[j], population)
                val_table.append(scores[i][j])
    # Rank the agents by their scores
    ranked_indices = np.argsort(np.sum(scores, axis=1))[::-1]

    ranked_population = [population[i] for i in ranked_indices]
    is_winning = False
    winner_count = 0
    for game_score in val_table:
        if game_score > 0:
            is_winning = True
            winner_count += 1

    # Save the weights of the best agents
    if is_winning is True:
        if winner_count > num_elites:
            num_of_saves = num_elites
        else:
            num_of_saves = winner_count
        for agent_index in range(num_of_saves):
            torch.save(
                ranked_population[agent_index].state_dict(),
                "./best_agents" + str(agent_index) + ".pt",
            )

    # Create a new population through selection, crossover, and mutation
    new_population = []

    # Keep the top-performing agents in the population
    elites = ranked_population[:num_elites]
    for elite in elites:
        new_population.append(elite)
    # Select parents for breeding using tournament selection
    tournament_size = 3  # 4
    for i in range(num_elites, POP_SIZE):
        parent1 = None
        parent2 = None
        for _ in range(tournament_size):
            idx = np.random.randint(POP_SIZE)
            if parent1 is None or scores[idx][i] > scores[parent1][i]:
                p1 = idx
            elif parent2 is None or scores[idx][i] > scores[parent2][i]:
                p2 = idx
                population[i].optimizer.step()
                population[i].scheduler.step()
                parent1 = ranked_population[p1]
                parent2 = ranked_population[p2]

                # Perform crossover to create a new child
                child = Agent()
                for name, param in child.named_parameters():
                    if np.random.rand() > 0.5:
                        param.data.copy_(parent1.state_dict()[name].data)
                    else:
                        param.data.copy_(parent2.state_dict()[name].data)

                # Perform mutation on the child
                for agent_score, agent in zip(scores, population):
                    # Mutate the agent based on its score
                    avg_score = 0
                    avg_counter = 0
                    for item in agent_score:
                        if item != 0:
                            avg_score += item
                            avg_counter += 1
                    avg_score = avg_score / avg_counter
                    agent = mutate(agent, MUTATION_RATE, avg_score)
                    new_population.append(agent)
                    population = new_population
