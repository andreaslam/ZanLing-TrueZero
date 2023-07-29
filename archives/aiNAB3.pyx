# FIRST GEN WITH ALPHA BETA PRUNING
# zan1ling4 真零 | TrueZero
# imports
import numpy as np
import chess
from chess import Move
import torch
from torch import nn
import tqdm
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

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
        matrix_game = torch.tensor(matrix_game, dtype=torch.float32).to("cuda")
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


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return F.gelu(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.01):
        super(TransformerModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = TransformerEncoderLayer(hidden_dim, nhead=8)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.act = GELU()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 1)
        x = self.transformer_encoder(x)
        x = x.permute(0, -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.act(x)
        return x
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def generate_move(self, board, depth=10):
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
            board.pop()
        m_dict = {
            k: v
            for k, v in sorted(
                m_dict.items(), key=lambda item: item[1], reverse=True
            )  # reverse=False to find the best move with highest score
        }
        best_move = list(m_dict.keys())[0]  # best move, first key
        with open("./games.txt", "a+") as f:
            f.write(str(best_move) + "\n")
        del m_dict
        return best_move


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
                loss = agent1.loss(outputs, targets)
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
POP_SIZE = 10  # 10
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
population = [TransformerModel(input_dim=833, hidden_dim=2048, output_dim=1, num_layers=4).to('cuda') for _ in range(POP_SIZE)]
num_elites = int(POP_SIZE * 0.4)
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
    s_table = {}  # TODO: implement fairer logic for evaluation
    for x in range(POP_SIZE):
        s_table[x] = 0
    for i in range(POP_SIZE):
        for j in range(POP_SIZE):
            if i != j:
                mutate(population[i], np.random.rand(), np.random.randint(-10, 10))
                mutate(population[j], np.random.rand(), np.random.randint(-10, 10))
                raw_score = play_game(population[i], population[j], population)
                if raw_score == -5:  # draw
                    s_table[j] -= 5
                    s_table[i] -= 5
                else:
                    s_table[i] += raw_score
                    s_table[j] += raw_score * -1
    # Rank the agents by their scores
    s_table = {
        k: v
        for k, v in sorted(
            s_table.items(), key=lambda item: item[1], reverse=True
        )  # reverse=False to find the best agent with highest score
    }
    best_agents = list(s_table.keys())[:num_elites]  # best agent, first key
    print(s_table)
    print(best_agents)
    is_winning = False
    winner_count = 0
    for game_score in s_table.values():
        if game_score > 0:
            is_winning = True
            winner_count += 1

    # Save the weights of the best agents
    if is_winning is True:
        if winner_count > num_elites:
            num_of_saves = num_elites
        else:
            num_of_saves = winner_count
        counter = 0
        for agent_index in best_agents:
            torch.save(
                population[agent_index].state_dict(),
                "./best_agents" + str(counter) + ".pt",
            )
            counter += 1

    # Create a new population through selection, crossover, and mutation
    new_population = []

    # Keep the top-performing agents in the population
    for elite in best_agents:
        new_population.append(population[elite])
    # Select parents for breeding using tournament selection
    for _ in range(POP_SIZE):  # exclude parents, already included
        random_parents = np.random.randint(0, len(new_population), size=2)
        parent1 = new_population[random_parents[0]]
        parent2 = new_population[random_parents[1]]

        child = TransformerModel(input_dim=833, hidden_dim=8192, output_dim=1, num_layers=12).to('cuda') # implement the transformer here
        for name, param in child.named_parameters():
            if np.random.rand() > 0.5:
                param.data.copy_(parent1.state_dict()[name].data)
            else:
                param.data.copy_(parent2.state_dict()[name].data)
        child = mutate(child, np.random.rand(), np.random.rand())
        new_population.append(child)
        population = new_population
