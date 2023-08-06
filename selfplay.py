import torch
import mcts_trainer
import chess
import network
import os
import pickle  # probably serde in rust impl
import torch.optim
import torch.nn.functional as F


if torch.cuda.is_available():
    d = torch.device("cuda")
else:
    d = torch.device("cpu")

print("Using: " + str(d))


# create TrueZero class (different from network class)
class TrueZero:
    def __init__(self, net, optimiser, iterations, epochs, batch_size) -> None:
        self.net = net  # use pretrained net for now
        self.iterations = iterations
        self.optimiser = optimiser
        self.num_epochs = epochs
        self.batch_size = batch_size
        
        

    def self_play(self):
        board = chess.Board()
        memory = []  # board only, torch.Size([B, 21, 8, 8])
        pi_list = []
        while not board.is_game_over():
            best_move, memory_piece, pi, move_idx = mcts_trainer.move(
                board, self.net
            )  # pi is going to be the target for training
            board.push(chess.Move.from_uci(best_move))
            print(best_move)
            pi_full = torch.zeros(1880)
            # print(move_idx)
            for index, value in zip(move_idx, pi):
                pi_full[index] = value
            pi = pi_full
            # print(pi)
            memory.append(memory_piece)
            pi_list.append(pi)
        memory = torch.stack(memory)
        # print(memory.shape)
        # get game outcome
        z = 0
        outcome = board.result()
        print("GAME OVER", outcome)
        training_ready_data = []
        for mem, pl in zip(memory, pi_list):
            # access move turn
            move_turn = mem[0][0][0]
            # print(move_turn)
            # since the first square is white's turn, if it's white's turn it's true (1)
            # if it's black's turn, the square is false (0)
            if (outcome == "1-0" and move_turn == 1) or (
                outcome == "0-1" and move_turn == 0
            ):
                z = 1
            elif (outcome == "0-1" and move_turn == 1) or (
                outcome == "1-0" and move_turn == 0
            ):
                z = -1
            elif outcome == "1/2-1/2":  # explicit is better than implicit
                z = 0
            # print(z)
            training_ready_data.append([mem, z, pl])

        return training_ready_data

    def training_loop(self):
        self.net.train()  # training mode, different from self.train(), which is the process of training
        for epoch in range(self.num_epochs):
            training, value_target, policy_target = self.request_rand_states()
            training = training.to(d)
            out_policy, out_value = self.net(training)
            policy_loss = F.cross_entropy(out_policy, policy_target.to(d))  # target pi
            value_loss = F.mse_loss(out_value.squeeze(), value_target.to(d))  # target z
            loss = policy_loss.float() + value_loss.float()
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            loss = loss.item()
            print("current loss:", loss)
            torch.save(self.net.state_dict(), "tz.pt")
            # torch.save(self.optimiser.state_dict(), "optimiser.pt")
            self.net.weights_path = "tz.pt"
            self.net.optimiser_path = "optimiser.pt"

    def data_loop(self):
        memory = []
        try:
            self.net.load_state_dict(torch.load(net.weights_path, map_location=d))
            self.optimiser.load_state_dict(torch.load(net.optimiser_path, map_location=d))
        except FileNotFoundError:
            pass
        for iteration in range(self.iterations):
            memory.append(self.self_play())
        with open("data.bin", "wb") as f:
            pickle.dump(memory, f)

    def request_rand_states(self):
        # load the games from the bin
        with open("data.bin", "rb") as f:
            data = pickle.load(f)
        # [n] - access the nth game
        # [n][n] - access the nth move in the nth game

        min_value = 0  # Minimum value for random integers
        max_value = len(data)  # Maximum value for random integers

        # print("MAXV", max_value)
        num_samples = min(
            len(data), 128
        )  # you can't sample more than you have, 128 positions max/pass
        # print("NS", num_samples)
        random_games = torch.randint(
            min_value, max_value, size=(num_samples,)
        )  # indexes of random games to sample

        # print(random_games)

        training_batch = []
        value_target = []
        policy_target = []
        for game_idx in random_games:
            # print(game_idx)
            picked_game = data[game_idx]
            max_value = len(picked_game[0])
            number_of_samples = torch.randint(
                1, max_value + 1, size=(1,)
            )  # how many samples to pick?
            random_idx = torch.randint(
                min_value, max_value + 1, size=(number_of_samples,)
            )  # what are the indexes of the selected games?
            for game in random_idx:
                training_batch.append(picked_game[game][0])
                p = picked_game[game][2].clone().detach()
                value_target.append(torch.tensor(picked_game[game][1]))
                policy_target.append(torch.tensor(p.clone().detach()))
        training_batch = torch.stack(training_batch)
        value_target = torch.stack(value_target).float()
        policy_target = torch.stack(policy_target)
        return training_batch, value_target, policy_target


net = network.TrueNet(num_resBlocks=10, device=d, num_hidden=128)

optim = torch.optim.AdamW(lr=1e-3, params=net.parameters(), weight_decay=1e-10)
tz = TrueZero(net, optim, 20, 25, 100)

while True:
    tz.data_loop()
    tz.training_loop()
    os.remove("data.bin")
    print(tz.net.weights_path)
    print(tz.net.optimiser_path)
