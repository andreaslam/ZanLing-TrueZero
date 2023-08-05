import torch
import mcts_trainer
import chess
import network
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
            best_move, memory_piece, pi = mcts_trainer.move(
                board, self.net
            )  # pi is going to be the target for training
            board.push(chess.Move.from_uci(best_move))
            print(best_move)
            memory.append(memory_piece)
            pi_list.append(pi)
        memory = torch.stack(memory)
        print(memory.shape)
        # get game outcome
        z = 0
        outcome = board.result()
        print("GAME OVER",outcome)
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
        for epoch in self.num_epochs:
            training, target = self.request_rand_states()
            training, target = training.to(d), target.to(d)
            out_policy, out_value = self.net(training)

            policy_loss = F.cross_entropy(out_policy, target[1])  # target pi
            value_loss = F.mse_loss(out_value, target[0])  # target z
            loss = policy_loss + value_loss

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            torch.save(self.net.state_dict(), "tz_" + str(epoch) + ".pt")
            torch.save(self.optimiser.state_dict(), "optimiser_" + str(epoch) + ".pt")

    def data_loop(self):
        memory = []
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

        num_samples = min(
            len(data), 128
        )  # you can't sample more than you have, 128 positions max/pass

        random_games = torch.randint(
            min_value, max_value + 1, size=(num_samples,)
        )  # indexes of random games to sample

        training_batch = []
        target_batch = []
        for game_idx in random_games:
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
                target_batch.append((picked_game[game][1], picked_game[game][2]))

            training_batch = torch.stack(training_batch)
            target_batch = torch.stack(target_batch)

        return training_batch, target_batch


net = network.TrueNet(num_resBlocks=1, device=d, num_hidden=128)

optim = torch.optim.AdamW(lr=1e-3, params=net.parameters(), weight_decay=1e-4)
tz = TrueZero(net, optim, 10, 100, 128)

while True:
    tz.data_loop()
    tz.training_loop()
