# train a brand new net instance without any selfplay
import time
from enum import Enum
from lib.data.file import DataFile
from lib.train import ScalarTarget, TrainSettings
from lib.games import Game
import torch
from lib.loop import LoopBuffer
from lib.logger import Logger
import os
import torch.optim as optim
import network
from tqdm import tqdm
import time
import torch.nn as nn
import client


def load_file(games_path: str):
    game = Game.find("chess")
    data = DataFile.open(game, games_path)
    return data


game = Game.find("chess")

if torch.cuda.is_available():
    d = torch.device("cuda")
else:
    d = torch.device("cpu")

# print("Using: " + str(d))

# model = torch.jit.script(network.TrueNetXS(16384)).to(d)
model = torch.jit.script(
    network.TrueNet(
        num_resBlocks=8,
        num_hidden=64,
        head_channel_policy=8,
        head_channel_values=4,
    )
).to(d)
model.eval()
# model = torch.jit.load("tz_6515.pt", map_location=d).eval()
if not os.path.exists("experiment_nets"):  # no net
    os.makedirs("experiment_nets")
torch.jit.save(model, "experiment_nets/tz_test_0.pt")

with open("log_experiment.npz", "w") as f:
    f.write("")

BUFFER_SIZE = 15000000
BATCH_SIZE = 16384
loopbuf = LoopBuffer(
    Game.find("chess"), target_positions=BUFFER_SIZE, test_fraction=0.1
)

train_settings = TrainSettings(
    game=game,
    scalar_target=ScalarTarget.Final,
    value_weight=1.0,
    wdl_weight=0.0,
    moves_left_weight=0.0,
    moves_left_delta=0.0,
    policy_weight=1,
    sim_weight=0.0,
    train_in_eval_mode=False,
    clip_norm=5.0,
    mask_policy=True,
)

op = optim.AdamW(params=model.parameters(), lr=5e-2)
log = Logger()

data_paths = []
game_folder = "games"
for file in os.listdir(game_folder):
    data_paths.append(f"{game_folder}/" + file)

data_paths = [x.split(".")[0] for x in data_paths]
data_paths = set(data_paths)
if data_paths:
    data_paths = list(dict.fromkeys(data_paths))  # remove duplicates
    for file in data_paths:
        # print(file)
        try:
            data = load_file(file)
            loopbuf.append(None, data)
        except Exception:
            continue
    if os.path.exists("log_experiment.npz"):
        try:
            log = log.load("log_experiment.npz")
            # print("loaded log")
        except Exception as e:
            # print("[Error]", e)
            os.remove("log_experiment.npz")  # reset
    # print("[loaded files] buffer size:", loopbuf.position_count)
    for file in os.listdir(game_folder):
        data_paths.append(f"{game_folder}/" + file)

    data_paths = [x.split(".")[0] for x in data_paths]
    data_paths = set(data_paths)
else:
    print("no files!")

# print(loopbuf.position_count)

num_steps_training = 10
starting_gen = 0
while True:
    train_sampler = loopbuf.sampler(
        batch_size=BATCH_SIZE,
        unroll_steps=None,
        include_final=False,
        random_symmetries=False,
        only_last_gen=False,
        test=False,
    )

    test_sampler = loopbuf.sampler(
        batch_size=BATCH_SIZE,
        unroll_steps=None,
        include_final=False,
        random_symmetries=False,
        only_last_gen=False,
        test=True,
    )
    last_gen_test_sampler = loopbuf.sampler(
        batch_size=BATCH_SIZE,
        unroll_steps=None,
        include_final=False,
        random_symmetries=False,
        only_last_gen=True,
        test=True,
    )
    log.start_batch()
    model.train()
    for gen in tqdm(range(num_steps_training)):
        # print("starting training")
        if gen != 0:
            log.start_batch()
        # print("starting sampling")
        batch = train_sampler.next_batch()
        # print("sampled")
        # print("training")
        train_settings.train_step(batch, network=model, optimizer=op, logger=log)
        # print("train step complete")

    with torch.no_grad():
        model.eval()
        test_batch = test_sampler.next_batch()
        train_settings.evaluate_batch(
            network=model, batch=test_batch, log_prefix="test", logger=log
        )
        last_gen_test_batch = last_gen_test_sampler.next_batch()
        train_settings.evaluate_batch(
            network=model,
            batch=last_gen_test_batch,
            log_prefix="last gen test",
            logger=log,
        )

    log.finished_data()
    try:
        log.save("log_experiment.npz")
    except Exception:
        print("[Warning] failed to save log_experiment.npz")

    train_sampler.close()
    test_sampler.close()
    last_gen_test_sampler.close()
    starting_gen += 1
    model_path = "experiment_nets/tz_test_" + str(starting_gen) + ".pt"
    # print(model_path)
    model.eval()
    with torch.no_grad():
        torch.jit.save(model, model_path)
