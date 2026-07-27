# train a brand new net instance without any selfplay
import os

import network as network
import torch
from lib.data.file import DataFile
from lib.games import Game
from lib.logger import Logger
from lib.loop import LoopBuffer
from lib.train import ScalarTarget, TrainSettings
from paths import data_dir, data_path
from torch import optim
from tqdm import tqdm


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

model = torch.jit.script(network.TrueNet().to(d)).eval()
data_dir("experiment_nets")
torch.jit.save(model, data_path("experiment_nets/tz_test_0.pt"))

LOG_EXPERIMENT_PATH = data_path("log_experiment.npz")
with open(LOG_EXPERIMENT_PATH, "w") as f:
    f.write("")

# training-regime constants
# the previous regime ran far too many optimizer steps per generation over a tiny
# effective replay window, overfitting noisy self-play targets (policy loss stalled,
# last-gen test loss > train loss). AlphaZero/LC0/KataGo keep the optimisation-to-new-
# data ratio near ~1 epoch over a LARGE, decorrelated window of many recent generations.


# keep a large replay window (many generations) so targets are decorrelated,
# cap the number of optimizer steps so each new generation is seen ~`EPOCHS_PER_GEN`
# times rather than hundreds of times.
BUFFER_SIZE = 1_500_000  # ~ matches replay capacity; keep several generations
BATCH_SIZE = 2048
# How many times each new position is (on average) trained on. ~1.0 is the AlphaZero/LC0
# regime; the old config effectively used hundreds.
EPOCHS_PER_GEN = 1.0
loopbuf = LoopBuffer(
    Game.find("chess"), target_positions=BUFFER_SIZE, test_fraction=0.1
)

train_settings = TrainSettings(
    game=game,
    scalar_target=ScalarTarget.Final,
    value_weight=0.1,
    wdl_weight=0.1,
    moves_left_weight=0.1,
    moves_left_delta=0.1,
    policy_weight=1,
    sim_weight=0.0,
    train_in_eval_mode=False,
    clip_norm=5.0,
    mask_policy=True,
)

op = optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=0.01)
log = Logger()

data_paths = []
game_folder = data_dir("python_client_games")
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
    if os.path.exists(LOG_EXPERIMENT_PATH):
        try:
            log = log.load(LOG_EXPERIMENT_PATH)
        except Exception:
            os.remove(LOG_EXPERIMENT_PATH)  # reset
    # print("[loaded files] buffer size:", loopbuf.position_count)
    for file in os.listdir(game_folder):
        data_paths.append(f"{game_folder}/" + file)

    data_paths = [x.split(".")[0] for x in data_paths]
    data_paths = set(data_paths)
else:
    print("no files!")

# print(loopbuf.position_count)

# Optimizer steps per generation. We derive this from the amount of NEW data added each
# generation so that new positions are trained ~EPOCHS_PER_GEN times, instead of a fixed
# (and previously wildly excessive) count. We read the current buffer's newest generation
# size; if unavailable we fall back to a conservative default.
POSITIONS_PER_GEN_ESTIMATE = 25_000  # measured: gen-size/positions ~ 2.5e4 in log.npz
num_steps_training = max(
    1, int(EPOCHS_PER_GEN * POSITIONS_PER_GEN_ESTIMATE / BATCH_SIZE)
)
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
        if gen != 0:
            log.start_batch()
        batch = train_sampler.next_batch()
        train_settings.train_step(batch, network=model, optimizer=op, logger=log)

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
        log.save(LOG_EXPERIMENT_PATH)
    except Exception:
        print("[Warning] failed to save log_experiment.npz")

    train_sampler.close()
    test_sampler.close()
    last_gen_test_sampler.close()
    starting_gen += 1
    model_path = data_path("experiment_nets/tz_test_" + str(starting_gen) + ".pt")
    model.eval()
    with torch.no_grad():
        torch.jit.save(model, model_path)
