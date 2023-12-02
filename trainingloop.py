import einops
import torch
import torch.nn.functional as nnf
import torchvision.utils
import torch.nn as nn
import torch.optim as optim
from lib.data.file import DataFile
from lib.data.position import PositionBatch
from lib.games import Game
from lib.logger import Logger
from lib.train import TrainSettings
from lib.train import ScalarTarget
from queue import Queue





def load_file(games_path: str):
    
    game = Game.find("chess")
    print(games_path)
    data = DataFile.open(game, games_path)
    
    print(data.positions)


def train():
    print("HI")

    game = Game.find("chess")

    games_path = r"C:\Users\andre\RemoteFolder\tz-rust\sample.off"

    data = DataFile.open(game, games_path)
    p = data.positions[0]

    model_path = r"C:\Users\andre\RemoteFolder\tz-rust\chess_16x128_gen3634.pt"
    model = torch.jit.load(model_path, map_location=torch.device("cuda"))

    model.eval()

    b = PositionBatch(game, [p], False, False)
    input_full = b.input_full.to("cuda")

    # print(input_full.shape)
    # print(input_full.dtype)
    # torchvision.utils.save_image(
    #     einops.rearrange(input_full, "b c h w -> (b c) 1 h w"),
    #     "chess_input.png", nrow=21, pad_value=0.4
    # )

    train = TrainSettings(
        game=game,
        scalar_target=ScalarTarget.Final,
        value_weight=0.1,
        wdl_weight=0.0,
        moves_left_weight=0.0,
        moves_left_delta=0.0,
        policy_weight=1,
        sim_weight=0.0,
        train_in_eval_mode=False,
        clip_norm=5.0,
        mask_policy=True,
    )

    op = optim.AdamW(params=model.parameters())

    log = Logger()
    log.start_batch()
    train.train_step(batch=b, network=model, optimizer=op, logger=log)

    with torch.no_grad():
        batch_scalar_logits, batch_policy_logits = model(input_full)
        print((batch_scalar_logits, batch_policy_logits))

        scalar_logits = batch_scalar_logits[0]

        print("value:", scalar_logits[0].tanh())
        print("wdl:", nnf.softmax(scalar_logits[1:4], -1))
