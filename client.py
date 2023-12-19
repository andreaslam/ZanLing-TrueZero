import re
import socket
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
import torch.nn as nn
import json
import math


def make_msg_send(
    is_continue,
    initialise_identity,
    nps,
    evals_per_second,
    job_path,
    net_request,
    has_net,
    purpose,
):
    return {
        "is_continue": is_continue,  #  set false if sending stop signal
        "initialise_identity": initialise_identity,  #  rust-datagen
        "nps": nps,  #  nps statistics
        "evals_per_second": evals_per_second,  # evals/s statistics
        "job_path": job_path,  #  game file path
        "net_path": net_request,
        "has_net": has_net,
        "purpose": purpose,
    }


def make_msg_return(verification, net_path):
    return {
        "verification": verification,  #  set false if sending stop signal
        "net_path": net_path,  #  rust-datagen
    }


class MessageSend(Enum):  # message from python to rust
    PYTHON_ID = "python-training"


class MessageRecv(Enum):  # message from rust to python
    NEW_NETWORK = "send-net"
    STOP_SERVER = "shutdown"
    RUST_ID = "rust-datagen"
    JOB = "jobsend"
    NET_REQUEST = "requesting-net"
    INIT = "initialise"


class Server:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.file = None

    def connect(self) -> None:
        while True:
            try:
                self.socket.connect((self.host, self.port))
                print("Connected to server!")
                self.file = self.socket.makefile("r")
                break
            except ConnectionRefusedError as e:
                print(f"Connection failed: {e}. Retrying...")
                continue

    def send(self, message: dict) -> None:
        obj = json.dumps(message) + "\n"
        self.socket.sendall(obj.encode("utf-8"))
        print(f"[Sent] {message}")

    def receive(self) -> str:
        assert self.file is not None
        return self.file.readline()

    def close(self) -> None:
        self.socket.close()


# Constants
HOST = "127.0.0.1"
PORT = 8080
BUFFER_SIZE = 50000
BATCH_SIZE = 32768  # (power of 2, const)

assert BATCH_SIZE > 0 and (BATCH_SIZE & (BATCH_SIZE - 1)) == 0

SAMPLING_RATIO = 2  # how often to train on each pos


def load_file(games_path: str):
    game = Game.find("chess")
    data = DataFile.open(game, games_path)
    return data


def main():
    # Initialisation
    game = Game.find("chess")

    if torch.cuda.is_available():
        d = torch.device("cuda")
    else:
        d = torch.device("cpu")

    print("Using: " + str(d))

    # check if neural net folder exists

    if not os.path.exists("nets"):  # no net
        os.makedirs("nets")

        # initialise a fresh batch of NN, if not already

    # load the latest generation net

    training_nets = list(
        filter(lambda x: x.startswith("tz_") and x.endswith(".pt"), os.listdir("nets"))
    )

    training_nets.sort()

    training_nets = ["nets/" + x for x in training_nets if os.path.isfile("nets/" + x)]
    print(training_nets)
    if (
        len(os.listdir("nets")) == 0 or len(training_nets) == 0
    ):  # no net folder or no net
        with torch.no_grad():
            net = torch.jit.script(
                network.TrueNet(num_resBlocks=2, num_hidden=64, head_nodes=100).to(d)
            )
            net.eval()
            torch.jit.save(
                net, "nets/tz_0.pt"
            )  # if it doesn't exist, create one and save into folder

        with open(
            "traininglog.txt", "w+"
        ) as f:  # overwrite all content and start new training session
            f.write("nets/tz_0.pt\n")

    # load the latest generation net
    if os.path.isfile("traininglog.txt"):  # yes log, yes net
        with open("traininglog.txt", "r") as f:  # resume previous training session
            recorded_sessions = f.readlines()
        print(len(recorded_sessions), len(training_nets))
        if len(recorded_sessions) != len(training_nets):
            with open("traininglog.txt", "w") as f:
                for net in training_nets:
                    f.write(net + "\n")
            with open("traininglog.txt", "r") as f:  # resume previous training session
                recorded_sessions = f.readlines()
    else:  # no log, yes net
        f.write(training_nets[-1] + "\n")
    model_path = recorded_sessions[-1].strip()  # take latest entry as starting

    print(model_path)
    model = torch.jit.load(model_path, map_location=d)

    model.eval()

    pattern = r"tz_(\d+)\.pt"

    starting_gen = int(
        re.findall(pattern, model_path)[0]
    )  # can do this since only 1 match per file maximum

    print("starting generation:", starting_gen)
    server = Server(HOST, PORT)
    server.connect()

    # login loop
    data_paths = None
    if os.path.isfile(
        "datafile.txt"
    ):  # create the file if it doesn't exist, this file stores the path of training data and reset every time after a net has been saved
        with open(
            "datafile.txt", "r"
        ) as f:  # overwrite all content and start new training session
            data_paths = f.readlines()
            data_paths = [
                x
                for x in data_paths
                if os.path.isfile(x + ".bin")
                and os.path.isfile(x + ".json")
                and os.path.isfile(x + ".off")
            ]
            data_paths = None if len(data_paths) == 0 else data_paths
    else:  # no file
        with open("datafile.txt", "w+") as f:
            game_files = os.listdir("games")
            filtered_files = set()
            for string in game_files:
                split_string = string.split(".", 1)
                truncated_path = (
                    split_string[0] if len(split_string) > 1 else string
                )  # get the part before the first ".",
                # reason is that loopbuffer opens a path not a file with extensions
                filtered_files.add(truncated_path)

            data_paths = [x for x in filtered_files if os.path.isfile(x)]
            print(data_paths)

    while True:
        server.send(
            make_msg_send(
                True,
                "python-training",
                None,
                None,
                None,
                model_path,
                True,
                MessageRecv.INIT.value,
            )
        )
        received_data = server.receive()
        received_data = json.loads(received_data)
        if MessageSend.PYTHON_ID.value in received_data.values():
            break

    loopbuf = LoopBuffer(
        Game.find("chess"), target_positions=BUFFER_SIZE, test_fraction=0.2
    )

    train_settings = TrainSettings(
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

    while True:
        log = Logger()
        log.start_batch()
        received_data = server.receive()
        received_data = json.loads(received_data)
        print(f"[Received] {received_data}")
        if data_paths:
            for file in data_paths:
                data = load_file(file)
                loopbuf.append(log, data)
        if (
            MessageSend.PYTHON_ID.value in list(received_data.values())
            or MessageRecv.RUST_ID.value in list(received_data.values())
            or MessageRecv.NET_REQUEST.value in list(received_data.values())
        ):
            msg = make_msg_send(
                True,
                None,
                None,
                None,
                None,
                model_path,
                True,
                MessageRecv.NEW_NETWORK.value,
            )
            server.send(msg)

        if MessageRecv.JOB.value in list(received_data.values()):
            file_path = received_data["job_path"]
            data = load_file(file_path)
            loopbuf.append(log, data)
            with open("datafile.txt", "a") as f:
                f.write(file_path + "\n")
            print("buffer size:", loopbuf.position_count)
            if loopbuf.position_count >= BUFFER_SIZE:
                sample = loopbuf.sampler(
                    batch_size=BATCH_SIZE,
                    unroll_steps=None,
                    include_final=False,
                    random_symmetries=False,
                    only_last_gen=False,
                    test=True,
                )

                num_steps_training = (
                    len(data.positions) / BATCH_SIZE
                ) * SAMPLING_RATIO  # calculate number of training steps to take
                if num_steps_training < 1:
                    print("[Warning] minimum training step is", num_steps_training)
                    num_steps_training = 1
                    print("[Warning] set training step to 1")
                num_steps_training = int(num_steps_training)
                for _ in range(num_steps_training):
                    log.start_batch()
                    batch = sample.next_batch()
                    train_settings.train_step(
                        batch, network=model, optimizer=op, logger=log
                    )
                log.save("log.npz")
                sample.close()
                starting_gen += 1
                model_path = "nets/tz_" + str(starting_gen) + ".pt"
                print(model_path)
                with torch.no_grad():
                    torch.jit.save(model, model_path)
                with open("traininglog.txt", "a") as f:
                    f.write(model_path + "\n")
                with open("datafile.txt", "w") as f:
                    pass

                # send to rust server
                msg = make_msg_send(
                    True, None, None, None, None, model_path, True, "send-net"
                )
                server.send(msg)

        if MessageRecv.STOP_SERVER.value in received_data.values():
            server.close()
            print("Connection closed.")
            break


if __name__ == "__main__":
    main()
