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
import datetime
import network
import json
import io
import time

def make_msg_send(
    purpose,
):
    return {
        "purpose": purpose,
    }


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
        # print(f"[Sent] {message}")

    def receive(self) -> str:
        assert self.file is not None
        return self.file.readline()

    def close(self) -> None:
        self.socket.close()


# Constants
HOST = "127.0.0.1"
PORT = 38475
BUFFER_SIZE = 500000
BATCH_SIZE = 2048  # (power of 2, const)

assert BATCH_SIZE > 0 and (BATCH_SIZE & (BATCH_SIZE - 1)) == 0

SAMPLING_RATIO = 3  # how often to train on each pos


def serialise_net(model) -> list[int]:
    buffer = io.BytesIO()
    torch.jit.save(model, buffer)
    serialised_data = buffer.getvalue()
    return [byte for byte in serialised_data]


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

    # check if neural net and games folder exists

    if not os.path.exists("nets"):  # no net
        os.makedirs("nets")

    if not os.path.exists("games"):  # no games folder
        os.makedirs("games")

    # initialise a fresh batch of NN, if not already

    # load the latest generation net
    
    pattern = r"tz_(\d+)\.pt"
    training_nets = []
    net_id = {}
    for net in os.listdir("nets"):
        # Use re.match to apply the regex pattern
        match = re.match(pattern, net)
        
        # Check if there's a match
        if match:
            # Extract groups from the match object
            group = int(match.groups()[0]) # only 1 match
            net_id["./nets/"+net] = group
            training_nets.append(net)
    
    net_id = dict(
        sorted(net_id.items(), key=lambda x: x[1])
    )  # sort the entries of the nets and get the latest one

    training_nets = list(net_id.keys())
    
    
    if (
        len(os.listdir("nets")) == 0 or len(training_nets) == 0
    ):  # no net folder or no net
        with torch.no_grad():
            net = torch.jit.script(
                network.TrueNet(
                    num_resBlocks=8,
                    num_hidden=64,
                    head_channel_policy=8,
                    head_channel_values=4,
                ).to(d)
            )
            net.eval()
            torch.jit.save(
                net, "nets/tz_0.pt"
            )  # if it doesn't exist, create one and save into folder

        with open(
            "traininglog.txt", "w+"
        ) as f:  # overwrite all content and start new training session
            f.write("nets/tz_0.pt\n")

        training_nets.append("nets/tz_0.pt")

    # load the latest generation net
    if os.path.isfile("traininglog.txt"):  # yes log, yes net
        with open("traininglog.txt", "r") as f:  # resume previous training session
            recorded_sessions = f.readlines()
            recorded_sessions = [
                item.strip() for item in recorded_sessions if item != ""
            ]
        if recorded_sessions != training_nets:
            with open(
                "traininglog.txt", "w"
            ) as f:  # reset the entries in the document and reset according to available files
                for net in training_nets:
                    f.write(net + "\n")
            with open("traininglog.txt", "r") as f:  # reread the file with updated net
                recorded_sessions = f.readlines()
    else:  # no log, yes net
        with open("traininglog.txt", "r") as f:  # start a new session
            f.write(training_nets[-1] + "\n")
        with open("traininglog.txt", "r") as f:  # reread the file with updated net
            recorded_sessions = f.readlines()

    model_path = recorded_sessions[-1].strip()  # take latest entry as starting

    print(model_path)
    model = torch.jit.load(model_path, map_location=d)

    model.eval()

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
        with open("datafile.txt", "r") as f:
            data_paths = f.readlines()
            data_paths = [item.strip() for item in data_paths if item != ""]
            data_paths = [
                x
                for x in data_paths
                if os.path.isfile(x.strip() + ".bin")
                and os.path.isfile(x.strip() + ".json")
                and os.path.isfile(x.strip() + ".off")
            ]

    else:  # no file, datafile start from scratch
        with open("datafile.txt", "w+") as f:
            pass

    while True:
        server.send(
            make_msg_send(
                {"Initialise": "PythonTraining"},
            )
        )
        received_data = server.receive()
        received_data = json.loads(received_data)
        purpose = str(received_data)
        if "IdentityConfirmation" in purpose:
            if "PythonTraining" in purpose:
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

    op = optim.AdamW(params=model.parameters(), lr=2e-3)
    log = Logger()
    if data_paths:
        data_paths = list(dict.fromkeys(data_paths))  # remove duplicates
        for file in data_paths:
            try:
                data = load_file(file)
                loopbuf.append(None, data)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("[loaded files] buffer size:", loopbuf.position_count, current_time)
            except Exception:
                continue
    if os.path.exists("log.npz"):
        try:
            log = log.load("log.npz")
            print("loaded log")
        except Exception as e:
            print("[Error]", e)
            os.remove("log.npz")  # reset
    counter = 0
    while True:
        log.start_batch()
        received_data = server.receive()
        raw_data = json.loads(received_data)
        received_data = json.loads(received_data)
        received_data = str(received_data)
        # print(f"[Received] {received_data}")
        if (
            "PythonTraining" in received_data
            or "RustDataGen" in received_data
            or "RequestingNet" in received_data
        ):
            net_send = serialise_net(model)
            msg = make_msg_send(
                {"NewNetworkData": net_send},
            )
            server.send(msg)

        if "JobSendPath" in received_data:
            file_path = raw_data["purpose"]["JobSendPath"]
            print(file_path)
            with open("datafile.txt", "a") as f:
                f.write(file_path + "\n")
            data = load_file(file_path)
            loopbuf.append(log, data)
            log.finished_data()
            try:
                log.save("log.npz")
            except Exception:
                print("[Warning] failed to save log.npz")

            print("[loaded files] buffer size:", loopbuf.position_count)
            if loopbuf.position_count >= BUFFER_SIZE:
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

                num_steps_training = (
                    len(data.positions) / BATCH_SIZE
                ) * SAMPLING_RATIO  # calculate number of training steps to take
                if num_steps_training < 1:
                    print("[Warning] minimum training step is", num_steps_training)
                    num_steps_training = 1
                    print("[Warning] set training step to 1")
                num_steps_training = int(num_steps_training)
                model.train()
                print("training model!")
                print("num_steps_training:", num_steps_training)

                for gen in range(num_steps_training):
                    if gen != 0:
                        log.start_batch()
                    batch = train_sampler.next_batch()
                    train_settings.train_step(
                        batch, network=model, optimizer=op, logger=log
                    )

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
                    log.save("log.npz")
                except Exception:
                    print("[Warning] failed to save log.npz")

                train_sampler.close()
                test_sampler.close()
                last_gen_test_sampler.close()
                starting_gen += 1
                model_path = "nets/tz_" + str(starting_gen) + ".pt"
                print(model_path)
                model.eval()
                with torch.no_grad():
                    torch.jit.save(model, model_path)
                with open("traininglog.txt", "a") as f:
                    f.write(model_path + "\n")
                with open("datafile.txt", "w") as f:
                    f.write("")

                # send to rust server
                msg = make_msg_send(
                    {"NewNetworkPath": model_path},
                )
                server.send(msg)
                net_send = serialise_net(model)
                msg = make_msg_send(
                    {"NewNetworkData": net_send},
                )
                server.send(msg)

        if "StopServer" in received_data:
            server.close()
            print("Connection closed.")
            break

        if "JobSendData" in received_data:
            bin_data = raw_data["purpose"]["JobSendData"][0]
            off_data = raw_data["purpose"]["JobSendData"][1]
            meta_data = raw_data["purpose"]["JobSendData"][2]
            bin_data, off_data, meta_data = (
                bytes(dict(bin_data)["BinFile"]),
                bytes(dict(off_data)["OffFile"]),
                bytes(dict(meta_data)["MetaDataFile"]),
            )
            
            if not os.path.exists("./python_client_games"):
                os.makedirs("./python_client_games")
            else:
                pass
            
            path = "./python_client_games/temp_games_" + str(counter) + "_" + str(int(time.time()))
            with open(path + ".bin", "wb") as file:
                file.write(bin_data)

            with open(path + ".off", "wb") as file:
                file.write(off_data)

            decoded_string = meta_data.decode("utf-8")
            data = json.loads(decoded_string)
            with open(path + ".json", "w") as file:
                json.dump(
                    data, file, indent=4
                )  # Use indent parameter for pretty formatting (optional)
            with open("datafile.txt", "a") as f:
                f.write(path + "\n")
            data = load_file(path)
            loopbuf.append(log, data)
            print("[loaded files] buffer size:", loopbuf.position_count)
            log.finished_data()
            try:
                log.save("log.npz")
            except Exception:
                print("[Warning] failed to save log.npz")
            counter += 1
            if loopbuf.position_count >= BUFFER_SIZE:
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

                num_steps_training = (
                    len(data.positions) / BATCH_SIZE
                ) * SAMPLING_RATIO  # calculate number of training steps to take
                min_sampling = 15
                if num_steps_training < min_sampling:
                    print("[Warning] minimum training step is 1, current training step is:", num_steps_training)
                    num_steps_training = min_sampling
                    print("[Warning] set training step to 1")
                num_steps_training = int(num_steps_training)
                model.train()
                print("training model!")
                print("num_steps_training:", num_steps_training)

                for gen in range(num_steps_training):
                    if gen != 0:
                        log.start_batch()
                    batch = train_sampler.next_batch()
                    train_settings.train_step(
                        batch, network=model, optimizer=op, logger=log
                    )

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
                    log.save("log.npz")
                except Exception:
                    print("[Warning] failed to save log.npz")

                train_sampler.close()
                test_sampler.close()
                last_gen_test_sampler.close()
                starting_gen += 1
                model_path = "nets/tz_" + str(starting_gen) + ".pt"
                print(model_path)
                model.eval()
                with torch.no_grad():
                    torch.jit.save(model, model_path)
                with open("traininglog.txt", "a") as f:
                    f.write(model_path + "\n")
                with open("datafile.txt", "w") as f:
                    f.write("")

                # send to rust server
                msg = make_msg_send(
                    {"NewNetworkPath": model_path},
                )
                server.send(msg)

        if "StopServer" in received_data:
            server.close()
            print("Connection closed.")
            break

                


if __name__ == "__main__":
    main()
