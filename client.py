import io
import json
import os
import re
import socket
import time
from datetime import datetime

import torch
import torch.optim as optim

import network
from lib.data.file import DataFile
from lib.games import Game
from lib.logger import Logger
from lib.loop import LoopBuffer
from lib.train import ScalarTarget, TrainSettings


class Server:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.file = None

    def connect(self):
        while True:
            try:
                self.socket.connect((self.host, self.port))
                print("Connected to server!")
                self.file = self.socket.makefile("r")
                break
            except ConnectionRefusedError as e:
                print(f"Connection failed: {e}. Retrying...")
                continue

    def send(self, message):
        message = {"purpose": message}
        obj = json.dumps(message) + "\n"
        self.socket.sendall(obj.encode("utf-8"))

    def receive(self):
        assert self.file is not None
        return self.file.readline()

    def close(self):
        self.socket.close()


def serialise_net(model):
    buffer = io.BytesIO()
    torch.jit.save(model, buffer)
    return [byte for byte in buffer.getvalue()]


def load_file(games_path):
    game = Game.find("chess")
    return DataFile.open(game, games_path)


HOST = "127.0.0.1"
PORT = 38475
BUFFER_SIZE = 1000000
BATCH_SIZE = 2048
MIN_SAMPLING = 10
SAMPLING_RATIO = 0.25

assert BATCH_SIZE > 0 and (BATCH_SIZE & (BATCH_SIZE - 1)) == 0


def main():
    game = Game.find("chess")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    os.makedirs("nets", exist_ok=True)
    os.makedirs("games", exist_ok=True)

    training_nets = check_net_exists(device, r"tz_(\d+)\.pt")
    model_path = get_model_path(training_nets)
    model = torch.jit.load(model_path, map_location=device).eval()
    starting_gen = int(re.findall(r"tz_(\d+)\.pt", model_path)[0])
    server = Server(HOST, PORT)
    server.connect()
    data_paths = get_previous_data_paths()
    get_verification(server, "PythonTraining")
    loopbuf = LoopBuffer(game, target_positions=BUFFER_SIZE, test_fraction=0.2)
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
    op = optim.AdamW(params=model.parameters(), lr=1e-3)
    log = load_previous_data(data_paths, loopbuf)

    while True:
        log.start_batch()
        received_data = server.receive()
        raw_data = json.loads(received_data)
        received_data = str(raw_data)

        if "RequestingNet" in received_data:
            send_net_in_bytes(model, server)

        if "JobSendPath" in received_data:
            data = extract_incoming_data_given_path(loopbuf, log, raw_data)
            if loopbuf.position_count >= BUFFER_SIZE:
                full_train_and_send(
                    model, starting_gen, server, loopbuf, train_settings, op, log, data
                )

        if "JobSendData" in received_data:
            data = extract_incoming_data_given_bytes(loopbuf, log, raw_data)
            if loopbuf.position_count >= BUFFER_SIZE:
                full_train_and_send(
                    model, starting_gen, server, loopbuf, train_settings, op, log, data
                )

        if "StopServer" in received_data:
            server.close()
            print("Connection closed.")
            break


def extract_incoming_data_given_bytes(loopbuf, log, raw_data):
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

    path = f"./python_client_games/temp_games_{int(time.time())}"
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
    print(path)
    data = load_file(path)
    loopbuf.append(log, data)
    print("[loaded files] buffer size:", loopbuf.position_count)
    log.finished_data()
    try:
        log.save("log.npz")
    except Exception:
        print("[Warning] failed to save log.npz")
    return data


def full_train_and_send(
    model, starting_gen, server, loopbuf, train_settings, op, log, data
):
    print("[loaded files] buffer size:", loopbuf.position_count)
    if loopbuf.position_count >= BUFFER_SIZE:
        train_sampler, test_sampler, last_gen_test_sampler = initialise_samplers(
            loopbuf
        )
        num_steps_training = get_num_steps_training(data, MIN_SAMPLING)
        model.train()
        print("training model!")
        print("num_steps_training:", num_steps_training)
        train_net(model, train_settings, op, log, train_sampler, num_steps_training)
        test_net(model, train_settings, log, test_sampler, last_gen_test_sampler)
        log.finished_data()
        try:
            log.save("log.npz")
        except Exception:
            print("[Warning] failed to save log.npz")
        starting_gen += 1
        model_path = save_and_register_net(model, starting_gen)
        send_new_net(model_path, model, server)


def send_new_net(model_path, model, server):
    msg = {"NewNetworkPath": model_path}
    server.send(msg)
    net_send = serialise_net(model)
    msg = {"NewNetworkData": net_send}
    server.send(msg)


def save_and_register_net(model, starting_gen):
    model_path = f"nets/tz_{starting_gen}.pt"
    print(model_path)
    model.eval()
    with torch.no_grad():
        torch.jit.save(model, model_path)
    with open("traininglog.txt", "a") as f:
        f.write(model_path + "\n")
    if not os.path.exists("datafile.txt"):
        with open("datafile.txt", "w"):
            pass
    return model_path


def test_net(model, train_settings, log, test_sampler, last_gen_test_sampler):
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
    test_sampler.close()
    last_gen_test_sampler.close()


def train_net(model, train_settings, op, log, train_sampler, num_steps_training):
    for gen in range(num_steps_training):
        if gen != 0:
            log.start_batch()
        batch = train_sampler.next_batch()
        train_settings.train_step(batch, network=model, optimizer=op, logger=log)
    train_sampler.close()


def get_num_steps_training(data, min_sampling):
    num_steps_training = (len(data.positions) / BATCH_SIZE) * SAMPLING_RATIO
    if num_steps_training < min_sampling:
        print("[Warning] minimum training step is", min_sampling)
        num_steps_training = min_sampling
        print("[Warning] set training step to", min_sampling)
    return int(num_steps_training)


def initialise_samplers(loopbuf):
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
    return train_sampler, test_sampler, last_gen_test_sampler


def extract_incoming_data_given_path(loopbuf, log, raw_data):
    file_path = raw_data["purpose"]["JobSendPath"]
    with open("datafile.txt", "a") as f:
        f.write(file_path + "\n")
    data = load_file(file_path)
    loopbuf.append(log, data)
    print("[loaded files] buffer size:", loopbuf.position_count)
    log.finished_data()
    try:
        log.save("log.npz")
    except Exception:
        print("[Warning] failed to save log.npz")
    return data


def send_net_in_bytes(model, server):
    net_send = serialise_net(model)
    msg = {"NewNetworkData": net_send}
    server.send(msg)


def load_previous_data(data_paths, loopbuf):
    log = Logger()
    if data_paths:
        data_paths = list(dict.fromkeys(data_paths))
        for file in data_paths:
            try:
                data = load_file(file)
                loopbuf.append(None, data)
            except Exception:
                continue
    if os.path.exists("log.npz"):
        try:
            log = log.load("log.npz")
            print("loaded log")
        except Exception as e:
            print("[Error]", e)
            os.remove("log.npz")
    print("[loaded files] buffer size:", loopbuf.position_count)
    return log


def get_verification(server, identity):
    while True:
        server.send({"Initialise": identity})
        received_data = server.receive()
        received_data = json.loads(received_data)
        purpose = str(received_data)
        if "IdentityConfirmation" in purpose and identity in purpose:
            break
    print("identity verified")


def get_previous_data_paths():
    data_paths = None
    if os.path.isfile("datafile.txt"):
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
    else:
        with open("datafile.txt", "w+"):
            pass
    return data_paths


def get_model_path(training_nets):
    if os.path.isfile("traininglog.txt"):
        with open("traininglog.txt", "r") as f:
            recorded_sessions = f.readlines()
            recorded_sessions = [
                item.strip() for item in recorded_sessions if item != ""
            ]
        if recorded_sessions != training_nets:
            with open("traininglog.txt", "w") as f:
                f.write("\n".join(training_nets) + "\n")
            recorded_sessions = training_nets
    else:
        with open("traininglog.txt", "w") as f:
            f.write(training_nets[-1] + "\n")
        recorded_sessions = training_nets

    model_path = recorded_sessions[-1].strip()
    return model_path


def check_net_exists(device, pattern):
    training_nets = []
    net_id = {}
    for net in os.listdir("nets"):
        match = re.match(pattern, net)
        if match:
            group = int(match.groups()[0])
            net_id[f"./nets/{net}"] = group
            training_nets.append(net)

    net_id = dict(sorted(net_id.items(), key=lambda x: x[1]))
    training_nets = list(net_id.keys())

    if not os.listdir("nets") or not training_nets:
        with torch.no_grad():
            net = torch.jit.script(
                # network.TrueNet(
                #     num_resBlocks=8,
                #     num_hidden=64,
                #     head_channel_policy=8,
                #     head_channel_values=4,
                network.TrueNetXS(num_hidden=64).to(device)
            ).eval()
            torch.jit.save(net, "nets/tz_0.pt")

        with open("traininglog.txt", "w+") as f:
            f.write("nets/tz_0.pt\n")

        training_nets.append("nets/tz_0.pt")
    return training_nets


if __name__ == "__main__":
    main()
