import hashlib
import io
import json
import os
import re
import socket
import time

import network as network
import torch
from lib.data.file import DataFile
from lib.games import Game
from lib.logger import Logger
from lib.loop import LoopBuffer
from lib.train import ScalarTarget, TrainSettings
from paths import data_dir, data_path
from torch import optim


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
    with torch.no_grad():
        model.eval()
        torch.jit.save(model, buffer)
    return buffer.getvalue()


def model_weights_checksum(model):
    checksum = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in sorted(model.state_dict().items()):
            tensor = tensor.detach().cpu().contiguous()
            checksum.update(name.encode("utf-8"))
            checksum.update(str(tensor.dtype).encode("utf-8"))
            checksum.update(str(tuple(tensor.shape)).encode("utf-8"))
            checksum.update(tensor.numpy().tobytes())
    return checksum.hexdigest()


def load_file(games_path):
    game = Game.find("chess")
    return DataFile.open(game, games_path)


HOST = "127.0.0.1"
PORT = 38475
BUFFER_SIZE = 1500000
BATCH_SIZE = 2048
# Don't start training until the replay window has at least this many positions, so the
# network is not fit to a tiny, highly-correlated slice of one generation.
MIN_BUFFER_TO_TRAIN = 250_000
# Target number of times each *new* position is trained on per generation. AlphaZero/LC0
# keep this near ~1; the old regime reached into the hundreds and overfit noisy targets.
EPOCHS_PER_GEN = 1.0
# Uniform full-buffer sampling makes the newest generation only ~1/60 of batches once
# the replay window is full. Mix in recent-generation batches so each generation has a
# visible training signal without discarding the older, decorrelating replay data.
RECENT_BATCH_FRACTION = 0.5
MIN_TRAINING_STEPS = 16

assert BATCH_SIZE > 0 and (BATCH_SIZE & (BATCH_SIZE - 1)) == 0

data_dir("hidden")
DATAFILE_PATH = data_path("hidden/datafile.txt")
TRAINING_LOG_PATH = data_path("hidden/traininglog.txt")
LOG_NPZ_PATH = data_path("log.npz")
LAST_SENT_NET_CHECKSUM = None


def create_optimizer(model):
    # reset AdamW state whenever a rejected candidate is discarded; stale momentum
    # would otherwise keep pushing the accepted net towards the failed candidate.
    return optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=1e-4)


def main():
    game = Game.find("chess")
    # Pick a single, well-defined device. The previous version could select "mps"
    # while batches were built on DEVICE ("cuda"/"cpu"), causing a device mismatch.
    if torch.cuda.is_available():
        device = "cuda"
    elif (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        device = "mps"
    else:
        device = "cpu"
    print(f"Using: {device}")
    data_dir("nets")
    data_dir("games")

    training_nets = check_net_exists(device, r"tz_(\d+)\.pt")
    model_path = get_model_path(training_nets)
    model = torch.jit.load(model_path, map_location=device).eval()
    starting_gen = int(re.findall(r"tz_(\d+)\.pt", model_path)[0])
    accepted_model_path = model_path
    accepted_gen = starting_gen
    pending_candidate_path = None
    pending_candidate_gen = None
    print(f"starting generation: {starting_gen}")
    server = Server(HOST, PORT)
    server.connect()
    data_paths = get_previous_data_paths()
    get_verification(server, "PythonTraining")
    loopbuf = LoopBuffer(game, target_positions=BUFFER_SIZE, test_fraction=0.2)
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
    op = create_optimizer(model)
    log = load_previous_data(data_paths, loopbuf)

    while True:
        log.start_batch()
        received_data = server.receive()
        raw_data = json.loads(received_data)
        received_data = str(raw_data)

        if "RequestingNet" in received_data:
            if pending_candidate_path is None:
                send_net_in_bytes(model, server)
            else:
                send_model_from_path(accepted_model_path, device, server)

        test_result = extract_test_result(raw_data)
        if test_result is not None:
            if pending_candidate_path is None or pending_candidate_gen is None:
                print("[Training] Ignoring SPRT result with no pending candidate")
            elif test_result.get("accept_new_net", False):
                accepted_gen = pending_candidate_gen
                accepted_model_path = promote_candidate_net(
                    pending_candidate_path, accepted_gen
                )
                print(f"[Training] Candidate accepted: {accepted_model_path}")
                pending_candidate_path = None
                pending_candidate_gen = None
            else:
                print(
                    "[Training] Candidate rejected; reverting weights and optimiser "
                    f"to accepted net {accepted_model_path}"
                )
                delete_file_if_exists(pending_candidate_path)
                model = torch.jit.load(accepted_model_path, map_location=device).eval()
                op = create_optimizer(model)
                pending_candidate_path = None
                pending_candidate_gen = None

        # Train once per newly-received generation. The old condition
        # `position_count >= BUFFER_SIZE` only fired during the initial fill and on
        # eviction wrap-arounds, so generations were trained an inconsistent number of
        # times (and sometimes not at all). We now trigger on every new generation once
        # the buffer has warmed up past MIN_BUFFER_TO_TRAIN.
        if "JobSendPath" in received_data:
            data = extract_incoming_data_given_path(loopbuf, log, raw_data)
            if pending_candidate_path is not None:
                print(
                    "[Training] Candidate pending SPRT; ingesting data but not training"
                )
            elif loopbuf.position_count >= MIN_BUFFER_TO_TRAIN:
                candidate_gen = accepted_gen + 1
                candidate_path = full_train_and_send(
                    model, candidate_gen, server, loopbuf, train_settings, op, log, data
                )
                if candidate_path is not None:
                    pending_candidate_path = candidate_path
                    pending_candidate_gen = candidate_gen

        if "JobSendData" in received_data:
            data = extract_incoming_data_given_bytes(loopbuf, log, raw_data)
            if pending_candidate_path is not None:
                print(
                    "[Training] Candidate pending SPRT; ingesting data but not training"
                )
            elif loopbuf.position_count >= MIN_BUFFER_TO_TRAIN:
                candidate_gen = accepted_gen + 1
                candidate_path = full_train_and_send(
                    model, candidate_gen, server, loopbuf, train_settings, op, log, data
                )
                if candidate_path is not None:
                    pending_candidate_path = candidate_path
                    pending_candidate_gen = candidate_gen

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

    data_dir("python_client_games")

    path = data_path(f"python_client_games/temp_games_{int(time.time())}")
    with open(path + ".bin", "wb") as file:
        file.write(bin_data)

    with open(path + ".off", "wb") as file:
        file.write(off_data)
    decoded_string = meta_data.decode("utf-8")
    data = json.loads(decoded_string)
    with open(path + ".json", "w") as file:
        json.dump(data, file, indent=4)
    with open(DATAFILE_PATH, "a") as f:
        f.write(path + "\n")
    print(path)
    data = load_file(path)
    loopbuf.append(log, data)
    print("[loaded files] buffer size:", loopbuf.position_count)
    log.finished_data()
    try:
        log.save(LOG_NPZ_PATH)
    except Exception:
        print("[Warning] failed to save log.npz")
    return data


def full_train_and_send(
    model, starting_gen, server, loopbuf, train_settings, op, log, data
):
    if loopbuf.position_count >= MIN_BUFFER_TO_TRAIN:
        train_sampler, recent_train_sampler, test_sampler, last_gen_test_sampler = (
            initialise_samplers(loopbuf)
        )
        num_steps_training = get_num_steps_training(data)
        model.train()
        print("training model!")
        print("num_steps_training:", num_steps_training)
        train_net(
            model,
            train_settings,
            op,
            log,
            train_sampler,
            recent_train_sampler,
            num_steps_training,
        )
        test_net(model, train_settings, log, test_sampler, last_gen_test_sampler)
        log.finished_data()
        try:
            log.save(LOG_NPZ_PATH)
        except Exception:
            print("[Warning] failed to save log.npz")
        model_path = save_candidate_net(model, starting_gen)
        if send_new_net(model_path, model, server):
            return model_path
        delete_file_if_exists(model_path)
    return None


def send_new_net(model_path, model, server):  # sends both new net and path
    if is_duplicate_net(model):
        print("[Training] Network weights unchanged; skipping duplicate net send")
        return False
    msg = {"NewNetworkPath": model_path}
    server.send(msg)
    return send_net_in_bytes(model, server, force=True)


def save_candidate_net(model, starting_gen):
    model_path = data_path(f"nets/tz_candidate_{starting_gen}.pt")
    print(model_path)
    model.eval()
    with torch.no_grad():
        torch.jit.save(model, model_path)
    if not os.path.exists(DATAFILE_PATH):
        with open(DATAFILE_PATH, "w"):
            pass
    return model_path


def promote_candidate_net(candidate_path, accepted_gen):
    accepted_path = data_path(f"nets/tz_{accepted_gen}.pt")
    if os.path.abspath(candidate_path) != os.path.abspath(accepted_path):
        os.replace(candidate_path, accepted_path)
    with open(TRAINING_LOG_PATH, "a") as f:
        f.write(accepted_path + "\n")
    return accepted_path


def delete_file_if_exists(path):
    if path and os.path.exists(path):
        os.remove(path)


def extract_test_result(raw_data):
    purpose = raw_data.get("purpose", {})
    if isinstance(purpose, dict) and "TestResult" in purpose:
        return purpose["TestResult"]
    return None


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


def train_net(
    model,
    train_settings,
    op,
    log,
    train_sampler,
    recent_train_sampler,
    num_steps_training,
):
    recent_interval = max(1, round(1 / RECENT_BATCH_FRACTION))
    for gen in range(num_steps_training):
        if gen != 0:
            log.start_batch()
        use_recent = RECENT_BATCH_FRACTION > 0 and gen % recent_interval == 0
        sampler = recent_train_sampler if use_recent else train_sampler
        batch = sampler.next_batch()
        train_settings.train_step(batch, network=model, optimizer=op, logger=log)
    train_sampler.close()
    recent_train_sampler.close()


def get_num_steps_training(data):
    # Train each new generation ~EPOCHS_PER_GEN times through the recent sampler, while
    # using the remaining batches for broad replay. This keeps learning focused on the
    # policy just generated without overfitting it for many epochs.
    new_positions = len(data.positions)
    recent_fraction = max(RECENT_BATCH_FRACTION, 1e-6)
    num_steps_training = (new_positions / BATCH_SIZE) * EPOCHS_PER_GEN / recent_fraction
    num_steps_training = max(num_steps_training, MIN_TRAINING_STEPS)
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
    recent_train_sampler = loopbuf.sampler(
        batch_size=BATCH_SIZE,
        unroll_steps=None,
        include_final=False,
        random_symmetries=False,
        only_last_gen=True,
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
    return train_sampler, recent_train_sampler, test_sampler, last_gen_test_sampler


def extract_incoming_data_given_path(loopbuf, log, raw_data):
    file_path = raw_data["purpose"]["JobSendPath"]
    with open(DATAFILE_PATH, "a") as f:
        f.write(file_path + "\n")
    data = load_file(file_path)
    loopbuf.append(log, data)
    print("[loaded files] buffer size:", loopbuf.position_count)
    log.finished_data()
    try:
        log.save(LOG_NPZ_PATH)
    except Exception:
        print("[Warning] failed to save log.npz")
    return data


def is_duplicate_net(model):
    return LAST_SENT_NET_CHECKSUM == model_weights_checksum(model)


def send_net_in_bytes(model, server, force=False):
    global LAST_SENT_NET_CHECKSUM
    checksum = model_weights_checksum(model)
    if not force and LAST_SENT_NET_CHECKSUM == checksum:
        print("[Training] Network weights unchanged; skipping duplicate net send")
        return False
    net_send = serialise_net(model)
    msg = {"NewNetworkData": [byte for byte in net_send]}
    server.send(msg)
    LAST_SENT_NET_CHECKSUM = checksum
    return True


def send_model_from_path(model_path, device, server):
    print(f"[Training] Sending accepted net while candidate is pending: {model_path}")
    accepted_model = torch.jit.load(model_path, map_location=device).eval()
    return send_net_in_bytes(accepted_model, server, force=True)


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
    if os.path.exists(LOG_NPZ_PATH):
        try:
            log = log.load(LOG_NPZ_PATH)
            print("loaded log")
        except Exception as e:
            print("[Error]", e)
            os.remove(LOG_NPZ_PATH)
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
    if os.path.isfile(DATAFILE_PATH):
        with open(DATAFILE_PATH, "r") as f:
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
        with open(DATAFILE_PATH, "w+"):
            pass
    return data_paths


def get_model_path(training_nets):
    if os.path.isfile(TRAINING_LOG_PATH):
        with open(TRAINING_LOG_PATH, "r") as f:
            recorded_sessions = f.readlines()
            recorded_sessions = [
                item.strip() for item in recorded_sessions if item != ""
            ]
        if recorded_sessions != training_nets:
            with open(TRAINING_LOG_PATH, "w") as f:
                f.write("\n".join(training_nets) + "\n")
            recorded_sessions = training_nets
    else:
        with open(TRAINING_LOG_PATH, "w") as f:
            f.write(training_nets[-1] + "\n")
        recorded_sessions = training_nets

    model_path = recorded_sessions[-1].strip()
    return model_path


def check_net_exists(device, pattern):
    training_nets = []
    net_id = {}
    for net in os.listdir(data_dir("nets")):
        match = re.match(pattern, net)
        if match:
            group = int(match.groups()[0])
            net_id[data_path(f"nets/{net}")] = group
            training_nets.append(net)

    net_id = dict(sorted(net_id.items(), key=lambda x: x[1]))
    training_nets = list(net_id.keys())

    if not os.listdir(data_dir("nets")) or not training_nets:
        with torch.no_grad():
            net = torch.jit.script(
                # network.TrueNet().to(device)
                network.TrueNetXS(num_hidden=64).to(device)
            ).eval()
            torch.jit.save(net, data_path("nets/tz_0.pt"))

        with open(TRAINING_LOG_PATH, "w+") as f:
            f.write(data_path("nets/tz_0.pt") + "\n")

        training_nets.append(data_path("nets/tz_0.pt"))
    return training_nets


if __name__ == "__main__":
    main()
