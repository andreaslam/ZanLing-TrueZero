import hashlib
import io
import json
import os
import queue
import re
import socket
import threading
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
                time.sleep(1)

    def send(self, message):
        message = {"purpose": message}
        obj = json.dumps(message) + "\n"

        try:
            self.socket.sendall(obj.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"[Server] Failed to send message: {e}")

    def receive(self):
        assert self.file is not None
        return self.file.readline()

    def close(self):
        try:
            if self.file is not None:
                self.file.close()
        except OSError:
            pass

        try:
            self.socket.close()
        except OSError:
            pass


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

# Do not train until the replay buffer has at least this many positions.
MIN_BUFFER_TO_TRAIN = 250_000

# Normal training target:
# approximately one pass over newly generated positions.
EPOCHS_PER_GEN = 1.0

# Fraction of normal training batches sampled from the newest generation.
RECENT_BATCH_FRACTION = 0.5

# Small files still need to produce a candidate.
# This is intentionally modest because each file is only ~2k-5k positions.
MIN_TRAINING_STEPS = 16

# IMPORTANT:
# Startup training is NOT treated as one epoch over the entire replay
# buffer. That would massively oversample the newest 2k-5k positions
# through the recent sampler.
#
# Instead, perform a small bootstrap session over the full replay buffer.
STARTUP_TRAINING_STEPS = 16

assert BATCH_SIZE > 0 and (BATCH_SIZE & (BATCH_SIZE - 1)) == 0

data_dir("hidden")

DATAFILE_PATH = data_path("hidden/datafile.txt")
TRAINING_LOG_PATH = data_path("hidden/traininglog.txt")
LOG_NPZ_PATH = data_path("log.npz")

LAST_SENT_NET_CHECKSUM = None


def create_optimizer(model):
    # Reset AdamW state whenever a rejected candidate is discarded.
    # Otherwise stale momentum could continue pushing the accepted net
    # toward the rejected candidate.
    return optim.AdamW(
        params=model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )


def receiver_loop(server, message_queue, receiver_stop_event):
    """
    Continuously receive messages from the server.

    Network reception is deliberately separated from training so that
    server.receive() cannot block the training scheduler.
    """
    while not receiver_stop_event.is_set():
        try:
            received_data = server.receive()

            # Empty string means the peer closed the connection.
            if received_data == "":
                print("[Receiver] Server connection closed.")
                receiver_stop_event.set()
                return

            message_queue.put(received_data)

        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            OSError,
        ) as e:
            print(f"[Receiver] Connection lost: {e}")
            receiver_stop_event.set()
            return

        except Exception as e:
            print(f"[Receiver] Unexpected receive error: {e}")
            receiver_stop_event.set()
            return


def get_message(message_queue, timeout=0.05):
    try:
        return message_queue.get(timeout=timeout)
    except queue.Empty:
        return None


def main():
    game = Game.find("chess")

    # Select exactly one device.
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

    training_nets = check_net_exists(
        device,
        r"tz_(\d+)\.pt",
    )

    model_path = get_model_path(training_nets)

    model = torch.jit.load(
        model_path,
        map_location=device,
    ).eval()

    starting_gen = int(
        re.findall(
            r"tz_(\d+)\.pt",
            model_path,
        )[0]
    )

    accepted_model_path = model_path
    accepted_gen = starting_gen

    pending_candidate_path = None
    pending_candidate_gen = None

    print(f"starting generation: {starting_gen}")

    server = Server(HOST, PORT)
    server.connect()

    # Verification happens synchronously before the receiver thread starts.
    get_verification(
        server,
        "PythonTraining",
    )

    data_paths = get_previous_data_paths()

    loopbuf = LoopBuffer(
        game,
        target_positions=BUFFER_SIZE,
        test_fraction=0.2,
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

    op = create_optimizer(model)

    log = load_previous_data(
        data_paths,
        loopbuf,
    )

    # ---------------------------------------------------------------
    # Startup bootstrap
    #
    # If the replay buffer is already warm, we train a SMALL number
    # of batches from the FULL replay buffer.
    #
    # We do NOT use the recent-generation sampler here.
    # ---------------------------------------------------------------

    startup_training_due = loopbuf.position_count >= MIN_BUFFER_TO_TRAIN

    if startup_training_due:
        print(
            "[Training] Replay buffer already contains "
            f"{loopbuf.position_count:,} positions."
        )

        print(
            "[Training] Buffer is warm; scheduling "
            f"{STARTUP_TRAINING_STEPS} startup training steps."
        )
    else:
        print(
            "[Training] Replay buffer below minimum "
            f"({loopbuf.position_count:,}/"
            f"{MIN_BUFFER_TO_TRAIN:,}); waiting for data."
        )

    # Normal training is triggered by NEW data.
    training_due = False

    # Receiver thread.
    message_queue = queue.Queue()

    receiver_stop_event = threading.Event()

    receiver_thread = threading.Thread(
        target=receiver_loop,
        args=(
            server,
            message_queue,
            receiver_stop_event,
        ),
        daemon=True,
        name="server-receiver",
    )

    receiver_thread.start()

    stop_requested = False

    while True:
        # ===========================================================
        # 1. Process incoming messages without blocking the trainer.
        # ===========================================================

        received_data = get_message(
            message_queue,
            timeout=0.05,
        )

        if received_data is not None:
            log.start_batch()

            try:
                raw_data = json.loads(received_data)
            except json.JSONDecodeError as e:
                print(f"[Server] Ignoring malformed JSON message: {e}")
                continue

            received_data_string = str(raw_data)

            # -------------------------------------------------------
            # Network request.
            # -------------------------------------------------------

            if "RequestingNet" in received_data_string:
                if pending_candidate_path is None:
                    send_net_in_bytes(
                        model,
                        server,
                    )
                else:
                    # While a candidate is undergoing SPRT, expose the
                    # accepted network, not the pending candidate.
                    send_model_from_path(
                        accepted_model_path,
                        device,
                        server,
                    )

            # -------------------------------------------------------
            # SPRT result.
            # -------------------------------------------------------

            test_result = extract_test_result(raw_data)

            if test_result is not None:
                if pending_candidate_path is None or pending_candidate_gen is None:
                    print("[Training] Ignoring SPRT result with no pending candidate")

                elif test_result.get(
                    "accept_new_net",
                    False,
                ):
                    accepted_gen = pending_candidate_gen

                    accepted_model_path = promote_candidate_net(
                        pending_candidate_path,
                        accepted_gen,
                    )

                    print(f"[Training] Candidate accepted: {accepted_model_path}")

                    pending_candidate_path = None
                    pending_candidate_gen = None

                else:
                    print(
                        "[Training] Candidate rejected; reverting "
                        "weights and optimiser to accepted net "
                        f"{accepted_model_path}"
                    )

                    delete_file_if_exists(pending_candidate_path)

                    model = torch.jit.load(
                        accepted_model_path,
                        map_location=device,
                    ).eval()

                    op = create_optimizer(model)

                    pending_candidate_path = None
                    pending_candidate_gen = None

            # -------------------------------------------------------
            # New data via path.
            # -------------------------------------------------------

            if "JobSendPath" in received_data_string:
                data = extract_incoming_data_given_path(
                    loopbuf,
                    log,
                    raw_data,
                )

                if loopbuf.position_count >= MIN_BUFFER_TO_TRAIN:
                    training_due = True

                    print("[Training] New generation received; training marked due.")
                else:
                    print(
                        "[Training] New data received, but replay "
                        "buffer is still below training minimum."
                    )

            # -------------------------------------------------------
            # New data via bytes.
            # -------------------------------------------------------

            if "JobSendData" in received_data_string:
                data = extract_incoming_data_given_bytes(
                    loopbuf,
                    log,
                    raw_data,
                )

                if loopbuf.position_count >= MIN_BUFFER_TO_TRAIN:
                    training_due = True

                    print("[Training] New generation received; training marked due.")
                else:
                    print(
                        "[Training] New data received, but replay "
                        "buffer is still below training minimum."
                    )

            # -------------------------------------------------------
            # Explicit shutdown.
            # -------------------------------------------------------

            if "StopServer" in received_data_string:
                print("[Server] Stop requested.")
                stop_requested = True

        # ===========================================================
        # 2. STARTUP BOOTSTRAP TRAINING
        #
        # This happens independently of the generator sending a file.
        # It uses the FULL replay buffer and only 16 steps.
        # ===========================================================

        if (
            startup_training_due
            and loopbuf.position_count >= MIN_BUFFER_TO_TRAIN
            and pending_candidate_path is None
        ):
            candidate_gen = accepted_gen + 1

            print(
                "[Training] Starting startup bootstrap training "
                f"for generation {candidate_gen}."
            )

            candidate_path = full_train_and_send(
                model=model,
                starting_gen=candidate_gen,
                server=server,
                loopbuf=loopbuf,
                train_settings=train_settings,
                op=op,
                log=log,
                data=None,
                startup=True,
            )

            startup_training_due = False

            if candidate_path is not None:
                pending_candidate_path = candidate_path
                pending_candidate_gen = candidate_gen

                print(
                    f"[Training] Startup candidate {candidate_gen} is now pending SPRT."
                )

        # ===========================================================
        # 3. NORMAL NEW-GENERATION TRAINING
        #
        # Only actual new data makes this due.
        # ===========================================================

        if (
            training_due
            and loopbuf.position_count >= MIN_BUFFER_TO_TRAIN
            and pending_candidate_path is None
        ):
            candidate_gen = accepted_gen + 1

            print(f"[Training] Starting training for generation {candidate_gen}.")

            candidate_path = full_train_and_send(
                model=model,
                starting_gen=candidate_gen,
                server=server,
                loopbuf=loopbuf,
                train_settings=train_settings,
                op=op,
                log=log,
                data=data if "data" in locals() else None,
                startup=False,
            )

            training_due = False

            if candidate_path is not None:
                pending_candidate_path = candidate_path
                pending_candidate_gen = candidate_gen

                print(f"[Training] Candidate {candidate_gen} is now pending SPRT.")

        # ===========================================================
        # 4. Shutdown.
        # ===========================================================

        if stop_requested:
            receiver_stop_event.set()
            server.close()

            print("[Server] Connection closed.")
            break

        # A disconnected generator should NOT stop training.
        if receiver_stop_event.is_set():
            if not getattr(
                main,
                "_receiver_disconnect_reported",
                False,
            ):
                print(
                    "[Training] Generator/server connection is currently unavailable."
                )

                print("[Training] Continuing from the persisted replay buffer.")

                main._receiver_disconnect_reported = True


def extract_incoming_data_given_bytes(
    loopbuf,
    log,
    raw_data,
):
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
        json.dump(
            data,
            file,
            indent=4,
        )

    with open(DATAFILE_PATH, "a") as f:
        f.write(path + "\n")

    print(path)

    data = load_file(path)

    loopbuf.append(
        log,
        data,
    )

    print(
        "[loaded files] buffer size:",
        loopbuf.position_count,
    )

    log.finished_data()

    try:
        log.save(LOG_NPZ_PATH)
    except Exception:
        print("[Warning] failed to save log.npz")

    return data


def full_train_and_send(
    model,
    starting_gen,
    server,
    loopbuf,
    train_settings,
    op,
    log,
    data=None,
    startup=False,
):
    if loopbuf.position_count < MIN_BUFFER_TO_TRAIN:
        return None

    if startup:
        (
            train_sampler,
            test_sampler,
            last_gen_test_sampler,
        ) = initialise_startup_samplers(loopbuf)

        num_steps_training = STARTUP_TRAINING_STEPS

        model.train()

        print(f"[Training] startup bootstrap: {num_steps_training} full-replay steps")

        train_startup(
            model=model,
            train_settings=train_settings,
            op=op,
            log=log,
            train_sampler=train_sampler,
            num_steps_training=num_steps_training,
        )

    else:
        (
            train_sampler,
            recent_train_sampler,
            test_sampler,
            last_gen_test_sampler,
        ) = initialise_samplers(loopbuf)

        if data is None:
            print(
                "[Training] WARNING: normal training requested "
                "without a data object; using minimum training steps."
            )

            num_steps_training = MIN_TRAINING_STEPS
        else:
            num_steps_training = get_num_steps_training(data)

        model.train()

        print("[Training] training model!")
        print(
            "[Training] new positions:",
            len(data.positions) if data is not None else 0,
        )
        print(
            "[Training] num_steps_training:",
            num_steps_training,
        )

        train_net(
            model=model,
            train_settings=train_settings,
            op=op,
            log=log,
            train_sampler=train_sampler,
            recent_train_sampler=recent_train_sampler,
            num_steps_training=num_steps_training,
        )

    test_net(
        model,
        train_settings,
        log,
        test_sampler,
        last_gen_test_sampler,
    )

    log.finished_data()

    try:
        log.save(LOG_NPZ_PATH)
    except Exception:
        print("[Warning] failed to save log.npz")

    model_path = save_candidate_net(
        model,
        starting_gen,
    )

    if send_new_net(
        model_path,
        model,
        server,
    ):
        return model_path

    delete_file_if_exists(model_path)

    return None


def train_startup(
    model,
    train_settings,
    op,
    log,
    train_sampler,
    num_steps_training,
):
    """
    Startup bootstrap training.

    IMPORTANT:
    This sampler is the full replay sampler, NOT the recent-generation
    sampler. Therefore the 16 startup batches are distributed across the
    replay buffer rather than hammering the final 2k-5k position file.
    """
    for step in range(num_steps_training):
        if step != 0:
            log.start_batch()

        batch = train_sampler.next_batch()

        train_settings.train_step(
            batch,
            network=model,
            optimizer=op,
            logger=log,
        )

    train_sampler.close()


def send_new_net(
    model_path,
    model,
    server,
):
    if is_duplicate_net(model):
        print("[Training] Network weights unchanged; skipping duplicate net send")

        return False

    msg = {"NewNetworkPath": model_path}

    server.send(msg)

    return send_net_in_bytes(
        model,
        server,
        force=True,
    )


def save_candidate_net(
    model,
    starting_gen,
):
    model_path = data_path(f"nets/tz_candidate_{starting_gen}.pt")

    print(model_path)

    model.eval()

    with torch.no_grad():
        torch.jit.save(
            model,
            model_path,
        )

    if not os.path.exists(DATAFILE_PATH):
        with open(DATAFILE_PATH, "w"):
            pass

    return model_path


def promote_candidate_net(
    candidate_path,
    accepted_gen,
):
    accepted_path = data_path(f"nets/tz_{accepted_gen}.pt")

    if os.path.abspath(candidate_path) != os.path.abspath(accepted_path):
        os.replace(
            candidate_path,
            accepted_path,
        )

    with open(
        TRAINING_LOG_PATH,
        "a",
    ) as f:
        f.write(accepted_path + "\n")

    return accepted_path


def delete_file_if_exists(path):
    if path and os.path.exists(path):
        os.remove(path)


def extract_test_result(raw_data):
    purpose = raw_data.get(
        "purpose",
        {},
    )

    if isinstance(purpose, dict) and "TestResult" in purpose:
        return purpose["TestResult"]

    return None


def test_net(
    model,
    train_settings,
    log,
    test_sampler,
    last_gen_test_sampler,
):
    with torch.no_grad():
        model.eval()

        test_batch = test_sampler.next_batch()

        train_settings.evaluate_batch(
            network=model,
            batch=test_batch,
            log_prefix="test",
            logger=log,
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
    recent_interval = max(
        1,
        round(1 / RECENT_BATCH_FRACTION),
    )

    for gen in range(num_steps_training):
        if gen != 0:
            log.start_batch()

        use_recent = RECENT_BATCH_FRACTION > 0 and gen % recent_interval == 0

        sampler = recent_train_sampler if use_recent else train_sampler

        batch = sampler.next_batch()

        train_settings.train_step(
            batch,
            network=model,
            optimizer=op,
            logger=log,
        )

    train_sampler.close()
    recent_train_sampler.close()


def get_num_steps_training(data):
    """
    Normal generation training.

    A 2k-5k position file therefore produces approximately:

        2k -> max(16, ~2) = 16 steps
        5k -> max(16, ~5) = 16 steps

    With 50% recent sampling, the newest generation is therefore seen
    roughly 8 batches = 16,384 sampled positions at minimum.

    This is substantially less aggressive than the previous startup
    behaviour, where a 328k-position buffer generated ~321 steps with
    ~160 recent batches.
    """
    new_positions = len(data.positions)

    recent_fraction = max(
        RECENT_BATCH_FRACTION,
        1e-6,
    )

    num_steps_training = (new_positions / BATCH_SIZE) * EPOCHS_PER_GEN / recent_fraction

    num_steps_training = max(
        num_steps_training,
        MIN_TRAINING_STEPS,
    )

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

    return (
        train_sampler,
        recent_train_sampler,
        test_sampler,
        last_gen_test_sampler,
    )


def initialise_startup_samplers(loopbuf):
    """
    Startup-specific samplers.

    The training sampler covers the FULL replay buffer.

    There is deliberately no recent training sampler here. Startup
    training is a bootstrap operation, not a full training generation.
    """
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

    return (
        train_sampler,
        test_sampler,
        last_gen_test_sampler,
    )


def extract_incoming_data_given_path(
    loopbuf,
    log,
    raw_data,
):
    file_path = raw_data["purpose"]["JobSendPath"]

    with open(
        DATAFILE_PATH,
        "a",
    ) as f:
        f.write(file_path + "\n")

    data = load_file(file_path)

    loopbuf.append(
        log,
        data,
    )

    print(
        "[loaded files] buffer size:",
        loopbuf.position_count,
    )

    log.finished_data()

    try:
        log.save(LOG_NPZ_PATH)
    except Exception:
        print("[Warning] failed to save log.npz")

    return data


def is_duplicate_net(model):
    return LAST_SENT_NET_CHECKSUM == model_weights_checksum(model)


def send_net_in_bytes(
    model,
    server,
    force=False,
):
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


def send_model_from_path(
    model_path,
    device,
    server,
):
    print(f"[Training] Sending accepted net while candidate is pending: {model_path}")

    accepted_model = torch.jit.load(
        model_path,
        map_location=device,
    ).eval()

    return send_net_in_bytes(
        accepted_model,
        server,
        force=True,
    )


def load_previous_data(
    data_paths,
    loopbuf,
):
    log = Logger()

    if data_paths:
        data_paths = list(dict.fromkeys(data_paths))

        for file in data_paths:
            try:
                data = load_file(file)

                loopbuf.append(
                    None,
                    data,
                )

            except Exception:
                continue

    if os.path.exists(LOG_NPZ_PATH):
        try:
            log = log.load(LOG_NPZ_PATH)

            print("loaded log")

        except Exception as e:
            print("[Error]", e)

            os.remove(LOG_NPZ_PATH)

    print(
        "[loaded files] buffer size:",
        loopbuf.position_count,
    )

    return log


def get_verification(
    server,
    identity,
):
    while True:
        server.send({"Initialise": identity})

        received_data = server.receive()

        if received_data == "":
            raise ConnectionError("Server disconnected during verification.")

        received_data = json.loads(received_data)

        purpose = str(received_data)

        if "IdentityConfirmation" in purpose and identity in purpose:
            break

    print("identity verified")


def get_previous_data_paths():
    data_paths = None

    if os.path.isfile(DATAFILE_PATH):
        with open(
            DATAFILE_PATH,
            "r",
        ) as f:
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
        with open(
            DATAFILE_PATH,
            "w+",
        ):
            pass

    return data_paths


def get_model_path(training_nets):
    if os.path.isfile(TRAINING_LOG_PATH):
        with open(
            TRAINING_LOG_PATH,
            "r",
        ) as f:
            recorded_sessions = f.readlines()

        recorded_sessions = [item.strip() for item in recorded_sessions if item != ""]

        if recorded_sessions != training_nets:
            with open(
                TRAINING_LOG_PATH,
                "w",
            ) as f:
                f.write("\n".join(training_nets) + "\n")

            recorded_sessions = training_nets

    else:
        with open(
            TRAINING_LOG_PATH,
            "w",
        ) as f:
            f.write(training_nets[-1] + "\n")

        recorded_sessions = training_nets

    model_path = recorded_sessions[-1].strip()

    return model_path


def check_net_exists(
    device,
    pattern,
):
    training_nets = []
    net_id = {}

    for net in os.listdir(data_dir("nets")):
        match = re.match(
            pattern,
            net,
        )

        if match:
            group = int(match.groups()[0])

            net_id[data_path(f"nets/{net}")] = group

            training_nets.append(net)

    net_id = dict(
        sorted(
            net_id.items(),
            key=lambda x: x[1],
        )
    )

    training_nets = list(net_id.keys())

    if not os.listdir(data_dir("nets")) or not training_nets:
        with torch.no_grad():
            net = torch.jit.script(network.TrueNetXS(num_hidden=64).to(device)).eval()

            torch.jit.save(
                net,
                data_path("nets/tz_0.pt"),
            )

        with open(
            TRAINING_LOG_PATH,
            "w+",
        ) as f:
            f.write(data_path("nets/tz_0.pt") + "\n")

        training_nets.append(data_path("nets/tz_0.pt"))

    return training_nets


if __name__ == "__main__":
    main()
