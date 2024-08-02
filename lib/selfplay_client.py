import dataclasses
import json
import os
import socket
import time
from dataclasses import dataclass
from typing import Union, Optional


@dataclass
class StartupSettings:
    game: str
    muzero: bool
    start_pos: str

    first_gen: int
    output_folder: str
    games_per_gen: int

    cpu_threads_per_device: int
    gpu_threads_per_device: int
    gpu_batch_size: int
    gpu_batch_size_root: int
    search_batch_size: int

    saved_state_channels: int
    eval_random_symmetries: bool

    def as_dict(self):
        return dataclasses.asdict(self)


@dataclass
class UctWeights:
    exploration_weight: Optional[float]
    moves_left_weight: Optional[float]
    moves_left_clip: Optional[float]
    moves_left_sharpness: Optional[float]

    @staticmethod
    def default():
        return UctWeights(
            exploration_weight=None,
            moves_left_weight=None,
            moves_left_clip=None,
            moves_left_sharpness=None,
        )

    def as_dict(self):
        return dataclasses.asdict(self)


@dataclass
class SelfplaySettings:
    max_game_length: int
    weights: UctWeights
    q_mode: str
    temperature: float
    zero_temp_move_count: int
    dirichlet_alpha: float
    dirichlet_eps: float
    search_policy_temperature_root: float
    search_policy_temperature_child: float
    search_fpu_root: str
    search_fpu_child: str
    search_virtual_loss_weight: float
    full_search_prob: float
    full_iterations: int
    part_iterations: int
    top_moves: int
    cache_size: int

    def as_dict(self):
        return dataclasses.asdict(self)


CONNECT_TRY_PERIOD = 1.0


def connect_to_selfplay_server(port: int) -> socket.socket:
    while True:
        last_attempt_start = time.time()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", port))
            return s
        except ConnectionRefusedError as e:
            print(e, f"on port {port}")

        delay = CONNECT_TRY_PERIOD - (time.time() - last_attempt_start)
        if delay > 0:
            time.sleep(delay)


class SelfplayClient:
    def __init__(self, port: int):
        self.s = connect_to_selfplay_server(port)
        self.f = self.s.makefile("r")

    def send(self, message: Union[dict, str]):
        s = json.dumps(message)
        print(f"Sending '{s}'")
        self.s.send((s + "\n").encode())

    def send_startup_settings(self, settings: StartupSettings):
        self.send({"StartupSettings": settings.as_dict()})

    def send_new_settings(self, settings: SelfplaySettings):
        self.send({"NewSettings": settings.as_dict()})

    def send_wait_for_new_network(self):
        self.send("WaitForNewNetwork")

    def send_dummy_network(self):
        self.send("UseDummyNetwork")

    def send_new_network(self, path: str):
        path = os.path.abspath(path)
        self.send({"NewNetwork": path})

    def send_stop(self):
        self.send("Stop")

    def wait_for_file(self) -> int:
        line = self.f.readline()
        if not line.endswith("\n"):
            raise IOError("Connection closed")

        message = json.loads(line)
        if message == "Stopped":
            raise RuntimeError("Selfplay server stopped")

        print(f"Received message {message}")
        return message["FinishedFile"]["index"]
