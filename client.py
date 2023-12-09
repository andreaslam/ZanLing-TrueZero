import socket
from enum import Enum
from lib.games import Game
from trainingloop import load_file, trainer_loop
from lib.loop import LoopBuffer
from lib.logger import Logger
import threading
from queue import Queue


class MessageSend(Enum):
    NEW_NETWORK = "newnet"
    STOP_SERVER = "stop"


class Server:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self) -> None:
        connected = False
        while not connected:
            try:
                self.socket.connect((self.host, self.port))
                print("Connected to server!")
                connected = True
            except ConnectionRefusedError as e:
                print(f"Connection failed: {e}. Retrying...")
                continue

    def send(self, message: str) -> None:
        self.socket.sendall(message.encode())

    def receive(self) -> str:
        data_received = self.socket.recv(16384)
        return data_received.decode()

    def close(self) -> None:
        self.socket.close()


# Constants
HOST = "127.0.0.1"
PORT = 8080
BUFFER_SIZE = 1000
BATCH_SIZE = 200

# Initialization
server = Server(HOST, PORT)
server.connect()

server.send("python-training")

loopbuf = LoopBuffer(
    Game.find("chess"), target_positions=BUFFER_SIZE, test_fraction=0.2
)
log = Logger()
log.start_batch()

logged_in = False
communication_queue = Queue()

thread = threading.Thread(target=trainer_loop, args=(communication_queue,))
thread.start()

while True:
    received_data = server.receive()
    print(f"Received: {received_data}")

    if (
        "python-training" in received_data
        or "rust-datagen" in received_data
        or "requesting-net" in received_data
    ):
        server.send("newnet: chess_16x128_gen3634.pt")

    if "new-training-data" in received_data and logged_in:
        file_path = received_data.split()[1].strip()
        data = load_file(file_path)
        loopbuf.append(log, data)

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
            batch = sample.next_batch()
            communication_queue.put(batch)

    if received_data == "shutdown":
        server.close()
        print("Connection closed.")
        thread.join()
        break
