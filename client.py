import socket
from enum import Enum
from lib.games import Game
from trainingloop import load_file, trainer_loop
from lib.loop import LoopBuffer
from lib.logger import Logger
import threading
from queue import Queue


class MessageSend(Enum):  # message from python to rust
    NEW_NETWORK = "newnet"
    STOP_SERVER = "stop"
    PYTHON_ID = "python-training"


class MessageRecv(Enum):  # message from rust to python
    NEW_NETWORK = "newnet"
    STOP_SERVER = "shutdown"
    RUST_ID = "rust-datagen"
    JOB = "new-training-data"
    NET_REQUEST = "requesting-net"


class Server:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self) -> None:
        while True:
            try:
                self.socket.connect((self.host, self.port))
                print("Connected to server!")
                break
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

if __name__ == "__main__":
    # Initialization
    server = Server(HOST, PORT)
    server.connect()

    server.send(MessageSend.PYTHON_ID)

    loopbuf = LoopBuffer(
        Game.find("chess"), target_positions=BUFFER_SIZE, test_fraction=0.2
    )

    communication_queue = Queue()

    thread = threading.Thread(target=trainer_loop, args=(communication_queue,))
    thread.start()

    while True:
        log = Logger()
        log.start_batch()
        received_data = server.receive()
        print(f"Received: {received_data}")

        if (
            MessageSend.PYTHON_ID in received_data
            or MessageRecv.RUST_ID in received_data
            or MessageRecv.NET_REQUEST in received_data
        ):
            server.send("newnet: chess_16x128_gen3634.pt")

        if MessageRecv.JOB in received_data:
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

        if received_data == MessageRecv.STOP_SERVER:
            server.close()
            print("Connection closed.")
            thread.join()
            break
