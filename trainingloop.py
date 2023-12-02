import socket
from enum import Enum
from lib.games import Game
from trainingloop import train, load_file
from lib.loop import LoopBuffer
from lib.logger import Logger

# message enums
class MessageSend(Enum):  # message to send to rust
    NEW_NETWORK = "newnet"
    STOP_SERVER = "stop"


class Server:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        connected = False
        while not connected:
            try:
                self.socket.connect((self.host, self.port))
                print("Connected to server!.")
                connected = True
            except ConnectionRefusedError as e:
                print(f"Connection failed: {e}. Retrying...")
                continue

    def send(self, message):
        self.socket.sendall(message.encode())

    def receive(self):
        data = self.socket.recv(16384)
        return data.decode()

    def close(self):
        self.socket.close()


HOST = "127.0.0.1"
PORT = 8080

server = Server(HOST, PORT)
server.connect()

# identification - this is python training
server.send("python-training")

# define file buffer


logged_in = False


BUFFER_SIZE = 1000
BATCH_SIZE = 200
loopbuf = LoopBuffer(
    Game.find("chess"), target_positions=BUFFER_SIZE, test_fraction=0.2
)
while True:
    log = Logger()
    log.start_batch()
    received_data = server.receive()
    print(f"Received: {received_data}")

    if "python-training" in received_data:  # connected to server, got confirmation
        logged_in = True
        print("logged in")

    if "new-training-data" in received_data and logged_in:
        # append file buffer

        file_path = received_data.split()[1].strip()
        data = load_file(file_path)
        loopbuf.append(log, data)

        if loopbuf.position_count >= 1000:
            sample = loopbuf.sampler(
                batch_size=BATCH_SIZE,
                unroll_steps=None,
                include_final=False,
                random_symmetries=False,
                only_last_gen=False,
                test=True,
            )
            batch = sample.next_batch()
            train(batch)
            loopbuf = LoopBuffer(
                Game.find("chess"), target_positions=BUFFER_SIZE, test_fraction=0.2
            )

    # Close the server connection outside the loop
    if received_data == "shutdown" and logged_in:
        server.close()
        print("Connection closed.")
