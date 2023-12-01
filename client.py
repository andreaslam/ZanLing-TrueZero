import socket
from enum import Enum
import time
import threading
import queue
from trainingloop import loop

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
        self.socket.connect((self.host, self.port))
        print("Connected to server...")

    def send(self, message):
        self.socket.sendall(message.encode())

    def receive(self):
        data = self.socket.recv(1024)
        return data.decode()

    def close(self):
        self.socket.close()


shared_queue = queue.Queue()

ml_loop_thread = threading.Thread(target=loop, args=(shared_queue))


HOST = "127.0.0.1"
PORT = 8080

server = Server(HOST, PORT)
server.connect()

# identification - this is python training

server.send("python-training")

while True:

    received_data = server.receive()
    print(f"Received: {received_data}")

    server.close()
    print("Connection closed.")
