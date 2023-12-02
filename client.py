import socket
from enum import Enum
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
        connected = False
        while not connected:
            try:
                self.socket.connect((self.host, self.port))
                print("Connected to server...")
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

shared_queue = queue.Queue()

ml_loop_thread = threading.Thread(target=loop, args =(shared_queue,))  # Comma added to create a single-item tuple

HOST = "127.0.0.1"
PORT = 8080

server = Server(HOST, PORT)
server.connect()

# identification - this is python training
server.send("python-training")

while True:
    received_data = server.receive()
    print(f"Received: {received_data}")
    if "new-training-data" in received_data:
    

# Close the server connection outside the loop
server.close()
print("Connection closed.")
