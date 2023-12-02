import socket
from enum import Enum
# import threading
# import queue
from trainingloop import train, load_file
import torch
from torchrl.data import ReplayBuffer, ListStorage 

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


HOST = "127.0.0.1"
PORT = 8080

server = Server(HOST, PORT)
server.connect()

# identification - this is python training
server.send("python-training")

# define file buffer

rb = ReplayBuffer(
    storage=ListStorage(max_size=1000),
    batch_size=5,
)
while True:
    received_data = server.receive()
    print(f"Received: {received_data}")
    if "new-training-data" in received_data:
        # append file buffer
        
        file_path = received_data.split()[1].strip()      
        load_file(file_path)

# Close the server connection outside the loop
    if received_data == "shutdown":
         
        server.close()
        print("Connection closed.")
    
    
