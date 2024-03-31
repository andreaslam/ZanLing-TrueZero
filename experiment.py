import subprocess
import json
import threading
import queue
import os
import requests
import client
import plot_tensorboard


def run_tensorboard(port_number):
    subprocess.Popen(["tensorboard", "--port=" + str(port_number), "--logdir", "runs", "--bind_all"])
    username = os.getlogin()
    remote_ip = requests.get("https://api.ipify.org").text
    access = f"ssh -L <TARGET PORT ON REMOTE DEVICE>:127.0.0.1:{port_number} {username}@{remote_ip}"
    print(f"TensorBoard is running. Access it at port: {port_number}")
    print(access)
    return remote_ip, access


def process_thread(input_data):
    return plot_tensorboard.plot(input_data)


if __name__ == "__main__":
    port_number_tensorboard = 6006
    HOST = "127.0.0.1"
    PORT = 38475
    remote_ip, access_str = run_tensorboard(port_number_tensorboard)
    server = client.Server(HOST, PORT)
    server.connect()

    while True:
        server.send(client.make_msg_send({"Initialise": "TBHost"}))
        received_data = server.receive()
        received_data = json.loads(received_data)
        purpose = str(received_data)
        if "IdentityConfirmation" in purpose and "TBHost" in purpose:
            break

    result_queue = queue.Queue()

    def update_tensorboard(file_path):
        result = process_thread(file_path)
        server.send(client.make_msg_send({"TBLink": [remote_ip, access_str]}))

    started_drawing = False

    while True:
        received_data = server.receive()
        raw_data = json.loads(received_data)
        received_data = str(received_data)

        if "TBHost" in received_data or "RequestingTBLink" in received_data:
            server.send(client.make_msg_send({"TBLink": (remote_ip, access_str)}))

        if "UpdateTB" in received_data:
            file_path = raw_data.get("purpose", {}).get("CreateTB", "")
            if result_queue.empty():
                server.send(client.make_msg_send({"TBLink": [remote_ip, access_str]}))
            else:
                if not started_drawing:
                    threading.Thread(
                        target=update_tensorboard, args=(file_path,)
                    ).start()
                    started_drawing = True
                else:
                    server.send(
                        client.make_msg_send({"TBLink": [remote_ip, access_str]})
                    )
                    started_drawing = False

        if "StopServer" in received_data:
            server.close()
            print("Connection closed.")
            break
