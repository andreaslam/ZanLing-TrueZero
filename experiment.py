import subprocess
import json
import threading
import queue
import os
import requests
import client
import plot_tensorboard


def run_tensorboard(port_number):
    try:
        subprocess.Popen(
            [
                "tensorboard",
                "--port=" + str(port_number),
                "--logdir",
                "runs",
                "--bind_all",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            shell=True,
        )
        username = os.getlogin()
        remote_ip = requests.get("https://api.ipify.org").text
        access = f"ssh -L <TARGET PORT ON REMOTE DEVICE>:127.0.0.1:{port_number} {username}@{remote_ip}"
        print(f"TensorBoard is running. Access it at port: {port_number}")
        print(access)
        return remote_ip, access
    except Exception as e:
        print("Error starting TensorBoard:", e)
        return None, None


def process_thread(input_data):
    return plot_tensorboard.plot_loop(input_data)


if __name__ == "__main__":
    port_number_tensorboard = 6006
    HOST = "127.0.0.1"
    PORT = 38475
    remote_ip, access_str = run_tensorboard(port_number_tensorboard)
    if remote_ip is None or access_str is None:
        exit(1)

    server = client.Server(HOST, PORT)
    server.connect()

    result_queue = queue.Queue()

    def update_tensorboard(file_path):
        try:
            server.send({"TBLink": [remote_ip, access_str]})
        except Exception as e:
            print("Error processing tensorboard data:", e)

    started_drawing = False

    while True:
        received_data = server.receive()
        if not received_data:
            continue

        try:
            raw_data = json.loads(received_data)
        except json.JSONDecodeError:
            print("Received invalid JSON data:", received_data)
            continue

        received_data = str(received_data)

        if "TBHost" in received_data or "RequestingTBLink" in received_data:
            file_path = "logs.npz"
            if result_queue.empty():
                server.send({"TBLink": [remote_ip, access_str]})
            else:
                if not started_drawing:
                    threading.Thread(
                        target=update_tensorboard, args=(file_path,)
                    ).start()
                    started_drawing = True
                else:
                    server.send({"TBLink": [remote_ip, access_str]})
                    started_drawing = False

        if "StopServer" in received_data:
            server.close()
            print("Connection closed.")
            break
