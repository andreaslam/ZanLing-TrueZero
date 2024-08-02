import logging
import re
import socket
from enum import Enum
from queue import Queue
import sys
import threading
import json
import os
import time
import torch
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QThread, Signal
import paramiko
from gui import LoginScreen, ExperimentTracker, DataReceiver
import network

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename="truescheduler.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class LiaseThread(QThread):
    data_received = Signal(str)

    def __init__(self, manager_gui_queue, server_manager_queue):
        super().__init__()
        self.manager_gui_queue = manager_gui_queue
        self.ssh_client = paramiko.SSHClient()
        self.server_manager_queue = server_manager_queue

    def run(self):
        init_msg = None
        logger.info("[LiaseThread] spawning run() from data_receiver")
        while True:  # Login first to SSH
            output = self.manager_gui_queue.get()
            if output == init_msg:
                self.manager_gui_queue.put(output)
            if output != init_msg and isinstance(output, dict):
                init_msg = output
                try:
                    # SSH connection setup
                    self.ssh_client.set_missing_host_key_policy(
                        paramiko.AutoAddPolicy()
                    )
                    password, hostname, ip, port, local_port, port_forwarding = list(
                        init_msg.values()
                    )
                    self.ssh_client.connect(
                        ip, port=port, username=hostname, password=password
                    )
                    logger.info("[LiaseThread] SSH login successful")
                    init_msg = None  # too early to send a success msg but not sure if failed yet
                    if port_forwarding:
                        ssh_transport = self.ssh_client.get_transport()
                        ssh_transport.request_port_forward("", local_port)
                        logger.info("[LiaseThread] SSH port forwarding successful")
                    break
                except Exception as e:
                    self.manager_gui_queue.put("failed")
                    logger.error(f"[LiaseThread] sending error back to GUI: '{e}'")
                    init_msg = "failed"

        # Send login to ServerListener
        init_msg = None
        msg = {"ip": ip, "port": port}
        logger.info(f"[LiaseThread] sending: '{msg}' to ServerListenerThread")
        self.server_manager_queue.put(msg)
        init_msg = msg
        while True:
            output = self.server_manager_queue.get()
            if output != init_msg:
                print(f"output {output}, init_msg {init_msg}")
                logger.debug(
                    f"[Liase thread] sending: '{output}' to GUI. Forwarded from ServerListenerThread"
                )
                self.manager_gui_queue.put(output)
                break
            else:
                self.server_manager_queue.put(output)

        # # receive experiment requests from gui.py
        # while True:
        #     logger.info(
        #         f"[LiaseThread] sending {output} to ServerListenerThread. Forwarded from GUI"
        #     )
        #     output = self.manager_gui_queue.get()
        #     self.server_manager_queue.put(output)


# ServerListenerThread class
class ServerListenerThread(QThread):
    data_received = Signal(str)

    def __init__(self, server_manager_queue):
        super().__init__()
        self.server_manager_queue = server_manager_queue
        self.ssh_client = paramiko.SSHClient()

    def run(self):
        init_msg = None
        logger.info("[ServerListenerThread] importing client...")
        import client

        logger.info("[ServerListenerThread] imported client!")
        while True:  # Wait for SSH login from LiaseThread
            output = self.server_manager_queue.get()
            logger.info(f"[ServerListenerThread] output (raw) {output}")
            if init_msg:
                self.server_manager_queue.put(init_msg)
            if output != init_msg and isinstance(output, dict):
                logger.info(
                    f"[ServerListenerThread] received {output} from LiaseThread"
                )
                init_msg = output
                host, port = list(output.values())
                server = client.Server(host, int(port))
                server.connect()
                logger.info("[ServerListenerThread] server connected")
                init_msg = "ok"
                self.server_manager_queue.put(init_msg)
                logger.info(f"[ServerListenerThread] sending {init_msg} to LiaseThread")
                break
        while True:
            server.send({"Initialise": "GUIMonitor"})
            while True:
                received_data = server.receive()
                if isinstance(received_data, dict):
                    break
            received_data = json.loads(received_data)
            purpose = str(received_data)
            if "IdentityConfirmation" in purpose and "GUIMonitor" in purpose:
                break

        # while True:
        #     received_data = server.receive()
        #     raw_data = json.loads(received_data)
        #     received_data = json.loads(received_data)
        #     received_data = str(received_data)

        #     self.server_manager_queue.put((raw_data, received_data))

        #     # TODO: implement listening to self.server_manager_queue
        #     if (
        #         "ExperimentDone" in received_data
        #     ):  # returns which experiment was finished
        #         finished_experiment = raw_data["purpose"]["ExperimentDone"]
        #         try:
        #             if os.path.exists("experiments.json"):
        #                 # If it exists, load the JSON data into a Python dictionary
        #                 with open("experiments.json", "r") as file:
        #                     prevs_scheduler = json.load(file)
        #                     temp_dict = prevs_scheduler
        #                     prev_experiments = list(prevs_scheduler.values())
        #                     for experiment in prev_experiments:
        #                         if finished_experiment in experiment:
        #                             del temp_dict[experiment]
        #                     prevs_scheduler = temp_dict

        #                 with open("experiments.json", "w") as json_file:
        #                     json.dump(prevs_scheduler, json_file)
        #             else:
        #                 # If it doesn't exist, create an empty dictionary
        #                 with open("experiments.json", "w") as file:
        #                     pass
        #                 prevs_scheduler = {}
        #         except Exception:
        #             prevs_scheduler = {}

        #     if "StopServer" in received_data:
        #         server.close()
        #         self.task_queue.join()
        #         logger.info("Connection closed.")
        #         quit()


def launch_gui(manager_gui_queue=None, testing=False):
    logger.info("[TrueScheduler launch_gui] App launched")
    app = QApplication(sys.argv)
    server_gui_queue = Queue()
    login_screen = LoginScreen(server_gui_queue)
    login_screen.resize(1000, 650)
    login_screen.show()

    welcome_screen = ExperimentTracker(server_gui_queue)
    welcome_screen.resize(1000, 650)
    data_receiver = DataReceiver(server_gui_queue)

    if testing:
        data_receiver_thread = threading.Thread(
            target=data_receiver.run_gui_testing,
            args=(login_screen, welcome_screen),
        )
    else:
        logger.info(
            "[TrueScheduler launch_gui] spawning run_gui thread from data_receiver"
        )
        data_receiver_thread = threading.Thread(
            target=data_receiver.run_gui,
            args=(login_screen, welcome_screen, data_receiver, manager_gui_queue),
        )

    data_receiver_thread.start()

    data_receiver.data_received.connect(welcome_screen.show)
    data_receiver.data_received.connect(login_screen.hide)
    sys.exit(app.exec())


def startup_scheduler():
    logging.basicConfig(filename="myapp.log", level=logging.INFO)
    logger.info("[TrueScheduler] started logging")

    manager_gui_queue = Queue()  # between GUI and LiaseThread
    server_manager_queue = Queue()  # between LiaseThread and ServerListenerThread
    gui_listener_thread = LiaseThread(manager_gui_queue, server_manager_queue)
    gui_listener_thread.start()

    server_listener_thread = ServerListenerThread(server_manager_queue)
    server_listener_thread.start()

    launch_gui(manager_gui_queue)
    logger.info("[TrueScheduler] finished logging")


if __name__ == "__main__":
    startup_scheduler()
