import sys
import threading
from queue import Queue
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QTabWidget,
    QDateTimeEdit,
    QHBoxLayout,
    QGridLayout,
    QCheckBox,
    QMessageBox,
)
from PySide6.QtCore import Signal, QObject, QDateTime, Qt, QThread, QCoreApplication
import re
import logging
import os
import json

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="myapp.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def validate_ip(ip):
    ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    if ip == "localhost":
        return True
    else:
        return bool(re.match(ip_pattern, ip))


class LoginScreen(QWidget):
    login_failed = Signal()

    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.setWindowTitle("TrueScheduler - SSH Login")

        container_layout = QVBoxLayout(self)
        container_layout.setAlignment(Qt.AlignCenter)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(30)

        form_layout = QGridLayout()

        self.hostname_label = QLabel("Hostname:")
        self.hostname_box = QLineEdit()
        self.hostname_box.setPlaceholderText("Enter hostname")
        form_layout.addWidget(self.hostname_label, 0, 0)
        form_layout.addWidget(self.hostname_box, 0, 1)

        self.ip_label = QLabel("IP Address:")
        self.ip_box = QLineEdit()
        self.ip_box.setPlaceholderText("Enter IP address")
        form_layout.addWidget(self.ip_label, 1, 0)
        form_layout.addWidget(self.ip_box, 1, 1)

        self.port_label = QLabel("Port:")
        self.port_box = QLineEdit()
        self.port_box.setPlaceholderText("Enter port")
        form_layout.addWidget(self.port_label, 2, 0)
        form_layout.addWidget(self.port_box, 2, 1)

        self.password_label = QLabel("Password:")
        self.text_box2 = QLineEdit()
        self.text_box2.setPlaceholderText("Enter password")
        self.text_box2.setEchoMode(QLineEdit.Password)
        form_layout.addWidget(self.password_label, 3, 0)
        form_layout.addWidget(self.text_box2, 3, 1)

        main_layout.addLayout(form_layout)

        checkbox_layout = QGridLayout()
        checkbox_layout.setSpacing(30)

        self.port_forwarding_checkbox = QCheckBox("Enable Port Forwarding")
        self.port_forwarding_checkbox.stateChanged.connect(self.toggle_port_box)
        checkbox_layout.addWidget(self.port_forwarding_checkbox, 0, 0, 1, 2)

        self.local_port_label = QLabel("Local Port:")
        self.local_port_box = QLineEdit()
        self.local_port_box.setPlaceholderText("Enter local port")
        self.local_port_box.setEnabled(False)
        checkbox_layout.addWidget(self.local_port_label, 1, 0)
        checkbox_layout.addWidget(self.local_port_box, 1, 1)

        main_layout.addLayout(checkbox_layout)

        button_layout = QVBoxLayout()
        button_layout.setSpacing(30)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit_form)
        button_layout.addWidget(self.submit_button)

        self.error_label = QLabel()
        button_layout.addWidget(self.error_label)
        self.success_label = QLabel()
        button_layout.addWidget(self.success_label)

        main_layout.addLayout(button_layout)
        self.login_failed.connect(self.show)

        container_layout.addLayout(main_layout)

    def submit_form(self):
        password = self.text_box2.text()
        hostname = self.hostname_box.text()
        ip = self.ip_box.text()
        port = self.port_box.text()
        local_port = (
            self.local_port_box.text()
            if self.port_forwarding_checkbox.isChecked()
            else ""
        )
        port_forwarding = self.port_forwarding_checkbox.isChecked()

        if not password or not hostname or not ip or not port:
            self.show_error_message("Please fill in all fields.")
            return

        if not port.isdigit() or (
            self.port_forwarding_checkbox.isChecked() and not local_port.isdigit()
        ):
            self.show_error_message("Port number should only contain digits.")
            return

        if not validate_ip(ip):
            self.show_error_message("Invalid IP address format.")
            return

        form_data = {
            "password": password,
            "hostname": hostname,
            "ip": ip,
            "port": port,
            "local_port": local_port,
            "port_forwarding": port_forwarding,
        }
        self.queue.put(form_data)
        self.text_box2.clear()
        self.hostname_box.clear()
        self.ip_box.clear()
        self.port_box.clear()
        self.local_port_box.clear()

    def show_error_message(self, message):
        self.error_label.setStyleSheet(
            "color: red; text-align: center; border: 2px solid red; padding: 5px;"
        )
        self.error_label.setText(message)
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.show()

    def show_success_message(self, message):
        self.success_label.setStyleSheet(
            "color: #00FF7F; text-align: center; border: 2px solid #00FF7F; padding: 5px;"
        )
        self.success_label.setText(message)
        self.success_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.success_label.show()

    def clear_error_message(self):
        self.error_label.clear()
        self.error_label.hide()

    def toggle_port_box(self):
        self.local_port_box.setEnabled(self.port_forwarding_checkbox.isChecked())


class ExperimentTracker(QWidget):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.setWindowTitle("TrueScheduler - Experiment Scheduler")

        self.file_name = "experiments.json"
        # Check if the file exists
        try:
            if os.path.exists(self.file_name):
                # If it exists, load the JSON data into a Python dictionary
                with open(self.file_name, "r") as file:
                    self.prevs = json.load(file)
            else:
                # If it doesn't exist, create an empty dictionary
                with open(self.file_name, "w") as file:
                    pass

                self.prevs = {}
        except Exception:
            self.prevs = {}
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        self.tab_widget = QTabWidget()

        self.add_tab()

        layout.addWidget(self.tab_widget)

        self.add_tab_button = QPushButton("Add New Experiment")
        self.add_tab_button.clicked.connect(self.add_tab)
        layout.addWidget(self.add_tab_button)

        self.submit_button = QPushButton("Schedule Experiments")
        self.submit_button.setStyleSheet(
            "background-color: steelblue; border-radius: 5px"
        )
        self.submit_button.clicked.connect(self.submit_form)
        self.warning_label = QLabel()
        self.error_label = QLabel()
        self.success_label = QLabel()
        layout.addWidget(self.warning_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.success_label)
        layout.addWidget(self.submit_button)

    def add_tab(self):
        new_tab = QWidget()
        new_tab_layout = QVBoxLayout(new_tab)

        experiment_name_label = QLabel("Experiment name:")
        experiment_name_box = QLineEdit()
        experiment_name_box.setPlaceholderText("Enter experiment name")
        experiment_name_box.setObjectName("experiment_name_box")
        self.original_index = self.tab_widget.count() + 1
        experiment_name_box.textChanged.connect(
            lambda text, tab=new_tab: self.update_tab_title(text, tab)
        )
        new_tab_layout.addWidget(experiment_name_label)
        new_tab_layout.addWidget(experiment_name_box)

        command_label = QLabel("Command to Run:")
        command_box = QLineEdit()
        command_box.setPlaceholderText("Enter command to run")
        command_box.setObjectName("command_box")
        new_tab_layout.addWidget(command_label)
        new_tab_layout.addWidget(command_box)

        test_parameters_label = QLabel("Test Parameters:")
        test_parameters_box = QLineEdit()
        test_parameters_box.setPlaceholderText("Enter test parameters ")
        test_parameters_box.setObjectName("test_parameters_box")
        new_tab_layout.addWidget(test_parameters_label)
        new_tab_layout.addWidget(test_parameters_box)

        datetime_picker_label = QLabel("Start Time:")
        datetime_picker = QDateTimeEdit()
        datetime_picker.setObjectName("datetime_picker")
        datetime_picker.setDateTime(
            QDateTime.currentDateTime()
        )  # Set current time as default
        new_tab_layout.addWidget(datetime_picker_label)
        new_tab_layout.addWidget(datetime_picker)

        self.tab_widget.addTab(new_tab, f"Experiment {self.tab_widget.count() + 1}")

    def update_tab_title(self, text, tab):
        index = self.tab_widget.indexOf(tab)
        if text == "":
            self.tab_widget.setTabText(index, f"Experiment {self.original_index}")
        elif index != -1:
            self.tab_widget.setTabText(index, text)

    def submit_form(self):
        self.clear_messages()
        form_data = {}
        current_time = QDateTime.currentDateTime()  # Get current time
        incorrects = False
        alr_scheduled = False
        missed = []
        success_scheduled = []
        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            datetime_picker = tab.findChild(QDateTimeEdit, "datetime_picker")
            experiment_name_box = tab.findChild(QLineEdit, "experiment_name_box")
            command_box = tab.findChild(QLineEdit, "command_box")
            test_parameters_box = tab.findChild(QLineEdit, "test_parameters_box")

            if experiment_name_box.text():
                experiment_name = experiment_name_box.text()
            else:
                experiment_name = f"Experiment {i+1}"
            if (
                datetime_picker.dateTime() >= current_time
            ):  # Check if datetime is in the future

                tab_data = {
                    "experiment_name": experiment_name,
                    "command": command_box.text(),
                    "test_parameters": test_parameters_box.text(),
                    "start_time": datetime_picker.dateTime().toString(),
                }

                if tab_data not in self.prevs.values():
                    print("self.prevs", self.prevs)
                    form_data[f"Experiment {i + 1}"] = tab_data
                    success_scheduled.append(experiment_name)
                    self.prevs[f"Experiment {i + 1}"] = tab_data
                else:
                    alr_scheduled = True
            else:
                incorrects = True
                missed.append(experiment_name)

        if form_data:  # Only put data in queue if there are valid experiments
            self.queue.put(form_data)
            if incorrects:
                self.show_warning_message(
                    f"The following experiments have already passed their start time: {', '.join([experiment for experiment in missed])}"
                )
            else:
                self.show_success_message(
                    f"Scheduled the following successfully: {', '.join([experiment for experiment in success_scheduled])}"
                )
            if alr_scheduled:
                self.show_warning_message(
                    f"The following experiments have been scheduled before: {', '.join([experiment for experiment in success_scheduled])}"
                )
        else:
            self.show_error_message(
                "No experiments scheduled. Check experiment parameters!"
            )

    def show_warning_message(self, message):
        self.warning_label.setStyleSheet(
            "color: yellow; text-align: center; border: 2px solid yellow; padding: 5px;"
        )
        self.warning_label.setText(message)
        self.warning_label.show()

    def show_error_message(self, message):
        self.error_label.setStyleSheet(
            "color: red; text-align: center; border: 2px solid red; padding: 5px;"
        )
        self.error_label.setText(message)
        self.error_label.show()

    def show_success_message(self, message):
        self.success_label.setStyleSheet(
            "color: #00FF7F; text-align: center; border: 2px solid #00FF7F; padding: 5px;"
        )
        self.success_label.setText(message)
        self.success_label.show()

    def clear_messages(self):
        self.warning_label.clear()
        self.warning_label.hide()
        self.error_label.clear()
        self.error_label.hide()
        self.success_label.clear()
        self.success_label.hide()


# TODO: build an experiment scheulder (SPRT) and an experiment rescheuduler (open json file and edit)


class DataReceiver(QObject):
    data_received = Signal()

    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run_gui(
        self, login_screen, experiment_tracker, data_receiver, manager_gui_queue
    ):
        done = False
        while True:  # login loop
            if done:
                break
            output = self.queue.get()  # Get input from login screen
            manager_gui_queue.put(output)  # Send input to manager_gui

            prev = None
            logger.info(f"pre-loop: {prev}, {output}")
            # loop for waiting response
            while True:
                listener_output = (
                    manager_gui_queue.get()
                )  # Get response from server listener
                manager_gui_queue.put(
                    listener_output
                )  # re-send input to manager_gui, since checking the response itself consumes the message
                logger.info(f"const recving: {prev}, {output}, {listener_output}")
                if listener_output != prev and listener_output:
                    if listener_output == "failed":
                        manager_gui_queue.get()
                        logger.info("GUI failed!")
                        login_screen.show_error_message("Incorrect credentials")
                        QThread.msleep(5000)
                        login_screen.login_failed.emit()
                        break
                    elif listener_output == "ok":
                        manager_gui_queue.get()
                        logger.info("GUI success!")
                        login_screen.show_success_message("Login success!")
                        login_screen.clear_error_message()
                        QThread.msleep(1000)
                        self.data_received.emit()
                        done = True
                        break
                    prev = listener_output

        self.data_received.emit()

        # receive form inputs from ExperimentTracker

        # while True:
        #     output = self.queue.get()
        #     print(output)
        #     with open(experiment_tracker.file_name, "w") as json_file:
        #         json.dump(experiment_tracker.prevs, json_file)
        #     # send all data to scheduler
        #     manager_gui_queue.put(output)

    def run_gui_testing(self, login_screen, experiment_tracker):
        password = "bar"
        print(
            f"[Note]: this is the demo version of TrueScheduler. use '{password}' to access the rest of the GUI."
        )
        print("debugging prints and logging enabled.")
        while True:
            output = self.queue.get()
            print(f"[DataReceiver from login]: {output}")
            print(output.get("password") != password)
            if output.get("password") != password:
                login_screen.show_error_message("Incorrect credentials")
                login_screen.login_failed.emit()
            else:
                login_screen.show_success_message("Login success!")
                login_screen.clear_error_message()
                self.data_received.emit()
                break  # Break the loop on successful authentication

        self.data_received.emit()

        # # load/create json file

        # while True:
        #     output = self.queue.get()
        #     print(f"[DataReceiver from form inputs]: {output}")
        #     print(output)
        #     with open(experiment_tracker.file_name, "w") as json_file:
        #         json.dump(experiment_tracker.prevs, json_file)


def main():
    app = QApplication(sys.argv)

    manager_gui_queue = Queue()
    login_screen = LoginScreen(manager_gui_queue)
    welcome_screen = QWidget()  # Placeholder for welcome screen
    data_receiver = DataReceiver(manager_gui_queue)
    data_receiver_thread = threading.Thread(
        target=data_receiver.run_gui,
        args=(login_screen, welcome_screen, data_receiver, manager_gui_queue),
        daemon=True,
    )
    data_receiver_thread.start()

    login_screen.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # tests the GUI only, this does not connect to actual server
    main()
