import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
import time
import shutil  # Added for directory operations


class TensorBoardManager:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.summary_writer = None

    def create_writer(self):
        if self.summary_writer:
            self.summary_writer.close()
        self.summary_writer = SummaryWriter(self.log_dir)

    def add_scalar(self, kpi, plot, x):
        if self.summary_writer is None:
            self.create_writer()
        self.summary_writer.add_scalar(kpi, plot, x)

    def close_writer(self):
        if self.summary_writer:
            self.summary_writer.close()
            self.summary_writer = None


def delete_previous_tensorboard_instance(log_dir):
    try:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            print(f"Deleted previous TensorBoard instance at {log_dir}")
    except Exception as e:
        print(f"Error deleting previous TensorBoard instance: {e}")


def process_data(kpi, plots, timestamp):
    try:
        tensorboard_manager = TensorBoardManager("runs/experiment_" + str(timestamp))
        for x, plot in enumerate(plots):
            tensorboard_manager.add_scalar(kpi, plot, x)
        tensorboard_manager.close_writer()
    except Exception as e:
        print(f"Error in process_data: {e}")


def replace_nan_with_previous_or_zero(d):
    for key, value in d.items():
        nonzero_index = 0
        while nonzero_index < len(value) and (
            np.isnan(value[nonzero_index]) or value[nonzero_index] == 0
        ):
            nonzero_index += 1
        if nonzero_index < len(value):
            first_nonzero_value = value[nonzero_index]
        else:
            first_nonzero_value = 0
        prev_value = first_nonzero_value
        for i in range(len(value)):
            if np.isnan(value[i]) or value[i] == 0:
                if i == 0:
                    value[i] = first_nonzero_value
                else:
                    value[i] = prev_value
            else:
                prev_value = value[i]


def lazy_data_loader():
    try:
        print("Loading data from log.npz")
        data = np.load("log.npz")
        data = {tuple(k): v for k, v in zip(data["keys"], data["values"])}
        data = {"-".join(key): value for key, value in data.items()}
        replace_nan_with_previous_or_zero(data)
        return data.items()  # Return an iterator of key-value pairs
    except Exception as e:
        print(f"Error in lazy_data_loader: {e}")


def lazy_plot(log_path, timestamp):
    try:
        lazy_data = lazy_data_loader()
        for kpi, plots in lazy_data:
            process_data(kpi, plots, timestamp)
        print(f"Updated plots for timestamp {timestamp}")
    except Exception as e:
        print(f"Error in lazy_plot: {e}")


def plot_loop(log_path):
    last_mtime = 0  # Initial last modification time
    prev_instance = ""
    while True:
        try:
            current_mtime = os.path.getmtime(log_path)

            if current_mtime > last_mtime:
                # Delete previous TensorBoard instance
                delete_previous_tensorboard_instance(prev_instance)
                timestamp = int(time.time())
                print(f"{log_path} has been modified. Updating plots...")
                lazy_plot(log_path, timestamp)
                last_mtime = current_mtime
                prev_instance = "runs/experiment_" + str(timestamp)

            time.sleep(60)
        except Exception as e:
            print(f"Error in plot_loop: {e}")
            time.sleep(120)


def run_tensorboard():
    import experiment

    experiment.run_tensorboard(6006)


if __name__ == "__main__":
    log_path = "log.npz"

    # Start the TensorBoard server in a separate process
    tensorboard_process = multiprocessing.Process(target=run_tensorboard)
    tensorboard_process.start()

    # Initial plot
    plot_loop(log_path)

    # Ensure the TensorBoard process is terminated before exiting
    tensorboard_process.terminate()
