import torch
import subprocess
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
import time


def add_scalar_to_tensorboard(writer, kpi, plot, x):
    writer.add_scalar(kpi, plot, x)


def process_data(kpi, plots):
    with SummaryWriter("runs/experiment1") as summary_writer:
        for x, plot in enumerate(plots):
            add_scalar_to_tensorboard(summary_writer, kpi, plot, x)


def replace_nan_with_previous_or_zero(d):
    for key, value in d.items():
        prev_value = 0
        for i in range(len(value)):
            if np.isnan(value[i]):
                value[i] = prev_value
            else:
                prev_value = value[i]


def lazy_data_loader():
    data = np.load("log.npz")
    data = {tuple(k): v for k, v in zip(data["keys"], data["values"])}
    data = {"-".join(key): value for key, value in data.items()}
    replace_nan_with_previous_or_zero(data)
    for item in data.items():
        yield item


def lazy_plot(path):
    lazy_data = lazy_data_loader()
    with multiprocessing.Pool() as pool:
        pool.starmap(process_data, lazy_data)


def plot(path):
    tic = time.perf_counter()
    lazy_plot(path)
    toc = time.perf_counter()
    print(f"Processed the tensorboard in {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    plot("log.npz")
