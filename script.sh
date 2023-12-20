#!/bin/sh

# utilities
add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt-get install -y curl wget git htop nano iputils-ping tree nethogs rsync unzip zip tmux
apt-get upgrade -y git

RUN apt update && \
    apt install -y libopencv-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# speedtest
curl -s https://packagecloud.io/install/repositories/ookla/speedtest-cli/script.deb.sh | bash
apt-get -y install speedtest




# apt install -y valgrind

# set max file handles to unlimited
ulimit -u 2048

# rust & deps
apt install -y pkg-config libssl-dev libclang-dev clang
curl https://sh.rustup.rs -sSf | sh -s -- -y
. ~/.cargo/env

# python venv 

python -m venv

# install project deps
apt install -y libglib2.0-0 libgl1 protobuf-compiler
pip install -r requirements.txt

pip3 install darkdetect
pip3 install pyqtgraph

# add tch-rs

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1 # just in case

# Set LD_LIBRARY_PATH to PyTorch path
# Find PyTorch installation directory
PYTORCH_PATH=$(python -c "import torch; print(torch.__path__[0])")

# Set LD_LIBRARY_PATH to PyTorch path
export LD_LIBRARY_PATH="$PYTORCH_PATH/lib:$LD_LIBRARY_PATH"

# build code
cargo build --release 
# cargo run --bin main --release &
# cargo run --bin server --release &
# python client.py &
