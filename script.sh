#!/bin/sh

# Store the current directory
ORIGINAL_DIR=$(pwd)

# Define variables for Python version and download URL
PYTHON_VERSION="3.11.4"
PYTHON_URL="https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tar.xz"

# Download Python source code
wget $PYTHON_URL

# Extract the downloaded Python source code
tar -xf Python-$PYTHON_VERSION.tar.xz

# Navigate to the extracted directory
cd Python-$PYTHON_VERSION || exit 1 # Exit if directory doesn't exist

# Install necessary dependencies
sudo apt-get update
sudo apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget

# Configure and install Python
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall

# Clean up downloaded files
cd "$ORIGINAL_DIR" || exit 1 # Return to the original directory
rm -r Python-$PYTHON_VERSION
rm Python-$PYTHON_VERSION.tar.xz

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

# # .gitconfig
# curl https://gist.githubusercontent.com/KarelPeeters/a3421a43e60524b3f12c8f626f7545d3/raw/ > ~/.gitconfig
# git config --global credential.helper cache

# rust & deps
apt install -y pkg-config libssl-dev libclang-dev clang
curl https://sh.rustup.rs -sSf | sh -s -- -y
. ~/.cargo/env

# custom repos

# install project deps
pip install -r requirements.txt
apt install -y libglib2.0-0 libgl1 protobuf-compiler

# add tch-rs

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1 # just in case


# for some reason these deps arent imported correctly
pip install darkdetect
pip install pyqtgraph
pip3 install pyqtgraph

# build code
cargo build --release 
cargo run --bin main --release &
cargo run --bin server --release &
python client.py &
