#!/bin/sh

# utilities
add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt-get install -y curl wget git htop nano iputils-ping tree nethogs rsync unzip zip tmux
apt-get upgrade -y git

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

git clone https://github.com/andreaslam/ZanLing-TrueZero

# install project deps
pip install -r requirements.txt
apt install -y libglib2.0-0 libgl1 protobuf-compiler

# add tch-rs

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1 # just in case

pip install darkdetect
pip install pyqtgraph
pip install PyQt6

# build code
cargo build --release 
cargo run --bin main --release &
cargo run --bin server --release &
python client.py &
