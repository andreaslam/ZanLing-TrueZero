# Use the official Ubuntu base image
FROM ubuntu:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Update package lists and install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3.12-venv \
    build-essential \
    curl \
    libglib2.0-0 \
    libgl1 \
    protobuf-compiler \
    pkg-config \
    libssl-dev \
    libclang-dev \
    clang \
    nano \
    htop \
    && apt-get clean \
    && apt-get autoclean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Set environment variables for Rust
ENV PATH="/root/.cargo/bin:${PATH}"

# Set up requirements.txt

RUN curl -o requirements-tch-only.txt -L https://github.com/andreaslam/ZanLing-TrueZero/raw/main/requirements-tch-only.txt

RUN python3 -m venv .venv

RUN .venv/bin/pip3 install -r requirements-tch-only.txt

ENV LIBTORCH_USE_PYTORCH=1
RUN PYTORCH_PATH=$(.venv/bin/python3 -c "import torch; print(torch.__path__[0])") && \
    export LD_LIBRARY_PATH="$PYTORCH_PATH/lib:$LD_LIBRARY_PATH"
