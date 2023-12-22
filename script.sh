#!/bin/sh

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1 # just in case

# Set LD_LIBRARY_PATH to PyTorch path
# Find PyTorch installation directory
PYTORCH_PATH=$(python3 -c "import torch; print(torch.__path__[0])")

# Set LD_LIBRARY_PATH to PyTorch path
export LD_LIBRARY_PATH="$PYTORCH_PATH/lib:$LD_LIBRARY_PATH"
