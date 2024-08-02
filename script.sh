#!/bin/sh

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1 
PYTORCH_PATH=$(python3 -c "import torch; print(torch.__path__[0])")
export LD_LIBRARY_PATH="$PYTORCH_PATH/lib:$LD_LIBRARY_PATH"
