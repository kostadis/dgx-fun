#!/bin/bash
set -eu
cd "$HOME/inferssd-trial/source"
export PATH="$HOME/inferssd-trial/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake --preset linux-cuda -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_PREFIX_PATH="$HOME/inferssd-trial/deps/usr" -DCMAKE_BUILD_RPATH="$HOME/inferssd-trial/deps/usr/lib/aarch64-linux-gnu"
cmake --build build-linux-cuda --parallel 2 --target inferssd-server inferssd-cpuprobe inferssd-ssdprobe inferssd-memprobe
CUDA_VISIBLE_DEVICES= python scripts/engine-resources.py --build build-linux-cuda --output ../engine-resources.json
../venv/bin/inferssd --version
