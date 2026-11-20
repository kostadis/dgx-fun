#!/bin/bash
set -eu
cd "$HOME/qwen38-dflash-trial/source"
export PATH="$HOME/inferssd-trial/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build build --parallel 2 --target llama-server
printf 'build complete\n'
date -u +%FT%TZ > ../build-complete.txt
