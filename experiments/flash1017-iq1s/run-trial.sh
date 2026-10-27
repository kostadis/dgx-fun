#!/usr/bin/env bash
set -euo pipefail
cd /home/kostadis/unsloth-flash1017-trial
exec ./llama.cpp/build/bin/llama-server \
 --model model-http/UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf \
 --alias DeepSeek-V4.1-Flash-1017-UD-IQ1_S \
 --host 0.0.0.0 --port 8002 \
 --ctx-size 8192 --parallel 1 \
 --n-gpu-layers 99 --fit off \
 --batch-size 256 --ubatch-size 128 \
 --flash-attn on --load-mode mmap --lazy-mode on \
 --override-tensor 'per_layer_token_embd.weight=CPU' \
 --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
 --chat-template-kwargs '{"enable_thinking":false}'
