#!/usr/bin/env bash
#
# spin-up-llamacpp-gemma4-26b-qat.sh — serve Google's official QAT Q4_0 GGUF of
# Gemma 4 26B-A4B with llama.cpp's llama-server on ONE Spark.
#
# The vLLM/NVFP4 sibling is spin-up-vllm-gemma4-26b-nvfp4.sh. This one uses
# Google's own quantization-aware-trained weights, which only ship as GGUF, so
# it also compares llama.cpp against vLLM on the same model.
#
# PREREQS (on the box)
#   llama.cpp built with CUDA for GB10:
#     git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp-gemma4
#     cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DLLAMA_CURL=OFF
#     cmake --build build -j 16 --target llama-server llama-bench
#   GGUF in the HF cache: hf download google/gemma-4-26B-A4B-it-qat-q4_0-gguf
#
# Runs as a host process (not docker) under nohup; log at ~/llamacpp-gemma4.log.
# Text only: the mmproj vision projector is not loaded unless MMPROJ=1.
#
# TEARDOWN
#   pkill -f '[l]lama-server.*gemma-4-26B'
#
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp-gemma4}"
PORT="${PORT:-8001}"
CTX="${CTX:-131072}"          # one pool shared by all slots (--kv-unified)
PARALLEL="${PARALLEL:-8}"     # concurrent slots
SERVED="${SERVED:-google/gemma-4-26B-A4B-it-qat}"
MMPROJ="${MMPROJ:-0}"
LOG="$HOME/llamacpp-gemma4.log"

snap=$(ls -d ~/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it-qat-q4_0-gguf/snapshots/*/ | head -1)
MODEL_FILE="${snap}gemma-4-26B_q4_0-it.gguf"
EXTRA=()
if [ "${MMPROJ}" = "1" ]; then
    EXTRA+=(--mmproj "${snap}gemma-4-26B-it-mmproj.gguf")
fi

if pgrep -f '[l]lama-server.*gemma-4-26B' >/dev/null; then
    echo "→ stopping existing llama-server..."
    pkill -f '[l]lama-server.*gemma-4-26B'
    sleep 3
fi

echo "=== spin-up-llamacpp-gemma4-26b-qat ==="
echo "  model:    ${MODEL_FILE}"
echo "  build:    $(git -C "${LLAMA_DIR}" log -1 --format='%h %cd')"
echo "  port:     ${PORT}  ctx: ${CTX}  parallel: ${PARALLEL}  mmproj: ${MMPROJ}"
free -g | sed -n 2p

nohup "${LLAMA_DIR}/build/bin/llama-server" \
    -m "${MODEL_FILE}" \
    --alias "${SERVED}" \
    --host 0.0.0.0 --port "${PORT}" \
    -ngl 999 \
    -fa on \
    -c "${CTX}" \
    -np "${PARALLEL}" \
    --kv-unified \
    --jinja \
    "${EXTRA[@]}" \
    > "${LOG}" 2>&1 &

echo "→ waiting for /health..."
for _ in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null; then
        echo "  ✓ ready"
        break
    fi
    if ! pgrep -f '[l]lama-server.*gemma-4-26B' >/dev/null; then
        echo "  ✗ llama-server exited. Last 40 log lines:"
        tail -40 "${LOG}"
        exit 1
    fi
    sleep 5
done

curl -sS "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply only with the word OK.\"}],\"max_tokens\":16,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
echo ""
free -g | sed -n 2p
echo "  teardown: pkill -f '[l]lama-server.*gemma-4-26B'"
