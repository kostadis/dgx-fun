#!/usr/bin/env bash
#
# spin-up-vllm-llama70b.sh — replace vllm-chat on the DGX Spark with
# Llama 3.3 70B Instruct AWQ. Native 128K context (no YaRN tricks),
# different family than Qwen for cross-family calibration.
#
# WHAT THIS DOES
#   1. Stop + remove the existing vllm-chat container (Qwen 14B AWQ).
#   2. Start a vLLM container on port 8001 serving Llama 3.3 70B Instruct AWQ.
#   3. Wait for "Application startup complete" in the container logs.
#      First run pulls ~40 GB from HuggingFace; 20-min budget.
#   4. Smoke-test /v1/models and a tiny chat completion.
#
# USAGE
#   scp spin-up-vllm-llama70b.sh lib-vllm-spinup.sh spark:~/
#   ssh spark 'bash ~/spin-up-vllm-llama70b.sh'
#
# CONFIGURATION
#   LLAMA_MODEL    HF model id; default casperhansen/llama-3.3-70b-instruct-awq
#                  (override if that quant has been deleted or you want a
#                   different AWQ variant — search HF for "Llama-3.3-70B-Instruct-AWQ")
#   LLAMA_PORT     host port; default 8001 (REPLACES vllm-chat)
#   GPU_UTIL       --gpu-memory-utilization; default 0.6 (~77 GB of 128 GB)
#   MAX_LEN        --max-model-len; default 65536 (Llama 3.3 native is 128K;
#                  64K is enough headroom for session_doc + leaves KV cache room)
#
# REVERTING TO QWEN
#   bash ~/src/dgx/spin-up-vllm-chat.sh    # if you've made one;
#   or paste the docker run from current-setup.md §3.
#
# CALLERS AFFECTED
#   Anything pointing at http://192.168.1.147:8001/v1/* will start hitting
#   Llama instead of Qwen — that's the point. llm_wiki, session_doc.py
#   with --dgx-endpoint, etc.
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

LLAMA_MODEL="${LLAMA_MODEL:-casperhansen/llama-3.3-70b-instruct-awq}"
LLAMA_PORT="${LLAMA_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.6}"
MAX_LEN="${MAX_LEN:-65536}"
CONTAINER_NAME="vllm-chat"
IMAGE="vllm/vllm-openai:latest"

llama70b_failure_hints() {
    cat <<EOF
  Common causes:
   - 'Repository Not Found': the AWQ repo name has changed.
     Try: LLAMA_MODEL=<other-repo> bash $0
   - OOM: lower MAX_LEN or GPU_UTIL.
   - AWQ-Marlin kernel mismatch: try vllm/vllm-openai:v0.6.x pinned.
EOF
}

vllm_load_hf_token

echo "=== spin-up-vllm-llama70b ==="
echo "  model:    ${LLAMA_MODEL}"
echo "  port:     ${LLAMA_PORT}"
echo "  gpu_util: ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:  ${MAX_LEN}"
echo ""

vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

echo "→ starting ${CONTAINER_NAME} with ${LLAMA_MODEL}..."
docker run -d \
    --runtime nvidia --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${LLAMA_PORT}:${LLAMA_PORT}" \
    --ipc=host \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${IMAGE}" \
    "${LLAMA_MODEL}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --host 0.0.0.0 --port "${LLAMA_PORT}"

# 20-min budget — first run includes ~40 GB HF download.
vllm_wait_ready "${CONTAINER_NAME}" 1200 "" llama70b_failure_hints 40

vllm_smoke_test localhost "${LLAMA_PORT}" "${LLAMA_MODEL}"

echo ""
echo "=== done ==="
echo ""
echo "From your workstation, you can now hit Llama 3.3 70B at:"
echo "  http://192.168.1.147:${LLAMA_PORT}/v1/chat/completions"
echo ""
echo "For the session_doc.py calibration experiment:"
echo "  DGX_MODEL=${LLAMA_MODEL} python session_doc.py ... \\"
echo "    --dgx-endpoint http://192.168.1.147:${LLAMA_PORT}/v1"
echo ""
echo "(The DGX adapter in campaignlib.py picks up DGX_MODEL via env var,"
echo " so no --dgx-model flag needed if you set this.)"
echo ""
echo "To revert to Qwen 14B: see current-setup.md §3 and re-run that docker invocation."
