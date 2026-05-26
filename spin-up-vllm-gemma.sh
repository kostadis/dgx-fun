#!/usr/bin/env bash
#
# spin-up-vllm-gemma.sh — bring up a vllm-gemma container on the DGX Spark.
#
# WHAT THIS DOES
#   1. Stop + remove any existing vllm-gemma container (clean restart).
#   2. Start a vLLM container on port 8002 serving the chosen Gemma model.
#   3. Wait for "Application startup complete" in the container logs.
#   4. Smoke-test the /v1/models endpoint.
#
# USAGE
#   Copy this script onto the Spark and run:
#     bash spin-up-vllm-gemma.sh
#   Or scp it over and run:
#     scp spin-up-vllm-gemma.sh lib-vllm-spinup.sh spark:~/
#     ssh spark 'bash ~/spin-up-vllm-gemma.sh'
#
# CONFIGURATION
#   Tweak the values below or override via env vars:
#     GEMMA_MODEL    HF model id; default google/gemma-2-9b-it
#     GEMMA_PORT     host port; default 8002
#     GPU_UTIL       --gpu-memory-utilization; default 0.15 (~19 GB of 128 GB)
#     MAX_LEN        --max-model-len; default 8192 (Gemma 2's native max)
#
#   For a SUPER fast iteration model, swap in:
#     GEMMA_MODEL=google/gemma-2-2b-it GPU_UTIL=0.05 bash spin-up-vllm-gemma.sh
#
#   For a NEWER Gemma 3 variant (requires recent vLLM):
#     GEMMA_MODEL=google/gemma-3-4b-it bash spin-up-vllm-gemma.sh
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-2-9b-it}"
GEMMA_PORT="${GEMMA_PORT:-8002}"
GPU_UTIL="${GPU_UTIL:-0.15}"
MAX_LEN="${MAX_LEN:-8192}"
CONTAINER_NAME="vllm-gemma"
IMAGE="vllm/vllm-openai:latest"

vllm_load_hf_token

echo "=== spin-up-vllm-gemma ==="
echo "  model:    ${GEMMA_MODEL}"
echo "  port:     ${GEMMA_PORT}"
echo "  gpu_util: ${GPU_UTIL}"
echo "  max_len:  ${MAX_LEN}"
echo ""

vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

echo "→ starting ${CONTAINER_NAME}..."
docker run -d \
    --runtime nvidia --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${GEMMA_PORT}:${GEMMA_PORT}" \
    --ipc=host \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${IMAGE}" \
    "${GEMMA_MODEL}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --host 0.0.0.0 --port "${GEMMA_PORT}"

# 10-minute budget — first run includes the model download.
vllm_wait_ready "${CONTAINER_NAME}" 600 "" "" 30
vllm_smoke_test localhost "${GEMMA_PORT}" "${GEMMA_MODEL}"

echo ""
echo "=== done ==="
echo ""
echo "From your workstation, you can now hit Gemma at:"
echo "  http://192.168.1.147:${GEMMA_PORT}/v1/chat/completions"
echo ""
echo "To use Gemma for closet_llm regen:"
echo "  MEMPALACE_WORKERS=16 python -m mempalace.closet_llm \\"
echo "    --palace ~/.mempalace/palaces/campaign-dev \\"
echo "    --endpoint http://192.168.1.147:${GEMMA_PORT}/v1 \\"
echo "    --model ${GEMMA_MODEL} \\"
echo "    --sample 16"
echo ""
echo "To stop:"
echo "  docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}"
