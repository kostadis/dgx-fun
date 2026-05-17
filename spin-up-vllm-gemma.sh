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
#     scp spin-up-vllm-gemma.sh kostadis@192.168.1.147:~/
#     ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-gemma.sh'
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

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-2-9b-it}"
GEMMA_PORT="${GEMMA_PORT:-8002}"
GPU_UTIL="${GPU_UTIL:-0.15}"
MAX_LEN="${MAX_LEN:-8192}"
CONTAINER_NAME="vllm-gemma"
IMAGE="vllm/vllm-openai:latest"

echo "=== spin-up-vllm-gemma ==="
echo "  model:    ${GEMMA_MODEL}"
echo "  port:     ${GEMMA_PORT}"
echo "  gpu_util: ${GPU_UTIL}"
echo "  max_len:  ${MAX_LEN}"
echo ""

# Stop any existing container with the same name.
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "→ stopping existing ${CONTAINER_NAME}..."
    docker stop "${CONTAINER_NAME}" 2>&1 || true
    docker rm "${CONTAINER_NAME}" 2>&1 || true
fi

# Confirm GPU is healthy before we ask vLLM to grab some of it.
echo "→ GPU status pre-launch:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | head -3
echo ""

# Start vllm-gemma.
echo "→ starting ${CONTAINER_NAME}..."
docker run -d \
    --runtime nvidia --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${GEMMA_PORT}:${GEMMA_PORT}" \
    --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${IMAGE}" \
    "${GEMMA_MODEL}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --host 0.0.0.0 --port "${GEMMA_PORT}"

# Wait for healthy.
echo ""
echo "→ waiting for 'Application startup complete' in container logs..."
echo "  (this includes model download on first run — could be 5+ min for a fresh 9B)"
echo ""
DEADLINE=$(( $(date +%s) + 600 ))  # 10 min budget
while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
    if docker logs "${CONTAINER_NAME}" 2>&1 | grep -q "Application startup complete"; then
        echo "  ✓ ready"
        break
    fi
    # Watch for early-failure signals.
    if docker logs "${CONTAINER_NAME}" 2>&1 | grep -qE "Error|CUDA error|out of memory|Traceback"; then
        echo ""
        echo "  ✗ container errored during startup. Last 30 log lines:"
        docker logs --tail 30 "${CONTAINER_NAME}" 2>&1
        exit 1
    fi
    # Print a heartbeat every 30s so we know we're alive.
    sleep 15
    elapsed=$(( $(date +%s) - DEADLINE + 600 ))
    last=$(docker logs --tail 1 "${CONTAINER_NAME}" 2>&1 | tr -d '\r' | cut -c1-100)
    echo "  [${elapsed}s] ${last}"
done

if [ "$(date +%s)" -ge "${DEADLINE}" ]; then
    echo ""
    echo "  ✗ 10-minute startup budget exceeded. Last 30 log lines:"
    docker logs --tail 30 "${CONTAINER_NAME}" 2>&1
    exit 1
fi

# Smoke-test the endpoint.
echo ""
echo "→ smoke-test: GET /v1/models"
curl -sS --max-time 5 "http://localhost:${GEMMA_PORT}/v1/models" | python3 -m json.tool 2>&1 | head -20

echo ""
echo "→ smoke-test: tiny chat completion"
curl -sS --max-time 30 "http://localhost:${GEMMA_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(cat <<EOF
{
  "model": "${GEMMA_MODEL}",
  "messages": [{"role": "user", "content": "Reply only with the word OK."}],
  "max_tokens": 10
}
EOF
)" | python3 -m json.tool 2>&1 | head -30

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
