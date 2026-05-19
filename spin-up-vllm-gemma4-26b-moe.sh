#!/usr/bin/env bash
#
# spin-up-vllm-gemma4-26b-moe.sh — replace vllm-chat on the DGX Spark
# with Gemma 4 26B MoE (A4B) at BF16. ~52 GB weights, 256K native
# context (we cap at 32K for KV-cache realism), MoE architecture.
#
# WHAT THIS DOES
#   1. Stop + remove the existing vllm-chat container (Llama 70B + spec
#      decode, or whatever else is sitting in the slot).
#   2. Start a vLLM container on port 8001 serving google/gemma-4-26b-a4b-it
#      in BF16, no quantization, no spec decode, no tool calling (yet).
#   3. Wait for "Application startup complete" — 30-min budget because
#      first run pulls ~52 GB and torch.compile may take longer on the
#      PLE / MoE paths than on dense models.
#   4. Smoke-test /v1/models and a tiny chat completion.
#
# USAGE
#   scp spin-up-vllm-gemma4-26b-moe.sh kostadis@192.168.1.147:~/
#   ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-gemma4-26b-moe.sh'
#
# CONFIGURATION
#   GEMMA_MODEL    HF model id; default google/gemma-4-26b-a4b-it
#   GEMMA_PORT     host port; default 8001 (REPLACES vllm-chat)
#   GPU_UTIL       --gpu-memory-utilization; default 0.75 (~96 GB of 128 GB)
#                  Drop to 0.7 if startup hits OOM.
#   MAX_LEN        --max-model-len; default 32768
#                  256K native is infeasible (~130 GB KV cache at FP16).
#                  Bump to 65536 only if you have a real long-context need
#                  and willing to lose concurrency headroom.
#
# REVERTING TO LLAMA 70B + SPEC DECODE
#   bash ~/spin-up-vllm-llama70b-specdecode.sh
#
set -euo pipefail

# Pick up HF_TOKEN from ~/.bashrc if not already in the env. SSH
# non-interactive does NOT source .bashrc, so we extract just the
# export line ourselves.
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.bashrc" ]; then
    eval "$(grep -E '^[[:space:]]*export[[:space:]]+HF_TOKEN=' "${HOME}/.bashrc" | tail -1)" 2>/dev/null || true
fi

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-4-26b-a4b-it}"
GEMMA_PORT="${GEMMA_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.75}"
MAX_LEN="${MAX_LEN:-32768}"
CONTAINER_NAME="vllm-chat"
IMAGE="vllm/vllm-openai:latest"

echo "=== spin-up-vllm-gemma4-26b-moe ==="
echo "  model:    ${GEMMA_MODEL}"
echo "  port:     ${GEMMA_PORT}"
echo "  gpu_util: ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:  ${MAX_LEN}"
echo ""

# Stop existing vllm-chat (whatever's in the slot).
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "→ stopping existing ${CONTAINER_NAME}..."
    docker stop "${CONTAINER_NAME}" 2>&1 || true
    docker rm "${CONTAINER_NAME}" 2>&1 || true
fi

# GPU healthcheck before allocating ~96 GB.
echo "→ GPU status pre-launch:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | head -3
echo ""

echo "→ starting ${CONTAINER_NAME} with ${GEMMA_MODEL}..."
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
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --host 0.0.0.0 --port "${GEMMA_PORT}"

# Wait for healthy. 30-min budget — bigger weights download + possible
# extra torch.compile passes for the PLE / MoE paths.
echo ""
echo "→ waiting for 'Application startup complete' in container logs..."
echo "  (first run includes ~52 GB BF16 download — be patient)"
echo ""
DEADLINE=$(( $(date +%s) + 1800 ))  # 30 min
while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
    if docker logs "${CONTAINER_NAME}" 2>&1 | grep -q "Application startup complete"; then
        echo "  ✓ ready"
        break
    fi
    if docker logs "${CONTAINER_NAME}" 2>&1 | grep -qE "Error|CUDA error|out of memory|Traceback|404 Client Error|Repository Not Found|unrecognized arguments|expert.*error|routing.*error"; then
        echo ""
        echo "  ✗ container errored during startup. Last 60 log lines:"
        docker logs --tail 60 "${CONTAINER_NAME}" 2>&1
        echo ""
        echo "  Common causes:"
        echo "   - OOM at startup: drop MAX_LEN to 16384, then GPU_UTIL to 0.7."
        echo "   - 'Repository Not Found': verify the model is publicly accessible"
        echo "     and (if gated) HF_TOKEN is set in the container env."
        echo "   - MoE / expert routing errors: vLLM Gemma 4 MoE support may be"
        echo "     incomplete on this image tag. Try a pinned older or newer tag."
        echo "   - PLE / custom modeling errors: --trust-remote-code is already on;"
        echo "     check that the HF repo includes the modeling_*.py files."
        exit 1
    fi
    sleep 15
    elapsed=$(( 1800 - (DEADLINE - $(date +%s)) ))
    last=$(docker logs --tail 1 "${CONTAINER_NAME}" 2>&1 | tr -d '\r' | cut -c1-100)
    echo "  [${elapsed}s] ${last}"
done

if [ "$(date +%s)" -ge "${DEADLINE}" ]; then
    echo ""
    echo "  ✗ 30-minute startup budget exceeded. Last 60 log lines:"
    docker logs --tail 60 "${CONTAINER_NAME}" 2>&1
    exit 1
fi

# Smoke tests.
echo ""
echo "→ smoke-test: GET /v1/models"
curl -sS --max-time 5 "http://localhost:${GEMMA_PORT}/v1/models" | python3 -m json.tool 2>&1 | head -20

echo ""
echo "→ smoke-test: tiny chat completion"
printf '%s' "{\"model\":\"${GEMMA_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply only with the word OK.\"}],\"max_tokens\":10}" > /tmp/req_smoke.json
curl -sS --max-time 60 "http://localhost:${GEMMA_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d @/tmp/req_smoke.json | python3 -m json.tool 2>&1 | head -30

echo ""
echo "=== done ==="
echo ""
echo "Phase A observation commands (see plan file):"
echo "  docker logs vllm-chat 2>&1 | grep -iE 'moe|expert|router|gate|a4b|active.param|per.layer' | head -40"
echo "  docker logs -f vllm-chat   # tail and watch stats logger"
echo ""
echo "To revert: bash ~/spin-up-vllm-llama70b-specdecode.sh"
