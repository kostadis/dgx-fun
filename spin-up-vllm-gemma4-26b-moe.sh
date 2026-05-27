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
#   scp spin-up-vllm-gemma4-26b-moe.sh lib-vllm-spinup.sh spark:~/
#   ssh spark 'bash ~/spin-up-vllm-gemma4-26b-moe.sh'
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

source "$(dirname "$0")/lib-vllm-spinup.sh"

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-4-26b-a4b-it}"
GEMMA_PORT="${GEMMA_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.75}"
MAX_LEN="${MAX_LEN:-32768}"
CONTAINER_NAME="vllm-chat"
IMAGE="vllm/vllm-openai:latest"

gemma4_failure_hints() {
    cat <<'EOF'
  Common causes:
   - OOM at startup: drop MAX_LEN to 16384, then GPU_UTIL to 0.7.
   - 'Repository Not Found': verify the model is publicly accessible
     and (if gated) HF_TOKEN is set in the container env.
   - MoE / expert routing errors: vLLM Gemma 4 MoE support may be
     incomplete on this image tag. Try a pinned older or newer tag.
   - PLE / custom modeling errors: --trust-remote-code is already on;
     check that the HF repo includes the modeling_*.py files.
EOF
}

vllm_load_hf_token

echo "=== spin-up-vllm-gemma4-26b-moe ==="
echo "  model:    ${GEMMA_MODEL}"
echo "  port:     ${GEMMA_PORT}"
echo "  gpu_util: ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:  ${MAX_LEN}"
echo ""

vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

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

# 30-min budget — bigger weights download + extra torch.compile passes
# for the PLE / MoE paths. Stricter error regex catches expert-routing
# errors specific to the MoE startup path.
vllm_wait_ready "${CONTAINER_NAME}" 1800 \
    "Error|CUDA error|out of memory|Traceback|404 Client Error|Repository Not Found|unrecognized arguments|expert.*error|routing.*error" \
    gemma4_failure_hints

vllm_smoke_test localhost "${GEMMA_PORT}" "${GEMMA_MODEL}"

echo ""
echo "=== done ==="
echo ""
echo "Phase A observation commands (see plan file):"
echo "  docker logs vllm-chat 2>&1 | grep -iE 'moe|expert|router|gate|a4b|active.param|per.layer' | head -40"
echo "  docker logs -f vllm-chat   # tail and watch stats logger"
echo ""
echo "To revert: bash ~/spin-up-vllm-llama70b-specdecode.sh"
