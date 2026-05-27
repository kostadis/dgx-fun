#!/usr/bin/env bash
#
# spin-up-vllm-gemma4-26b-moe-longctx.sh — long-context variant of the
# Gemma 4 26B MoE spin-up. Same model, same flags, same VRAM budget,
# but trades concurrency for per-session context length.
#
# WHY THIS EXISTS
#   The default spin-up-vllm-gemma4-26b-moe.sh caps --max-model-len at
#   32K, which sizes the KV pool for ~17 concurrent sessions. On a
#   single-user Spark (only kostadis hits it) that concurrency is wasted
#   and opencode in particular runs out of context room mid-session as
#   tool-call results accumulate.
#
#   This variant raises --max-model-len to 131072 (128K), which:
#     - gives opencode ~4x more room before context pressure forces
#       compact / eviction;
#     - still leaves ~4 concurrent session slots in the KV pool, plenty
#       for MemPalace + llm_wiki + opencode + one curl test;
#     - matches roughly what most coding agents expect.
#
#   Gemma 4's native context is 256K. Going past 128K halves concurrency
#   further per doubling. If you need 200K-256K, set MAX_LEN explicitly
#   and accept ~2 slots, or layer in --kv-cache-dtype fp8 (~2x KV).
#
# CAVEAT
#   Bigger context = more prefill per turn. Gemma 4 MoE on GB10 has no
#   tuned fused-MoE kernel and falls back to TRITON_ATTN; prefill is the
#   weak spot, not decode (see gemma4-26b-moe-observations.md). This
#   script fixes "I ran out of room" — it makes "each turn feels slow"
#   slightly worse. If turn latency dominates, revert to the 32K script.
#
# USAGE
#   scp spin-up-vllm-gemma4-26b-moe-longctx.sh lib-vllm-spinup.sh spark:~/
#   ssh spark 'bash ~/spin-up-vllm-gemma4-26b-moe-longctx.sh'
#
# CONFIGURATION
#   GEMMA_MODEL    HF model id; default google/gemma-4-26b-a4b-it
#   GEMMA_PORT     host port; default 8001 (REPLACES vllm-chat)
#   GPU_UTIL       --gpu-memory-utilization; default 0.75 (~96 GB of 128 GB)
#                  Drop to 0.7 if startup hits OOM.
#   MAX_LEN        --max-model-len; default 131072 (128K)
#                  65536  -> ~8 concurrent sessions
#                  131072 -> ~4 concurrent sessions (default here)
#                  204800 -> ~2-3 concurrent sessions
#                  262144 -> ~2 concurrent sessions (native max)
#
# REVERT TO 32K / HIGH-CONCURRENCY
#   bash ~/spin-up-vllm-gemma4-26b-moe.sh
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-4-26b-a4b-it}"
GEMMA_PORT="${GEMMA_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.75}"
MAX_LEN="${MAX_LEN:-131072}"
CONTAINER_NAME="vllm-chat"
IMAGE="vllm/vllm-openai:latest"

gemma4_longctx_failure_hints() {
    cat <<'EOF'
  Common causes:
   - OOM at startup: most likely the 128K KV cache. Drop MAX_LEN
     to 65536, then to 32768. If 65536 still OOMs, also drop
     GPU_UTIL to 0.7.
   - 'Repository Not Found': verify the model is publicly accessible
     and (if gated) HF_TOKEN is set in the container env.
   - MoE / expert routing errors: vLLM Gemma 4 MoE support may be
     incomplete on this image tag. Try a pinned older or newer tag.
   - PLE / custom modeling errors: --trust-remote-code is already on;
     check that the HF repo includes the modeling_*.py files.
EOF
}

vllm_load_hf_token

echo "=== spin-up-vllm-gemma4-26b-moe-longctx ==="
echo "  model:    ${GEMMA_MODEL}"
echo "  port:     ${GEMMA_PORT}"
echo "  gpu_util: ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:  ${MAX_LEN}  (long-context variant — lower concurrency)"
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

vllm_wait_ready "${CONTAINER_NAME}" 1800 \
    "Error|CUDA error|out of memory|Traceback|404 Client Error|Repository Not Found|unrecognized arguments|expert.*error|routing.*error" \
    gemma4_longctx_failure_hints

vllm_smoke_test localhost "${GEMMA_PORT}" "${GEMMA_MODEL}"

echo ""
echo "=== done ==="
echo ""
echo "Context check — confirm max_model_len matches MAX_LEN:"
echo "  curl -sS http://localhost:${GEMMA_PORT}/v1/models | python3 -m json.tool | grep -i max_model_len"
echo ""
echo "To revert to 32K / high-concurrency: bash ~/spin-up-vllm-gemma4-26b-moe.sh"
