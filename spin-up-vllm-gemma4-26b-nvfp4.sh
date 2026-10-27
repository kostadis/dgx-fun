#!/usr/bin/env bash
#
# spin-up-vllm-gemma4-26b-nvfp4.sh — serve Gemma 4 26B-A4B at NVFP4 (W4A4)
# on ONE Spark, with native tool calling + reasoning parser.
#
# Reproduces the XRPL Commons write-up (2026-05-04, "Gemma 4 on the DGX
# Spark: native tool calling and reasoning mode at 271 tok/s") — 51 tok/s
# single stream, 271 tok/s aggregate at 8 concurrent — against our BF16
# baseline from May (23.3 tok/s, gemma4-26b-moe-observations.md).
#
# Checkpoint: bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4 (~16.5 GB).
#
# DIFFERENCES FROM THE ARTICLE (deliberate)
#   - Image: stock vllm/vllm-openai:v0.22.0-aarch64 and NO patched gemma4.py
#     by default. v0.22.0's gemma4.py already maps the
#     experts.N.*.weight_scale keys the patch existed for. If load crashes,
#     rerun with IMAGE=vllm/vllm-openai:gemma4-cu130 PATCH=1 (the article's
#     exact combo; the patch is shipped in the HF repo).
#   - --gpu-memory-utilization 0.80, not 0.85 (feedback_gpu_util_080_default:
#     unified memory, 0.88 wedged spark2's sshd).
#   - Container name gemma4-nvfp4, so it never touches vllm-chat or the
#     DeepSeek containers. Stopping those is the caller's job.
#
# USAGE (on the box)
#   bash ~/spin-up-vllm-gemma4-26b-nvfp4.sh
#   IMAGE=vllm/vllm-openai:gemma4-cu130 PATCH=1 bash ~/spin-up-vllm-gemma4-26b-nvfp4.sh
#
# TEARDOWN
#   docker rm -f gemma4-nvfp4
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

MODEL="${MODEL:-bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4}"
SERVED="${SERVED:-google/gemma-4-26B-A4B-it}"
PORT="${PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.80}"
MAX_LEN="${MAX_LEN:-131072}"
MAX_SEQS="${MAX_SEQS:-16}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.0-aarch64}"
PATCH="${PATCH:-0}"
CONTAINER_NAME="gemma4-nvfp4"

gemma4_nvfp4_failure_hints() {
    cat <<'EOF'
  Common causes:
   - KeyError / shape mismatch on experts.*.weight_scale: stock gemma4.py
     can't load this checkpoint. Retry with
     IMAGE=vllm/vllm-openai:gemma4-cu130 PATCH=1.
   - Crash in the NVFP4 MoE kernel (vllm#39000): confirm --moe-backend marlin
     and VLLM_NVFP4_GEMM_BACKEND=marlin are both set.
   - OOM: drop MAX_LEN to 65536.
EOF
}

vllm_load_hf_token

PATCH_MOUNT=()
if [ "${PATCH}" = "1" ]; then
    snap=$(ls -d ~/.cache/huggingface/hub/models--bg-digitalservices--Gemma-4-26B-A4B-it-NVFP4/snapshots/*/ | head -1)
    PATCH_MOUNT=(-v "${snap}gemma4_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py:ro")
fi

echo "=== spin-up-vllm-gemma4-26b-nvfp4 ==="
echo "  model:    ${MODEL}  (served as ${SERVED})"
echo "  image:    ${IMAGE}  patch=${PATCH}"
echo "  port:     ${PORT}"
echo "  gpu_util: ${GPU_UTIL}  max_len: ${MAX_LEN}  max_seqs: ${MAX_SEQS}"
echo ""

vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

echo "→ starting ${CONTAINER_NAME}..."
docker run -d \
    --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${PORT}:${PORT}" \
    --ipc=host \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e VLLM_NVFP4_GEMM_BACKEND=marlin \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${PATCH_MOUNT[@]}" \
    "${IMAGE}" \
    --model "${MODEL}" \
    --served-model-name "${SERVED}" \
    --quantization modelopt \
    --moe-backend marlin \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-num-seqs "${MAX_SEQS}" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --host 0.0.0.0 --port "${PORT}"

vllm_wait_ready "${CONTAINER_NAME}" 1800 \
    "CUDA error|out of memory|Traceback|Repository Not Found|unrecognized arguments|EngineCore failed" \
    gemma4_nvfp4_failure_hints

vllm_smoke_test localhost "${PORT}" "${SERVED}" 64

echo ""
echo "=== done ==="
echo "  KV pool:  docker logs ${CONTAINER_NAME} 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency'"
echo "  teardown: docker rm -f ${CONTAINER_NAME}"
