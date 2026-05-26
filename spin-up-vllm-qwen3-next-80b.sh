#!/usr/bin/env bash
#
# spin-up-vllm-qwen3-next-80b.sh — replace vllm-chat on the DGX Spark
# with Qwen3-Next 80B A3B Instruct (FP8). Novel hybrid architecture:
# Gated DeltaNet (linear attention) + Gated Attention, MoE FFN.
# ~80B total / ~3B active per token. 256K native context (1M with YaRN).
#
# WHY THIS MODEL IS DIFFERENT FROM GEMMA 4 26B MoE
#   * Active params: 3B vs 4B — prefill should be in the same ballpark
#     OR better, since the linear-attention layers cost ~O(n) instead of
#     O(n²) at long context. The question is whether vLLM's hybrid-kernel
#     path on GB10 (sm_121) is mature enough to realize that — Qwen3-Next
#     support is newer in vLLM than Gemma 4 MoE.
#   * Most layers are Gated DeltaNet (linear attention, no KV cache).
#     Only the periodic full-attention layers carry KV. Result: KV cache
#     at 128K is materially cheaper than a Llama-shape, similar story to
#     Nemotron's Mamba-2 hybrid.
#   * FP8 weights (~80 GB) sit comfortably below 128 GB unified memory,
#     but the headroom is tight. Set GPU_UTIL=0.88 to get ~113 GB cap,
#     leaving ~33 GB for KV/state/activations. KV cache dtype is fp8 to
#     stretch that further. Drop GPU_UTIL or MAX_LEN if startup OOMs.
#   * Tool calling uses the `hermes` parser (standard Qwen3 chat
#     template). If tool calls come back malformed, try
#     TOOL_PARSER=qwen3_coder as the alternate — it's a stricter parser
#     designed for Qwen3-Coder's tool format, and may or may not be a
#     better fit for Qwen3-Next-Instruct.
#   * NO reasoning parser. This is the Instruct variant (not Thinking),
#     so no <think> blocks to strip. That was the explicit reason for
#     picking Instruct over Thinking — the Thinking variant would
#     re-trip the Nemotron-style llm_wiki failure (no <think> stripper
#     on the client side).
#
# WHAT THIS DOES
#   1. Stop + remove the existing vllm-chat container (Gemma 4 MoE, or
#      whatever else is sitting in the slot).
#   2. Start a vLLM container on port 8001 serving
#      Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 with:
#        - --max-model-len 131072 (128K — conservative start vs 256K native)
#        - --max-num-seqs 4 (KV is tight; don't over-batch)
#        - --gpu-memory-utilization 0.88
#        - --kv-cache-dtype fp8
#        - --enable-auto-tool-choice --tool-call-parser hermes
#        - --trust-remote-code
#   3. Wait for "Application startup complete" — 40 min budget. First
#      run pulls ~80 GB of FP8 weights from HF.
#   4. Smoke-test /v1/models and a tiny chat completion.
#
# USAGE
#   scp spin-up-vllm-qwen3-next-80b.sh lib-vllm-spinup.sh spark:~/
#   ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b.sh'
#
# CONFIGURATION
#   QWEN_MODEL     HF model id; default Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
#                  Alternatives:
#                    Qwen/Qwen3-Next-80B-A3B-Instruct        — BF16, ~160 GB,
#                       WILL NOT FIT on 128 GB unified. Listed for
#                       reference only.
#                    Qwen/Qwen3-Next-80B-A3B-Thinking-FP8    — reasoning
#                       variant; will leak <think> blocks to clients that
#                       don't strip them (broke llm_wiki for Nemotron).
#                       Don't pick this unless wiring up a reasoning
#                       parser too.
#   QWEN_PORT      host port; default 8001 (REPLACES vllm-chat)
#   GPU_UTIL       --gpu-memory-utilization; default 0.88 (~113 GB of 128 GB)
#                  80 GB weights + KV + activations. Drop to 0.85 if OOM.
#   MAX_LEN        --max-model-len; default 131072 (128K)
#                  65536  -> more concurrent slots, less per-session room
#                  131072 -> default — opencode-friendly
#                  262144 -> native max; expect 1-2 slots only
#   MAX_SEQS       --max-num-seqs; default 4. Raise cautiously — KV is
#                  the bottleneck, not compute.
#   KV_CACHE_DTYPE --kv-cache-dtype; default "fp8". Set "auto" to use
#                  BF16 KV (doubles KV memory; expect to also drop MAX_LEN).
#   TOOL_PARSER    --tool-call-parser; default "hermes". Alternate:
#                  "qwen3_coder" (stricter Qwen3-Coder format).
#
# REVERT TO GEMMA 4 26B MoE (current default before this swap)
#   bash ~/spin-up-vllm-gemma4-26b-moe-longctx.sh   # 128K variant
#   bash ~/spin-up-vllm-gemma4-26b-moe.sh           # 32K variant
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
QWEN_PORT="${QWEN_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.88}"
MAX_LEN="${MAX_LEN:-131072}"
MAX_SEQS="${MAX_SEQS:-4}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"
CONTAINER_NAME="vllm-chat"
IMAGE="vllm/vllm-openai:latest"

qwen3_next_failure_hints() {
    cat <<'EOF'
  Common causes:
   - OOM at startup: most likely the 88% GPU util + 128K KV. First
     try MAX_LEN=65536. If still OOM, drop GPU_UTIL to 0.85, then
     0.82. Bottom of the curve: MAX_LEN=32768 + GPU_UTIL=0.82.
   - 'Qwen3NextForCausalLM' not recognized / 'model arch unknown':
     vLLM image too old. Qwen3-Next support landed in vLLM 0.10.x.
     Pull a newer vllm/vllm-openai tag or pin to a known-good one.
   - 'fp8' kv-cache-dtype not supported: image build lacks FP8 KV
     kernels for this attention impl. Drop KV_CACHE_DTYPE=auto and
     also drop MAX_LEN to compensate for 2× KV memory.
   - 'tool-call-parser hermes' unrecognized: same fix — newer
     vllm/vllm-openai image. Or try TOOL_PARSER=qwen3_coder.
   - 'Repository Not Found': verify the FP8 repo is publicly
     accessible. If gated, ensure HF_TOKEN is exported in ~/.bashrc
     (the lib helper picks it up automatically).
   - Slow prefill but no error: Qwen3-Next's hybrid attention may
     fall through to a Triton fallback kernel on GB10 (sm_121) if
     vLLM hasn't shipped a tuned CUDA kernel for this arch + GPU
     combination yet. This is a perf ceiling, not a correctness bug.
EOF
}

vllm_load_hf_token

echo "=== spin-up-vllm-qwen3-next-80b ==="
echo "  model:       ${QWEN_MODEL}"
echo "  port:        ${QWEN_PORT}"
echo "  gpu_util:    ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:     ${MAX_LEN}"
echo "  max_seqs:    ${MAX_SEQS}"
echo "  kv_dtype:    ${KV_CACHE_DTYPE}"
echo "  tool_parser: ${TOOL_PARSER}"
echo ""

vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

echo "→ starting ${CONTAINER_NAME} with ${QWEN_MODEL}..."
docker run -d \
    --runtime nvidia --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${QWEN_PORT}:${QWEN_PORT}" \
    --ipc=host \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${IMAGE}" \
    "${QWEN_MODEL}" \
    --max-model-len "${MAX_LEN}" \
    --max-num-seqs "${MAX_SEQS}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_PARSER}" \
    --host 0.0.0.0 --port "${QWEN_PORT}"

# 40-min budget — first run pulls ~80 GB FP8 weights. Stricter error
# regex (anchored on actual exception classes) avoids matching benign
# parser/arch mentions in the args-dump line.
vllm_wait_ready "${CONTAINER_NAME}" 2400 \
    "Traceback \(most recent call last\)|CUDA error|CUDA out of memory|RuntimeError:|ValueError:|ImportError:|AttributeError:|OSError:|out of memory|404 Client Error|Repository Not Found|^error: unrecognized arguments|not recognized|unknown architecture" \
    qwen3_next_failure_hints \
    80

vllm_smoke_test localhost "${QWEN_PORT}" "${QWEN_MODEL}"

echo ""
echo "=== done ==="
echo ""
echo "Context check — confirm max_model_len matches MAX_LEN:"
echo "  curl -sS http://localhost:${QWEN_PORT}/v1/models | python3 -m json.tool | grep -i max_model_len"
echo ""
echo "Tool-call probe (recommended next step):"
echo "  MODEL=${QWEN_MODEL} ./test-toolcall.sh"
echo ""
echo "If tool calls come back malformed, retry with the alternate parser:"
echo "  TOOL_PARSER=qwen3_coder bash ~/spin-up-vllm-qwen3-next-80b.sh"
echo ""
echo "To revert: bash ~/spin-up-vllm-gemma4-26b-moe-longctx.sh"
