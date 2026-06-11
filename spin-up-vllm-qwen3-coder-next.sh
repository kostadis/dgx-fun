#!/usr/bin/env bash
#
# spin-up-vllm-qwen3-coder-next.sh — replace vllm-chat on the DGX Spark
# with Qwen3-Coder-Next (FP8). Agentic-coding fine-tune of Qwen3-Next-80B-A3B-Base:
# same hybrid Gated DeltaNet + Gated Attention + MoE architecture, specialised via
# RL on 800K executable tasks with environment interaction.
# ~80B total / ~3B active per token. 256K native context (1M with YaRN).
# SWE-bench Verified: 74.2%.
#
# WHY THIS DIFFERS FROM Qwen3-Next-80B-A3B-Instruct/Thinking
#   * Same base architecture and weight profile — same FP8 size (~80 GB),
#     same memory constraints, same OOM fallback ladder.
#   * Tool-call-parser is `qwen3_coder`, NOT `hermes`. Qwen3-Coder-Next
#     uses a stricter tool-call format; `hermes` may produce malformed calls.
#     If calls still come back wrong, try `qwen3_xml` as a fallback.
#   * Reasoning parser: OFF by default so a plain run of this script
#     reproduces the live spark1 deployment. Qwen3-Coder-Next is a single
#     checkpoint supporting both thinking and non-thinking modes (no separate
#     -Thinking variant). OBSERVED 2026-06-10: with `--reasoning-parser qwen3`
#     ON, this checkpoint wraps its ENTIRE answer in <think>...</think> with
#     nothing after </think>, so the parser routes the whole thing into
#     `reasoning` and leaves `content` NULL — clients reading content get an
#     empty reply. With the parser OFF (the default) the raw <think> tags
#     stay in content, so clients at least see the output. Set
#     REASONING_PARSER=qwen3 to opt into thinking mode anyway (expect null
#     content on this model). See current-setup.md §3.
#   * Agentic training: fine-tuned for long-horizon tool-use tasks. Expect
#     better performance on opencode/agentic workflows vs. the general Instruct
#     variant, with possibly more verbose tool calls.
#
# WHAT THIS DOES
#   1. Stop + remove the existing vllm-chat container.
#   2. Start a vLLM container on port 8001 serving
#      Qwen/Qwen3-Coder-Next-FP8 with:
#        - --max-model-len 131072 (128K — conservative start vs 256K native)
#        - --max-num-seqs 4 (KV is the bottleneck, not compute)
#        - --gpu-memory-utilization 0.88
#        - --kv-cache-dtype fp8
#        - --enable-auto-tool-choice --tool-call-parser qwen3_coder
#        - (no --reasoning-parser by default — see note above)
#        - --trust-remote-code
#   3. Wait for "Application startup complete" — 40 min budget. First
#      run pulls ~80 GB of FP8 weights from HF.
#   4. Smoke-test /v1/models and a tiny chat completion.
#
# USAGE
#   scp spin-up-vllm-qwen3-coder-next.sh lib-vllm-spinup.sh spark:~/
#   ssh spark 'bash ~/spin-up-vllm-qwen3-coder-next.sh'
#
# CONFIGURATION
#   CODER_MODEL    HF model id; default Qwen/Qwen3-Coder-Next-FP8
#   CODER_PORT     host port; default 8001 (REPLACES vllm-chat)
#   GPU_UTIL       --gpu-memory-utilization; default 0.88 (~113 GB of 128 GB)
#                  Drop to 0.85 if OOM at startup.
#   MAX_LEN        --max-model-len; default 131072 (128K)
#                  65536  -> more concurrent slots, less per-session room
#                  131072 -> default — opencode-friendly
#                  262144 -> native max; expect 1-2 slots only
#   MAX_SEQS       --max-num-seqs; default 4
#   KV_CACHE_DTYPE --kv-cache-dtype; default "fp8"
#   TOOL_PARSER    --tool-call-parser; default "qwen3_coder"
#                  Alternate: "qwen3_xml" if qwen3_coder produces malformed calls
#   REASONING_PARSER  --reasoning-parser; default "" (OFF — matches the live
#                  spark1 deployment). Set REASONING_PARSER=qwen3 to enable
#                  thinking mode, but note that nulls `content` on this model.
#
# REVERT
#   bash ~/spin-up-vllm-qwen3-next-80b.sh   # back to Qwen3-Next-80B Instruct
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

CODER_MODEL="${CODER_MODEL:-Qwen/Qwen3-Coder-Next-FP8}"
CODER_PORT="${CODER_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.88}"
MAX_LEN="${MAX_LEN:-131072}"
MAX_SEQS="${MAX_SEQS:-4}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER-}"
CONTAINER_NAME="vllm-chat"
# Pin to 0.22.0 — Qwen3-Next CUDA-graph degenerate-output bug (#40880)
# puts all content into <think> with nothing in `content` on 0.21.0.
IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.0-aarch64}"

qwen3_coder_next_failure_hints() {
    cat <<'EOF'
  Common causes:
   - OOM at startup: same profile as Qwen3-Next-80B (~80 GB weights).
     First try MAX_LEN=65536. If still OOM, drop GPU_UTIL to 0.85, then 0.82.
   - 'tool-call-parser qwen3_coder' unrecognized: vLLM image too old.
     Pull a newer vllm/vllm-openai tag. Requires vLLM >= 0.15.0.
     Fallback: TOOL_PARSER=hermes (may produce less structured tool calls).
   - 'reasoning-parser qwen3' unrecognized: same fix — newer image.
     Or set REASONING_PARSER="" to skip it.
   - Malformed tool calls at runtime: try TOOL_PARSER=qwen3_xml
   - 'Repository Not Found': verify HF_TOKEN is exported in ~/.bashrc
   - Slow prefill: same hybrid-attention Triton-fallback risk as Qwen3-Next
     on GB10 (sm_121). Perf ceiling, not a correctness bug.
EOF
}

vllm_load_hf_token

echo "=== spin-up-vllm-qwen3-coder-next ==="
echo "  model:       ${CODER_MODEL}"
echo "  port:        ${CODER_PORT}"
echo "  gpu_util:    ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:     ${MAX_LEN}"
echo "  max_seqs:    ${MAX_SEQS}"
echo "  kv_dtype:    ${KV_CACHE_DTYPE}"
echo "  tool_parser: ${TOOL_PARSER}"
echo "  reasoning:   ${REASONING_PARSER:-<off>}"
echo ""

REASONING_ARGS=()
if [ -n "${REASONING_PARSER}" ]; then
    REASONING_ARGS=(--reasoning-parser "${REASONING_PARSER}")
fi

vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

echo "→ starting ${CONTAINER_NAME} with ${CODER_MODEL}..."
docker run -d \
    --runtime nvidia --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${CODER_PORT}:${CODER_PORT}" \
    --ipc=host \
    --restart unless-stopped \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${IMAGE}" \
    "${CODER_MODEL}" \
    --max-model-len "${MAX_LEN}" \
    --max-num-seqs "${MAX_SEQS}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_PARSER}" \
    "${REASONING_ARGS[@]}" \
    --host 0.0.0.0 --port "${CODER_PORT}"

# 40-min budget — first run pulls ~80 GB FP8 weights.
vllm_wait_ready "${CONTAINER_NAME}" 2400 \
    "Traceback \(most recent call last\)|CUDA error|CUDA out of memory|RuntimeError:|ValueError:|ImportError:|AttributeError:|OSError:|out of memory|404 Client Error|Repository Not Found|^error: unrecognized arguments|not recognized|unknown architecture" \
    qwen3_coder_next_failure_hints \
    80

vllm_smoke_test localhost "${CODER_PORT}" "${CODER_MODEL}"

echo ""
echo "=== done ==="
echo ""
echo "Context check:"
echo "  curl -sS http://localhost:${CODER_PORT}/v1/models | python3 -m json.tool | grep -i max_model_len"
echo ""
echo "Tool-call probe (recommended next step):"
echo "  MODEL=${CODER_MODEL} ./test-toolcall.sh"
echo ""
echo "If tool calls come back malformed, retry with the XML parser:"
echo "  TOOL_PARSER=qwen3_xml bash ~/spin-up-vllm-qwen3-coder-next.sh"
echo ""
echo "To revert: bash ~/spin-up-vllm-qwen3-next-80b.sh"
