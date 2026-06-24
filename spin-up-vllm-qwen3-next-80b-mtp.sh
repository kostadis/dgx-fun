#!/usr/bin/env bash
#
# spin-up-vllm-qwen3-next-80b-mtp.sh — Qwen3-Next 80B A3B Instruct FP8 on
# the DGX Spark WITH native Multi-Token Prediction (MTP) speculative decode.
# Same model + slot as spin-up-vllm-qwen3-next-80b.sh; the only difference
# is `--speculative-config` turning on Qwen3-Next's own MTP head as the
# draft model. Intended for spark2 (experimental box) as a decode-rate A/B
# against the plain (non-MTP) build.
#
# WHY MTP (and why this is the only real decode lever on this box)
#   Decode is bandwidth-bound: ~273 GB/s LPDDR5X / ~3.5 GB read per token
#   (3B active, FP8) ≈ ~78 tok/s ceiling, ~50-60 real. No kernel/plugin
#   beats that floor — it's physics. MTP sidesteps it by drafting N future
#   tokens from the model's built-in MTP module and verifying them in ONE
#   backbone forward pass: every accepted token is a token you didn't pay
#   full bandwidth for. Effective speedup = ~(1 + accepted_per_step), so
#   the acceptance rate is the number that matters — MEASURE IT (below).
#   Expectation: ~55 -> ~80-90 tok/s if acceptance is healthy; "100+" is
#   optimistic, not a floor. MTP-1 has the highest acceptance (best for
#   single-stream latency); MTP-2 drafts further (more upside if acceptance
#   holds, less under load). This is a DECODE win only — prefill is
#   unchanged, so it helps the long-output render path, not read-heavy
#   prefill-bound jobs.
#
# CONFIRMED BEFORE WRITING THIS (2026-06-20)
#   * The FP8 checkpoint SHIPS the MTP head — `mtp.layers.0.*` present in
#     model.safetensors.index.json (3096 mtp tensors). So MTP runs on the
#     FP8 weights; no need for the ~160 GB BF16 model.
#   * vLLM method string is `qwen3_next_mtp` (official vLLM Qwen3-Next
#     recipe). The recipe pairs it with --no-enable-chunked-prefill.
#
# RISKS / THINGS TO WATCH
#   * GIBBERISH (vLLM issue #36872): a sibling FP8-A3B model produced
#     garbage + collapsing throughput with spec decode on some builds. The
#     smoke test below checks COHERENCE, not just a 200. If output is
#     garbled, drop SPEC_TOKENS to 1, then disable MTP entirely (run the
#     plain spin-up-vllm-qwen3-next-80b.sh) and report the image version.
#   * --no-enable-chunked-prefill means the prompt prefills in one pass.
#     FlashAttention keeps that O(n) in memory so 128K should hold, but if
#     startup/first-long-request OOMs, drop MAX_LEN to 65536.
#   * Spec decode + fp8 KV: not flagged incompatible by the recipe, but if
#     vLLM rejects the combo, set KV_CACHE_DTYPE=auto (doubles KV; also drop
#     MAX_LEN).
#
# WHAT THIS DOES
#   1. Stop + remove the existing vllm-chat container on the target box.
#   2. Start vLLM on port 8001 serving Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
#      with --speculative-config '{"method":"qwen3_next_mtp",
#      "num_speculative_tokens":N}' + --no-enable-chunked-prefill.
#   3. Wait for "Application startup complete" (weights cached -> ~5-10 min).
#   4. Smoke-test /v1/models and a coherence check.
#
# USAGE (target is spark2 — the experimental box)
#   scp spin-up-vllm-qwen3-next-80b-mtp.sh lib-vllm-spinup.sh spark2:~/
#   ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'
#
# CONFIGURATION
#   QWEN_MODEL     default Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
#   QWEN_PORT      host port; default 8001 (REPLACES vllm-chat)
#   GPU_UTIL       default 0.88. Drop to 0.85 if OOM (embed at 0.05 coexists
#                  on spark2).
#   MAX_LEN        default 131072 (128K). 65536 if no-chunked-prefill OOMs.
#   MAX_SEQS       default 4. MTP favors LOW concurrency — don't over-batch;
#                  acceptance and per-request speedup are best near 1-2
#                  in-flight sequences.
#   SPEC_TOKENS    --speculative-config num_speculative_tokens; default 2.
#                  Set 1 for max acceptance / single-stream latency (MTP-1).
#   KV_CACHE_DTYPE default "fp8". Set "auto" if spec decode rejects fp8 KV.
#   TOOL_PARSER    default "hermes".
#
# REVERT TO PLAIN (non-MTP) Qwen3-Next on this box
#   bash ~/spin-up-vllm-qwen3-next-80b.sh
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
QWEN_PORT="${QWEN_PORT:-8001}"
# UNIFIED MEMORY: --gpu-memory-utilization reserves a fraction of the SINGLE
# 128 GB pool shared by GPU and host. 0.88 (≈113 GB) left the host only ~15 GB
# for OS + the embed container + page cache — too little to fork sshd, which
# wedged spark2 hard enough to need a full reboot (2026-06-20). 0.80 (≈102 GB)
# leaves ~26 GB host headroom. This is the SAME fix current-setup.md records for
# the cross-box 122B OOM (0.85→0.80). Drop to 0.78 if it still starves.
# FULL REASONING (host-starvation + bandwidth-bound KV sizing, the "right number"
# derivation, the wedge signature): read ./gpu-reservation-and-kv-tradeoffs.md
# before changing GPU_UTIL, MAX_LEN, or MAX_SEQS.
GPU_UTIL="${GPU_UTIL:-0.80}"
# 256K window (native max). Safe BECAUSE chunked prefill (below) bounds the warmup
# batch to a 40K chunk. The 256K *no-chunked* attempt instead profiled a single
# 262K-token batch and WEDGED spark2 twice (host-RAM starvation, reboots,
# 2026-06-20). Validated 2026-06-20: MTP + chunked prefill IS accepted on vLLM
# 0.22.0 — 85% draft acceptance, ~56 tok/s, host stayed healthy through load.
MAX_LEN="${MAX_LEN:-262144}"
MAX_SEQS="${MAX_SEQS:-4}"
# Prefill batching. Default: chunked prefill ON with a 40K chunk — long prompts
# stream in 40K pieces, so the window (MAX_LEN) can exceed the warmup batch and the
# big context costs nothing at warmup. 8192 (vLLM default) is too small (many tiny
# chunks slow long-context prefill). Set NO_CHUNKED_PREFILL=1 to match the recipe
# literally — but then the window IS the warmup batch, so drop MAX_LEN to ~65536
# or the 256K warmup wedges the host.
NO_CHUNKED_PREFILL="${NO_CHUNKED_PREFILL:-0}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-40960}"   # chunk size when chunked prefill is on
SPEC_TOKENS="${SPEC_TOKENS:-2}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"
CONTAINER_NAME="vllm-chat"
# Pin to the GB10-proven 0.22.0 by default — it's the documented FLOOR for
# Qwen3-Next (fixes #40880 degenerate-output-under-CUDA-graph, which 0.21.0
# hits). spark2's cached :latest is stale at 0.21.0; override IMAGE to test
# a newer tag for MTP maturity.
IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.0-aarch64}"

SPEC_CONFIG="{\"method\": \"qwen3_next_mtp\", \"num_speculative_tokens\": ${SPEC_TOKENS}}"

# Prefill batching: cap the warmup/prefill batch at MAX_BATCHED_TOKENS. Chunked
# prefill stays ON (long prompts stream in MAX_BATCHED_TOKENS chunks) unless
# NO_CHUNKED_PREFILL=1. The cap is what keeps the 256K warmup from profiling a
# single 262K-token batch and starving the host.
if [ "${NO_CHUNKED_PREFILL}" = "1" ]; then
    # No chunked prefill: vLLM ties the prefill batch to max_model_len, so the
    # window IS the warmup batch. Don't cap it below max_model_len (vLLM errors).
    CHUNK_ARGS=(--no-enable-chunked-prefill)
else
    # Chunked prefill on: bound the chunk so a long prompt streams in pieces and
    # the warmup batch stays small even when the window is large.
    CHUNK_ARGS=(--max-num-batched-tokens "${MAX_BATCHED_TOKENS}")
fi

mtp_failure_hints() {
    cat <<'EOF'
  Common causes:
   - 'qwen3_next_mtp' unknown / speculative method not recognized:
     vLLM image too old for Qwen3-Next MTP. Pull a newer
     vllm/vllm-openai tag.
   - No MTP/draft weights found: the served checkpoint lacks the mtp head
     (the FP8 Instruct one HAS it — verified). Check QWEN_MODEL.
   - Gibberish output / throughput collapse (issue #36872): fp8-A3B +
     spec decode bug on some builds. Drop SPEC_TOKENS=1; if still garbled,
     revert to the plain build (bash ~/spin-up-vllm-qwen3-next-80b.sh).
   - OOM at startup: --no-enable-chunked-prefill prefills in one pass.
     Drop MAX_LEN=65536, then GPU_UTIL=0.85.
   - fp8 KV rejected with spec decode: set KV_CACHE_DTYPE=auto (and drop
     MAX_LEN to compensate for 2x KV).
   - chunked-prefill flag error: some vLLM builds want
     --enable-chunked-prefill=False instead of --no-enable-chunked-prefill.
EOF
}

vllm_load_hf_token

echo "=== spin-up-vllm-qwen3-next-80b-mtp ==="
echo "  model:       ${QWEN_MODEL}"
echo "  port:        ${QWEN_PORT}"
echo "  gpu_util:    ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:     ${MAX_LEN}"
echo "  max_seqs:    ${MAX_SEQS}"
echo "  spec_tokens: ${SPEC_TOKENS}  (MTP-${SPEC_TOKENS})"
echo "  prefill:     chunked=$([ "${NO_CHUNKED_PREFILL}" = "1" ] && echo off || echo on), batch=${MAX_BATCHED_TOKENS}"
echo "  kv_dtype:    ${KV_CACHE_DTYPE}"
echo "  tool_parser: ${TOOL_PARSER}"
echo "  spec_config: ${SPEC_CONFIG}"
echo ""

vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

echo "→ starting ${CONTAINER_NAME} with ${QWEN_MODEL} + MTP-${SPEC_TOKENS}..."
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
    --speculative-config "${SPEC_CONFIG}" \
    "${CHUNK_ARGS[@]}" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_PARSER}" \
    --host 0.0.0.0 --port "${QWEN_PORT}"

# 30-min budget — weights are cached on spark2, so this is load + compile +
# warmup, not a download. MTP adds a draft-module compile pass.
vllm_wait_ready "${CONTAINER_NAME}" 1800 \
    "Traceback \(most recent call last\)|CUDA error|CUDA out of memory|RuntimeError:|ValueError:|ImportError:|AttributeError:|OSError:|out of memory|404 Client Error|Repository Not Found|^error: unrecognized arguments|architecture unknown" \
    mtp_failure_hints \
    80

vllm_smoke_test localhost "${QWEN_PORT}" "${QWEN_MODEL}"

echo ""
echo "=== COHERENCE CHECK (issue #36872 — spec decode can emit garbage) ==="
COHERENCE=$(curl -sS "http://localhost:${QWEN_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${QWEN_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one grammatical English sentence about the ocean.\"}],\"max_tokens\":40,\"temperature\":0.2}" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || echo "<<PARSE FAILED>>")
echo "  model said: ${COHERENCE}"
echo "  -> if that is garbled/repeated tokens, MTP is producing garbage:"
echo "     SPEC_TOKENS=1 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh"
echo "     (or revert: bash ~/spin-up-vllm-qwen3-next-80b.sh)"
echo ""
echo "=== MTP ACCEPTANCE RATE (the number that determines the speedup) ==="
echo "  docker logs vllm-chat 2>&1 | grep -iE 'accept|draft|spec' | tail -20"
echo "  # or scrape /metrics:"
echo "  curl -sS http://localhost:${QWEN_PORT}/metrics | grep -iE 'spec_decode|accept|draft'"
echo ""
echo "=== done ==="
echo "Revert to plain (non-MTP): bash ~/spin-up-vllm-qwen3-next-80b.sh"
