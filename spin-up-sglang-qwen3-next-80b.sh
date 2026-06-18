#!/usr/bin/env bash
#
# spin-up-sglang-qwen3-next-80b.sh — serve Qwen3-Next 80B A3B Instruct
# (FP8) on spark2 using SGLang instead of vLLM. Same model, same flags,
# DIFFERENT ENGINE. This is the experiment: A/B SGLang against vLLM for
# the identical Qwen3-Next hybrid (Gated DeltaNet + Gated Attention + MoE,
# ~80B total / ~3B active).
#
# TARGET BOX: spark2 (192.168.1.121) — the experimental sidecar, NOT
# wired into production clients. spark1 keeps its vLLM Coder-Next.
#
# WHY SGLANG, WHY HERE
#   * spark2 is the calibration box. The point of this run is to feel
#     the friction of a second serving engine on GB10 (sm_121a, aarch64)
#     and to compare SGLang's hybrid-attention path against vLLM's on the
#     exact same weights. Suboptimal-by-design is fine; the comparison is
#     the deliverable.
#   * SGLang advertises native Qwen3-Next support with a claimed 1.3-2.1x
#     decode speedup over a naive path. Whether that materialises on GB10
#     (vs a Triton fallback kernel) is exactly what we're measuring.
#
# CANNOT COEXIST WITH vLLM
#   ~80 GB of FP8 weights live in 128 GB unified memory. Two copies do
#   not fit. This script therefore REPLACES whatever is in the spark2
#   chat slot on port 8001 (currently the vllm-chat container). It stops
#   BOTH any prior vllm-chat and any prior sglang-chat before launching.
#
# IMAGE CHOICE  (verified on spark2, 2026-06-11)
#   lmsysorg/sglang:v0.5.10.post1-cu130 — the OFFICIAL SGLang image,
#   multi-arch arm64, CUDA 13.0.1, SGLang 0.5.10. This is the one that
#   actually serves Qwen3-Next FP8 on GB10. Confirmed: clean content +
#   tool calls (qwen25 parser), ~42 tok/s decode.
#
#   WHY NOT 0.5.9: scitrera/dgx-spark-sglang:0.5.9-t5 (the older
#   NVIDIA-forum stable build) loads the weights and clears the
#   attention-backend gate, then DIES in CUDA-graph capture with a
#   DeepGEMM "Unknown recipe" assertion (layout.hpp:56) — its FP8 GEMM
#   kernels lack a recipe for this MoE shape on GB10. 0.5.10's DeepGEMM
#   has it. So: use 0.5.10. (The official lmsysorg/sglang:spark tag is
#   ~7 months old and predates Qwen3-Next — don't use that one either.)
#
#   TWO GB10-SPECIFIC GATES this model trips on SGLang, both handled below:
#     1. Attention backend: SGLang auto-picks FlashInfer, which it then
#        ASSERTS is unsupported for hybrid-GDN models on Blackwell. Must
#        pin --attention-backend triton (or trtllm_mha). See
#        ATTENTION_BACKEND below. On GB10, Triton is the only path — not
#        a slow fallback, the required one.
#     2. FP8 MoE GEMM: the DeepGEMM "Unknown recipe" above — fixed by
#        moving 0.5.9 -> 0.5.10. If a future image regresses here, the
#        honest fallback is: revert to vLLM (known-good for this exact
#        model on this exact box).
#
# WHAT THIS DOES
#   1. Stop + remove any prior vllm-chat and sglang-chat container.
#   2. Start an SGLang container named `sglang-chat` on port 8001 serving
#      Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 with flags mapped 1:1 from
#      the vLLM config (see FLAG MAPPING below).
#   3. Wait for uvicorn "Application startup complete" — 40 min budget.
#      The FP8 weights are already cached on spark2; the time sink is the
#      first-run image pull (~13 GB) + model load + CUDA graph capture.
#   4. Smoke-test /v1/models and a tiny chat completion.
#
# FLAG MAPPING (vLLM -> SGLang)
#   --max-model-len 131072        -> --context-length 131072
#   --gpu-memory-utilization 0.88 -> --mem-fraction-static 0.88
#   --max-num-seqs 8              -> --max-running-requests 8
#   --kv-cache-dtype fp8          -> --kv-cache-dtype fp8_e5m2
#   --tool-call-parser hermes     -> --tool-call-parser qwen25
#   --trust-remote-code           -> --trust-remote-code
#   (--enable-auto-tool-choice has no SGLang equivalent — setting
#    --tool-call-parser enables OpenAI function-calling output.)
#
# USAGE
#   scp spin-up-sglang-qwen3-next-80b.sh lib-vllm-spinup.sh spark2:~/
#   ssh spark2 'bash ~/spin-up-sglang-qwen3-next-80b.sh'
#
# CONFIGURATION
#   QWEN_MODEL     HF model id; default Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
#   QWEN_PORT      host port; default 8001 (REPLACES the chat slot)
#   MEM_FRACTION   --mem-fraction-static; default 0.88 (~113 GB of 128 GB).
#                  Drop to 0.85 / 0.83 if startup OOMs — SGLang's memory
#                  accounting differs from vLLM's, so 0.88 may be tighter.
#   MAX_LEN        --context-length; default 131072 (128K, matches spark2
#                  vLLM). 262144 = native max; expect fewer slots.
#   MAX_SEQS       --max-running-requests; default 8 (matches the current
#                  spark2 vLLM --max-num-seqs after the seqs=8 bump).
#   KV_CACHE_DTYPE --kv-cache-dtype; default fp8_e5m2 (scale-free FP8 KV,
#                  the closest working analog to vLLM's fp8). Set "auto"
#                  for BF16 KV if FP8 KV errors or hurts quality — KV is
#                  cheap here (only the full-attn layers carry it), so
#                  BF16 KV is affordable even at 128K.
#   ATTENTION_BACKEND --attention-backend; default "triton". REQUIRED on
#                  Blackwell (GB10/sm_121): SGLang asserts that only
#                  `triton` or `trtllm_mha` are supported for hybrid GDN
#                  (Gated DeltaNet) models on Blackwell — its FlashInfer
#                  auto-pick aborts at startup. So Triton is not a slow
#                  fallback here, it's the only supported path. Try
#                  `trtllm_mha` for potentially faster decode if Triton
#                  perf disappoints.
#   TOOL_PARSER    --tool-call-parser; default "qwen25" (parses the same
#                  Hermes-style <tool_call> blocks vLLM's `hermes` parser
#                  handles). Alternate: "qwen3_coder".
#   REASONING_PARSER --reasoning-parser; default "" (off — Instruct
#                  variant, no <think>). Set "qwen3" only for a Thinking
#                  variant, so traces land in reasoning_content not content.
#   IMAGE          container image; default lmsysorg/sglang:v0.5.10.post1-cu130
#
# REVERT TO vLLM (known-good on this box)
#   ssh spark2 'docker rm -f sglang-chat'
#   scp spin-up-vllm-qwen3-next-80b.sh lib-vllm-spinup.sh spark2:~/
#   ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b.sh'
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
QWEN_PORT="${QWEN_PORT:-8001}"
MEM_FRACTION="${MEM_FRACTION:-0.88}"
MAX_LEN="${MAX_LEN:-131072}"
MAX_SEQS="${MAX_SEQS:-8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e5m2}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
TOOL_PARSER="${TOOL_PARSER:-qwen25}"
REASONING_PARSER="${REASONING_PARSER:-}"
CONTAINER_NAME="sglang-chat"
IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.10.post1-cu130}"

sglang_failure_hints() {
    cat <<'EOF'
  Common causes:
   - OOM at startup: SGLang's mem-fraction-static accounting is not
     vLLM's. First drop MEM_FRACTION=0.85, then 0.83. If still OOM with
     headroom expected, drop MAX_LEN=65536. KV is cheap on this hybrid
     arch, so the weights + CUDA graphs are the usual culprit, not KV.
   - 'kv-cache-dtype fp8_e4m3 requires scales' or FP8 KV assertion:
     this build wants calibration scales for e4m3. Use the scale-free
     KV_CACHE_DTYPE=fp8_e5m2 (the default here), or KV_CACHE_DTYPE=auto.
   - 'Qwen3NextForCausalLM' unknown / arch not registered: the image's
     SGLang/Transformers is too old for Qwen3-Next. 0.5.10.post1 has it;
     if you pinned an older IMAGE, move forward, not back.
   - DeepGEMM 'Unknown recipe' (layout.hpp) in CUDA-graph capture: FP8
     MoE GEMM kernel lacks a recipe for this shape on GB10. This is the
     0.5.9 failure — fixed in 0.5.10.post1. Use the default IMAGE.
   - 'triton or trtllm_mha backend are the only supported backends on
     Blackwell GPUs for hybrid GDN models': SGLang's FlashInfer auto-pick
     is unsupported for Qwen3-Next on GB10. Handled by the default
     ATTENTION_BACKEND=triton; if you cleared that, set it back (or try
     ATTENTION_BACKEND=trtllm_mha).
   - PTXAS / sm_121a compile error: a kernel in this path wasn't built
     for GB10. Known rough edge on Spark SGLang images for some ops.
     This is the "SGLang MoE still stabilizing" caveat biting — the
     honest move is to revert to vLLM (see header) and log it.
   - 'tool-call-parser qwen25' unrecognized: try TOOL_PARSER=qwen3_coder,
     or drop the parser to bring the server up and probe tools later.
   - Container exits immediately: check `docker logs sglang-chat` — a
     bad flag name (SGLang != vLLM flag spellings) shows here.
EOF
}

vllm_load_hf_token

echo "=== spin-up-sglang-qwen3-next-80b (spark2) ==="
echo "  engine:      SGLang (image ${IMAGE})"
echo "  model:       ${QWEN_MODEL}"
echo "  port:        ${QWEN_PORT}  (replaces the chat slot)"
echo "  mem_frac:    ${MEM_FRACTION}  (~$(awk -v u="${MEM_FRACTION}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  ctx_len:     ${MAX_LEN}"
echo "  max_seqs:    ${MAX_SEQS}"
echo "  kv_dtype:    ${KV_CACHE_DTYPE}"
echo "  attn_backend: ${ATTENTION_BACKEND}  (triton/trtllm_mha only on Blackwell for hybrid GDN)"
echo "  tool_parser: ${TOOL_PARSER}"
echo "  reasoning:   ${REASONING_PARSER:-<off>}"
echo ""

# Optional --reasoning-parser. Empty by default (Instruct variant).
REASONING_ARGS=()
if [ -n "${REASONING_PARSER}" ]; then
    REASONING_ARGS=(--reasoning-parser "${REASONING_PARSER}")
fi

# Free the chat slot: stop BOTH possible prior occupants of port 8001.
vllm_stop_container "vllm-chat"
vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

echo "→ starting ${CONTAINER_NAME} (SGLang) with ${QWEN_MODEL}..."
# --network host: SGLang's forum recipe uses host networking on Spark.
# --privileged matches the forum recipe (GB10 needs broad device access
# for some kernels). --shm-size large for tensor-parallel IPC even at TP=1.
docker run -d \
    --runtime nvidia --gpus all \
    --privileged \
    --name "${CONTAINER_NAME}" \
    --network host \
    --ipc=host \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${IMAGE}" \
    python3 -m sglang.launch_server \
    --model-path "${QWEN_MODEL}" \
    --context-length "${MAX_LEN}" \
    --max-running-requests "${MAX_SEQS}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --attention-backend "${ATTENTION_BACKEND}" \
    --tool-call-parser "${TOOL_PARSER}" \
    --trust-remote-code \
    "${REASONING_ARGS[@]}" \
    --host 0.0.0.0 --port "${QWEN_PORT}"

# 40-min budget. Weights are cached; the cost is image pull + load +
# CUDA-graph capture. SGLang's HTTP layer is uvicorn, so the readiness
# marker "Application startup complete" is the same as vLLM's. Error
# regex is tuned for SGLang's failure strings (no vLLM-only patterns).
vllm_wait_ready "${CONTAINER_NAME}" 2400 \
    "Traceback \(most recent call last\)|CUDA error|CUDA out of memory|out of memory|RuntimeError:|ValueError:|ImportError:|AttributeError:|OSError:|AssertionError|ptxas|sm_121|404 Client Error|Repository Not Found|not recognized|unrecognized arguments|is not supported|No module named" \
    sglang_failure_hints \
    100

vllm_smoke_test localhost "${QWEN_PORT}" "${QWEN_MODEL}"

echo ""
echo "=== done ==="
echo ""
echo "Context check — confirm context_length matches MAX_LEN:"
echo "  curl -sS http://localhost:${QWEN_PORT}/v1/models | python3 -m json.tool | grep -i max_model_len"
echo ""
echo "Tool-call probe (recommended next step):"
echo "  MODEL=${QWEN_MODEL} ./test-toolcall.sh"
echo ""
echo "If tool calls come back malformed, retry with the alternate parser:"
echo "  TOOL_PARSER=qwen3_coder bash ~/spin-up-sglang-qwen3-next-80b.sh"
echo ""
echo "To revert to vLLM (known-good on this box):"
echo "  ssh spark2 'docker rm -f sglang-chat'"
echo "  ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b.sh'"
