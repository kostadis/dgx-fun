#!/usr/bin/env bash
#
# spin-up-vllm-qwen3-next-80b-turboquant.sh — replace vllm-chat on the
# DGX Spark with Qwen3-Next 80B A3B Instruct (FP8 weights) but with the
# KV cache compressed by TurboQuant instead of plain fp8.
#
# This is a sibling of spin-up-vllm-qwen3-next-80b.sh. SAME model, SAME
# FP8 weights, SAME 128K context, SAME hermes tool parser. The ONLY
# functional change is:
#     --kv-cache-dtype fp8   ->   --kv-cache-dtype turboquant_k8v4
# plus a newer vLLM image (v0.22.0) and a chunked-prefill safety flag.
#
# WHAT TURBOQUANT IS (and what it is NOT)
#   TurboQuant is a KV-CACHE quantization scheme (Hadamard rotation +
#   per-coordinate Lloyd-Max scalar quant for keys, uniform quant for
#   values). It does NOT re-quantize the model weights — the weights
#   stay FP8 exactly as in the plain script. So "use turboquant" only
#   means swapping the --kv-cache-dtype value. Enabled via named
#   presets passed to --kv-cache-dtype:
#     turboquant_k8v4    FP8 keys + 4-bit values   ~2.6x  +1.17% PPL  (DEFAULT)
#     turboquant_4bit_nc 4-bit keys + 4-bit values ~3.8x  +2.71% PPL
#     turboquant_k3v4_nc 3-bit keys + 4-bit values ~3.5x  +10.63% PPL
#     turboquant_3bit_nc 3-bit keys + 3-bit values ~4.9x  +20.59% PPL
#   Compression ratios/PPL are vLLM's published numbers and apply to the
#   FULL-ATTENTION layers only.
#
# WHY k8v4 IS THE DEFAULT HERE
#   It is the closest analogue to the plain-fp8 config (FP8 keys, just
#   adds 4-bit values), so a side-by-side vs spin-up-vllm-qwen3-next-80b.sh
#   isolates TurboQuant's *machinery* cost rather than conflating it with
#   an accuracy cliff. Dial up to 4bit_nc / k3v4_nc / 3bit_nc on a LATER
#   run if you want the quality-vs-compression cliff data point.
#
# WHY THIS NEEDS vLLM v0.22.0 (NOT JUST ">=0.20.0")
#   * Hybrid-model TurboQuant support (PR #39931) landed in v0.21.0 — on
#     a hybrid like Qwen3-Next only the periodic full-attention (SDPA)
#     layers get a compressed KV cache; the Gated DeltaNet recurrent
#     state stays fp16. Boundary-layer skipping is disabled for hybrids.
#   * BUT v0.21.0 has bug #40880: Qwen3-Next + TurboQuant + CUDA-graph
#     capture produces DEGENERATE OUTPUT. That was only fixed in v0.22.0
#     (CLOSED). Running this on the box's prior 0.21.0 image risks
#     silent garbage. So we pin v0.22.0.
#   * Open bug #40807 (TurboQuant + spec-decode + chunked-prefill CUDA
#     graph crash) does NOT apply — this slot runs no speculative decode.
#   * Open bug #41726 (crash on large chunked CONTINUATION prefill) CAN
#     apply at our 128K context. Mitigation baked in below:
#     --max-num-batched-tokens 4096. If it still crashes mid-prefill,
#     fall back to ENFORCE_EAGER=1 (disables CUDA graphs entirely —
#     slower decode, but sidesteps the whole CUDA-graph bug family).
#   * Ampere bug #40124 is irrelevant: the Spark is Blackwell sm_121
#     (SM >= 89), which has the FP8 path TurboQuant needs.
#
# HONEST TRADEOFF (calibration note)
#   vLLM's own study concludes plain fp8 KV is the better *default* —
#   TurboQuant trades ~20-34% throughput for memory. On Qwen3-Next most
#   layers are Gated DeltaNet (NO KV cache), so TurboQuant only shrinks
#   the few full-attention layers: small absolute memory win, full
#   compute overhead, and the Hadamard-rotation cost lands on the
#   prefill path (this box's read-heavy workload). Expect this to be a
#   touch SLOWER than plain fp8 with a modest memory saving. That's the
#   point of running it — feel the tradeoff firsthand.
#
# USAGE
#   scp spin-up-vllm-qwen3-next-80b-turboquant.sh lib-vllm-spinup.sh spark:~/
#   ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b-turboquant.sh'
#
# CONFIGURATION (env overrides)
#   QWEN_MODEL       HF id; default Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
#   QWEN_PORT        host port; default 8001 (REPLACES vllm-chat)
#   IMAGE            docker image; default vllm/vllm-openai:v0.22.0-aarch64
#   GPU_UTIL         --gpu-memory-utilization; default 0.88
#   MAX_LEN          --max-model-len; default 131072 (128K)
#   MAX_SEQS         --max-num-seqs; default 4
#   KV_CACHE_DTYPE   --kv-cache-dtype; default turboquant_k8v4
#                    (other presets listed above; "fp8"/"auto" also valid
#                    but then just use the plain script instead)
#   MAX_BATCHED      --max-num-batched-tokens; default 4096 (#41726 guard)
#   TOOL_PARSER      --tool-call-parser; default hermes
#   ENFORCE_EAGER    set =1 to add --enforce-eager (CUDA-graph fallback)
#
# REVERT TO PLAIN FP8 KV (known-good, instant)
#   bash ~/spin-up-vllm-qwen3-next-80b.sh
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
QWEN_PORT="${QWEN_PORT:-8001}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.0-aarch64}"
GPU_UTIL="${GPU_UTIL:-0.88}"
MAX_LEN="${MAX_LEN:-131072}"
MAX_SEQS="${MAX_SEQS:-4}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-turboquant_k8v4}"
MAX_BATCHED="${MAX_BATCHED:-4096}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
CONTAINER_NAME="vllm-chat"

EAGER_FLAG=()
if [ -n "${ENFORCE_EAGER}" ]; then
    EAGER_FLAG=(--enforce-eager)
fi

turboquant_failure_hints() {
    cat <<'EOF'
  Common causes:
   - 'turboquant_k8v4' (or other preset) unrecognized for --kv-cache-dtype:
     vLLM image too old. TurboQuant hybrid support is v0.21.0+, and the
     Qwen3-Next degenerate-output fix is v0.22.0+. Confirm the image is
     vllm/vllm-openai:v0.22.0-aarch64 (NOT :latest, which may lag/lead).
   - Degenerate / repeated / garbage output but no crash: this was bug
     #40880 (CUDA-graph capture) — supposed to be fixed in v0.22.0. If
     you see it anyway, re-run with ENFORCE_EAGER=1 to disable CUDA
     graphs and confirm whether it's a graph-capture regression.
   - Crash mid-prefill at long context / "query_start_loc" / workspace
     lock: open bug #41726 (large chunked continuation prefill). Already
     guarded with --max-num-batched-tokens 4096; if it persists, lower
     MAX_BATCHED further (2048) and/or set ENFORCE_EAGER=1.
   - OOM at startup: turboquant KV is SMALLER than fp8, so OOM is
     unlikely vs the plain script; if it happens, MAX_LEN=65536 then
     GPU_UTIL=0.85.
   - 'NotImplementedError' mentioning Mamba / GDN / linear attention:
     hybrid TurboQuant path not present — image predates PR #39931.
   - Tool calls malformed: TOOL_PARSER=qwen3_coder as the alternate.
EOF
}

vllm_load_hf_token

echo "=== spin-up-vllm-qwen3-next-80b-turboquant ==="
echo "  model:       ${QWEN_MODEL}"
echo "  image:       ${IMAGE}"
echo "  port:        ${QWEN_PORT}"
echo "  gpu_util:    ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:     ${MAX_LEN}"
echo "  max_seqs:    ${MAX_SEQS}"
echo "  kv_dtype:    ${KV_CACHE_DTYPE}   <-- TurboQuant"
echo "  max_batched: ${MAX_BATCHED}   (#41726 chunked-prefill guard)"
echo "  tool_parser: ${TOOL_PARSER}"
echo "  enforce_eager: ${ENFORCE_EAGER:-0}"
echo ""

vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

echo "→ starting ${CONTAINER_NAME} with ${QWEN_MODEL} (TurboQuant KV)..."
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
    --max-num-batched-tokens "${MAX_BATCHED}" \
    "${EAGER_FLAG[@]}" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_PARSER}" \
    --host 0.0.0.0 --port "${QWEN_PORT}"

# 40-min budget — first run on v0.22.0 recompiles even though weights
# are cached from the plain-fp8 run (same FP8 weights, shared HF cache).
vllm_wait_ready "${CONTAINER_NAME}" 2400 \
    "Traceback \(most recent call last\)|CUDA error|CUDA out of memory|RuntimeError:|ValueError:|ImportError:|AttributeError:|OSError:|NotImplementedError|out of memory|404 Client Error|Repository Not Found|^error: unrecognized arguments|not recognized|unknown architecture" \
    turboquant_failure_hints \
    80

vllm_smoke_test localhost "${QWEN_PORT}" "${QWEN_MODEL}"

echo ""
echo "=== done ==="
echo ""
echo "IMPORTANT — TurboQuant on Qwen3-Next hybrid is days-old in stable."
echo "Eyeball the smoke-test output above: if it is repeated/garbled, that"
echo "is bug #40880 resurfacing — re-run with ENFORCE_EAGER=1, and if that"
echo "fixes it, file/track a regression."
echo ""
echo "Tool-call probe (do this before trusting it for agentic use):"
echo "  MODEL=${QWEN_MODEL} ./test-toolcall.sh"
echo ""
echo "Compare KV memory vs plain fp8:"
echo "  ssh spark 'nvidia-smi --query-gpu=memory.used --format=csv'"
echo ""
echo "To revert to known-good plain fp8 KV:"
echo "  bash ~/spin-up-vllm-qwen3-next-80b.sh"
