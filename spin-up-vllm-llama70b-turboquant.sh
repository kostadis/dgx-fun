#!/usr/bin/env bash
#
# spin-up-vllm-llama70b-turboquant.sh — Llama 3.3 70B Instruct AWQ on the
# DGX Spark with the KV cache compressed by TurboQuant (turboquant_k8v4).
#
# This is the TurboQuant sibling of spin-up-vllm-llama70b.sh. SAME model,
# SAME AWQ weights. The point of this script is to test the hypothesis
# that TurboQuant is a *win* on a DENSE long-context model — the mirror
# image of the Qwen3-Next result (see turboquant-observations.md), where
# it was a wash because the hybrid arch barely uses a KV cache.
#
# WHY LLAMA 70B IS THE GOOD CASE (and Qwen3-Next wasn't)
#   Llama 3.3 70B is DENSE: 80 layers, GQA-8 heads, head_dim 128 — EVERY
#   layer carries a KV cache. So KV grows ~0.32 MB/token at fp16 and
#   becomes the binding memory constraint at long context. That is
#   exactly what TurboQuant compresses:
#       fp16 KV   ~0.32 MB/token   (baseline: spin-up-vllm-llama70b.sh
#                                   runs with no --kv-cache-dtype = fp16)
#       fp8  KV   ~0.16 MB/token   (2.0x)
#       k8v4 KV   ~0.123 MB/token  (2.6x)  <-- this script
#   At 64K that frees ~12.5 GiB per sequence; at 128K ~25 GiB. The
#   baseline script caps GPU_UTIL at 0.6 and MAX_LEN at 64K *because*
#   fp16 KV is the limiter — TurboQuant is what lets you raise both.
#
#   On the Spark specifically, decode is MEMORY-BANDWIDTH-bound
#   (~273 GB/s). Each decoded token reads weights (~40 GiB AWQ) + the
#   whole KV cache. At 128K the fp16 KV read (~40 GiB) EQUALS the weight
#   read, so halving+ the KV bytes can cut decode latency materially.
#   vLLM's published "TurboQuant is 20-34% slower" was measured on
#   datacenter GPUs (3-8 TB/s) where KV bandwidth isn't the bottleneck
#   and the dequant kernel overhead dominates. The Spark is ~12-30x more
#   bandwidth-starved, so the bytes saved are worth far more here. Whether
#   that nets out to a decode SPEEDUP (not just a capacity win) is the
#   open question this script exists to measure. PREDICTION, not fact.
#
# DEFAULTS DEMONSTRATE THE CAPACITY WIN, NOT A MATCHED A/B
#   This script defaults to GPU_UTIL=0.88 and MAX_LEN=131072 — i.e. it
#   uses the headroom TurboQuant frees (128K context / higher util than
#   the baseline's 64K / 0.6). That shows the capacity win but is NOT a
#   clean speed A/B, because three variables move at once.
#
#   For a CLEAN A/B that isolates KV dtype (the honest speed test), run
#   BOTH scripts with identical MAX_LEN and GPU_UTIL, varying only the
#   KV dtype:
#       # baseline (fp16 KV):
#       MAX_LEN=65536 GPU_UTIL=0.6 ssh spark 'bash ~/spin-up-vllm-llama70b.sh'
#       HOST=192.168.1.147 bash bench-longctx-needle.sh casperhansen/llama-3.3-70b-instruct-awq | tee llama-fp16-needle.txt
#       LENGTHS="1024 32768 65536" bash bench-prefill.sh casperhansen/llama-3.3-70b-instruct-awq | tee llama-fp16-prefill.txt
#       # then decode tok/s (see note at bottom of this header)
#
#       # turboquant (k8v4 KV), SAME ctx/util:
#       MAX_LEN=65536 GPU_UTIL=0.6 ssh spark 'bash ~/spin-up-vllm-llama70b-turboquant.sh'
#       HOST=192.168.1.147 bash bench-longctx-needle.sh casperhansen/llama-3.3-70b-instruct-awq | tee llama-k8v4-needle.txt
#       LENGTHS="1024 32768 65536" bash bench-prefill.sh casperhansen/llama-3.3-70b-instruct-awq | tee llama-k8v4-prefill.txt
#
#       diff llama-fp16-needle.txt  llama-k8v4-needle.txt   # quality: expect identical PASS
#       diff llama-fp16-prefill.txt llama-k8v4-prefill.txt  # prefill: expect TQ ~equal or slightly slower
#   The decode delta is the headline number — measure it explicitly
#   (the needle/prefill scripts are prefill-dominated). Quick decode probe:
#       curl -sS http://192.168.1.147:8001/v1/chat/completions \
#         -H 'Content-Type: application/json' \
#         -d '{"model":"casperhansen/llama-3.3-70b-instruct-awq","messages":[{"role":"user","content":"Count slowly from 1 to 400."}],"max_tokens":512,"temperature":0,"stream":false}' \
#         | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['usage'])"
#   ...divide completion_tokens by wall-clock (time the curl) for tok/s,
#   ideally after seeding a long context so the KV read actually bites.
#
# WHY vLLM v0.22.0
#   TurboQuant on dense MHA/GQA was usable from 0.20.0/0.21.0, but pin
#   0.22.0 to match the Qwen3-Next deployment and stay clear of the
#   0.21.0 CUDA-graph bug (#40880, hybrid-specific but no reason to risk
#   an older image). #41726 (large chunked continuation-prefill crash)
#   guarded with --max-num-batched-tokens 4096. Ampere bug #40124 N/A
#   (Spark is Blackwell sm_121).
#
# RISKS SPECIFIC TO THIS COMBO (untested — this is a staged experiment)
#   * AWQ weights + TurboQuant KV together: weight-quant and KV-quant are
#     independent code paths, but the combination isn't one I've verified
#     on sm_121. If it errors on the AWQ-Marlin + TurboQuant kernel
#     interaction, that's the thing to report.
#   * TurboQuant forces FlashAttention 2 (logged at startup) — on a DENSE
#     model that means EVERY attention layer drops from FA3 to FA2, a
#     bigger prefill hit than on the hybrid (where few layers carried KV).
#     This is the cost side of the trade; weigh it against the decode win.
#
# USAGE (STAGE THEN RUN MANUALLY — this file does not auto-run anything)
#   scp spin-up-vllm-llama70b-turboquant.sh lib-vllm-spinup.sh test-toolcall.sh spark:~/
#   ssh spark 'bash ~/spin-up-vllm-llama70b-turboquant.sh'
#
# CONFIGURATION (env overrides)
#   LLAMA_MODEL     HF id; default casperhansen/llama-3.3-70b-instruct-awq
#   LLAMA_PORT      host port; default 8001 (REPLACES vllm-chat)
#   IMAGE           docker image; default vllm/vllm-openai:v0.22.0-aarch64
#   GPU_UTIL        --gpu-memory-utilization; default 0.88
#                   (set 0.6 to match baseline for a clean A/B)
#   MAX_LEN         --max-model-len; default 131072 (128K, full native)
#                   (set 65536 to match baseline for a clean A/B)
#   MAX_SEQS        --max-num-seqs; default 8 (TurboQuant frees KV for
#                   real concurrency; baseline effectively ~3 at 64K fp16)
#   KV_CACHE_DTYPE  --kv-cache-dtype; default turboquant_k8v4
#                   (turboquant_4bit_nc / k3v4_nc / 3bit_nc for the cliff;
#                    fp8 or auto to fall back — but then use the plain script)
#   MAX_BATCHED     --max-num-batched-tokens; default 4096 (#41726 guard)
#   TOOL_PARSER     --tool-call-parser; default llama3_json (Llama 3.3
#                   tool format). NOTE: the plain baseline script enables
#                   NO tool flags — adding them here is a deliberate
#                   production-parity improvement, not part of the KV A/B,
#                   and does not affect KV memory or decode speed.
#   ENFORCE_EAGER   set =1 to add --enforce-eager (CUDA-graph fallback)
#
# REVERT TO BASELINE (fp16 KV) or another slot
#   bash ~/spin-up-vllm-llama70b.sh                 # Llama, plain fp16 KV
#   bash ~/spin-up-vllm-qwen3-next-80b-turboquant.sh # back to current prod
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

LLAMA_MODEL="${LLAMA_MODEL:-casperhansen/llama-3.3-70b-instruct-awq}"
LLAMA_PORT="${LLAMA_PORT:-8001}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.0-aarch64}"
GPU_UTIL="${GPU_UTIL:-0.88}"
MAX_LEN="${MAX_LEN:-131072}"
MAX_SEQS="${MAX_SEQS:-8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-turboquant_k8v4}"
MAX_BATCHED="${MAX_BATCHED:-4096}"
TOOL_PARSER="${TOOL_PARSER:-llama3_json}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
CONTAINER_NAME="vllm-chat"

EAGER_FLAG=()
if [ -n "${ENFORCE_EAGER}" ]; then
    EAGER_FLAG=(--enforce-eager)
fi

llama_turboquant_failure_hints() {
    cat <<'EOF'
  Common causes:
   - 'turboquant_k8v4' unrecognized for --kv-cache-dtype: image too old.
     Confirm IMAGE=vllm/vllm-openai:v0.22.0-aarch64.
   - AWQ + TurboQuant kernel interaction error / Marlin assert: this is
     the untested combo this experiment is probing. Capture the
     traceback. As a fallback, try KV_CACHE_DTYPE=fp8 (drops TurboQuant
     but keeps 2x KV savings) to confirm whether AWQ-weights are fine
     and it's specifically the TurboQuant path.
   - OOM at startup: turboquant KV is SMALLER than fp16/fp8, so OOM is
     unlikely; if it happens, MAX_LEN=65536 then GPU_UTIL=0.82.
   - Crash mid-prefill at long context / workspace lock: open #41726.
     Guarded with --max-num-batched-tokens 4096; if it persists lower
     MAX_BATCHED to 2048 and/or set ENFORCE_EAGER=1.
   - Degenerate/repeated output: re-run with ENFORCE_EAGER=1 to rule out
     a CUDA-graph capture issue.
   - 'Repository Not Found': the community AWQ repo name changed —
     LLAMA_MODEL=<other-awq-repo> bash $0
   - Tool calls malformed: TOOL_PARSER alternates are 'pythonic'
     (Llama 3.2 text) — but llama3_json is correct for 3.3.
EOF
}

vllm_load_hf_token

echo "=== spin-up-vllm-llama70b-turboquant ==="
echo "  model:       ${LLAMA_MODEL}"
echo "  image:       ${IMAGE}"
echo "  port:        ${LLAMA_PORT}"
echo "  gpu_util:    ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:     ${MAX_LEN}"
echo "  max_seqs:    ${MAX_SEQS}"
echo "  kv_dtype:    ${KV_CACHE_DTYPE}   <-- TurboQuant"
echo "  max_batched: ${MAX_BATCHED}   (#41726 chunked-prefill guard)"
echo "  tool_parser: ${TOOL_PARSER}"
echo "  enforce_eager: ${ENFORCE_EAGER:-0}"
echo ""
echo "  NOTE: defaults use the headroom TurboQuant frees (128K / 0.88)."
echo "  For a clean speed A/B vs spin-up-vllm-llama70b.sh, run BOTH with"
echo "  MAX_LEN=65536 GPU_UTIL=0.6 and vary only the KV dtype."
echo ""

vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

echo "→ starting ${CONTAINER_NAME} with ${LLAMA_MODEL} (TurboQuant KV)..."
docker run -d \
    --runtime nvidia --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${LLAMA_PORT}:${LLAMA_PORT}" \
    --ipc=host \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${IMAGE}" \
    "${LLAMA_MODEL}" \
    --max-model-len "${MAX_LEN}" \
    --max-num-seqs "${MAX_SEQS}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --max-num-batched-tokens "${MAX_BATCHED}" \
    "${EAGER_FLAG[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_PARSER}" \
    --host 0.0.0.0 --port "${LLAMA_PORT}"

# 20-min budget — AWQ load + torch.compile. First run also pulls weights.
vllm_wait_ready "${CONTAINER_NAME}" 1200 \
    "Traceback \(most recent call last\)|CUDA error|CUDA out of memory|RuntimeError:|ValueError:|ImportError:|AttributeError:|OSError:|NotImplementedError|out of memory|404 Client Error|Repository Not Found|^error: unrecognized arguments|not recognized|unknown architecture" \
    llama_turboquant_failure_hints \
    80

vllm_smoke_test localhost "${LLAMA_PORT}" "${LLAMA_MODEL}"

echo ""
echo "=== done ==="
echo ""
echo "Eyeball the smoke output: repeated/garbled => re-run ENFORCE_EAGER=1."
echo ""
echo "Tool-call probe:"
echo "  MODEL=${LLAMA_MODEL} HOST=localhost PORT=${LLAMA_PORT} ~/test-toolcall.sh"
echo ""
echo "Quality (long-context) — run from the laptop:"
echo "  HOST=192.168.1.147 bash bench-longctx-needle.sh ${LLAMA_MODEL}"
echo ""
echo "The headline metric is DECODE tok/s vs the fp16 baseline (see this"
echo "script's header for the matched-A/B recipe + decode probe)."
echo ""
echo "Revert to baseline fp16 KV:  bash ~/spin-up-vllm-llama70b.sh"
echo "Back to current prod:        bash ~/spin-up-vllm-qwen3-next-80b-turboquant.sh"
