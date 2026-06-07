#!/usr/bin/env bash
#
# spin-up-vllm-nemotron3-super-120b.sh — replace vllm-chat on a SINGLE
# DGX Spark with NVIDIA Nemotron 3 Super 120B A12B in **NVFP4**.
#
# Hybrid Nemotron-H: interleaved Mamba-2 + latent-MoE + a few attention
# layers. ~120B total / ~12B active per token. 512 routed experts, 22
# active + 1 shared. Reasoning is baked in (emits think traces). One MTP
# layer is present in the checkpoint (self-speculative decoding) but is
# NOT enabled here — see "MTP" below.
#
# WHY THIS EXPERIMENT
#   The incumbent that "could actually code" is Qwen3.5-122B-A10B-FP8 run
#   CROSS-BOX (TP=2 over RoCE, both Sparks). This model is ~the same total
#   size but NVFP4 (4-bit experts) so the weights are ~70 GB and fit on
#   ONE box — and it's 12B-active vs Qwen's 10B-active, so the old "too
#   sparse to code" worry doesn't apply. The question this answers: can a
#   single-box NVFP4 hybrid hold the coding bar that the two-box FP8 model
#   set? Judge on CODING OUTPUT, not tok/s. See project memory
#   `project_qwen35_first_working_model`.
#
# WHY THE PIECES ARE LOW-RISK (both halves already proven on this rig)
#   * NVFP4 inference on sm_121/GB10 already runs here — the cross-box
#     `PROFILE=minimax` slot serves nvidia/MiniMax-M2.7-NVFP4 on the same
#     NGC image. NVFP4 is auto-detected from hf_quant_config.json; no
#     --quantization flag needed.
#   * Nemotron-H hybrid (Mamba-2) already runs here — spark2 serves
#     Nemotron 3 Nano 30B (same family). This script is that pattern at
#     120B/NVFP4.
#   Only the COMBINATION (NVFP4 + Nemotron-H + 120B) is new. A GB10 forum
#   thread reports the stock NGC container loading this exact checkpoint
#   on a single Spark at gpu-util 0.85, ~22 min startup, 23-65 tok/s.
#
# IMAGE
#   nvcr.io/nvidia/vllm:26.05-py3 (via the local/vllm-ray:26.05 tag that
#   is already pulled on both boxes and proven to accept a bash-c
#   `vllm serve` command in spin-up-vllm-2box-rdma.sh). NVIDIA's own vLLM
#   build → NVFP4 kernels + Nemotron-H support baked in.
#
# REASONING PARSER
#   The repo ships super_v3_reasoning_parser.py which registers the parser
#   under the name **super_v3** (class SuperV3ReasoningParser) — NOT
#   "nemotron_v3" (that's the built-in name NVIDIA's generic recipe
#   assumes). We download the plugin and pass the self-contained
#   `--reasoning-parser-plugin <file> --reasoning-parser super_v3` pair so
#   we don't depend on whether the image bundles a built-in. Same plugin
#   mechanism the Nano script uses (nano_v3).
#
#   KNOWN LEAK (same as Nano/DeepSeek): the trace lands in a `reasoning`
#   field, which opencode's openai-compat provider drops silently. Fine
#   for a coding eval (the code lands in `content`); budget max_tokens
#   ~1.5-2x. See memory `todo_nano_v3_reasoning_leak`.
#
# MTP (multi-token prediction / self-speculative decode)
#   The checkpoint has num_nextn_predict_layers=1. NOT enabled on first
#   boot — prove NVFP4 + hybrid + parser load cleanly first, then turn MTP
#   on as a separate measured step (ENABLE_MTP=1). Stacking an immature
#   spec-decode path on top of two already-new things just muddies the
#   failure signal.
#
# WHAT THIS DOES
#   1. Ensure ~/vllm-plugins/super_v3_reasoning_parser.py exists (wget it).
#   2. Stop + remove any existing vllm-chat container.
#   3. Start vLLM on port 8001 serving the NVFP4 checkpoint, TP=1.
#   4. Wait for "Application startup complete" (60-min budget; fresh run
#      pulls ~70 GB of NVFP4 weights, then ~22 min load+compile).
#   5. Smoke-test /v1/models + a tiny chat completion (max_tokens=2048,
#      reasoning model).
#
# USAGE  (run FROM THE SPARK — scp it over first)
#   scp spin-up-vllm-nemotron3-super-120b.sh lib-vllm-spinup.sh spark:~/
#   ssh spark 'bash ~/spin-up-vllm-nemotron3-super-120b.sh'
#
# CONFIGURATION (env overrides)
#   SUPER_MODEL    HF id; default nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
#   SUPER_PORT     host port; default 8001 (REPLACES vllm-chat)
#   GPU_UTIL       --gpu-memory-utilization; default 0.85 (forum-proven on GB10)
#   MAX_LEN        --max-model-len; default 131072 (128K — matches the Qwen
#                  incumbent for a fair A/B; hybrid → KV is cheap, few attn
#                  layers). Native ceiling is 1M. Drop to 32768 if OOM.
#   MAX_SEQS       --max-num-seqs; default 8
#   ENABLE_MTP     1 = add MTP self-speculative decode (phase 2); default 0
#   MTP_TOKENS     num speculative tokens when ENABLE_MTP=1; default 1
#
# REVERT TO THE WORKING CROSS-BOX QWEN3.5 CODER (run from WORKSTATION):
#   PROFILE=qwen35 ./spin-up-vllm-2box-rdma.sh
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

SUPER_MODEL="${SUPER_MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
SUPER_PORT="${SUPER_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_LEN="${MAX_LEN:-131072}"
MAX_SEQS="${MAX_SEQS:-8}"
ENABLE_MTP="${ENABLE_MTP:-0}"
MTP_TOKENS="${MTP_TOKENS:-1}"
CONTAINER_NAME="vllm-chat"
IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.05-py3}"
PLUGIN_DIR="${HOME}/vllm-plugins"
PLUGIN_FILE="super_v3_reasoning_parser.py"
PLUGIN_URL="https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/resolve/main/${PLUGIN_FILE}"

super_failure_hints() {
    cat <<'EOF'
  Common causes:
   - OOM at startup: drop MAX_LEN (try 32768) and/or GPU_UTIL (0.85 → 0.80).
     ~70 GB NVFP4 weights + Mamba state + KV should leave headroom at
     0.85 on 128 GB, but the first try is a guess.
   - 'unknown reasoning parser super_v3' or plugin import error: the
     image's vLLM is too old for --reasoning-parser-plugin, OR the plugin
     failed to download. Check ~/vllm-plugins/super_v3_reasoning_parser.py
     exists and is non-empty. As a fallback try the built-in name:
     drop the two reasoning-parser flags and use --reasoning-parser nemotron_v3.
   - 'tool-call-parser qwen3_coder' unrecognized: image too old.
   - NVFP4 / quantization kernel error on sm_121: confirm the image is the
     NGC 26.0x build (NVFP4 kernels), not vllm/vllm-openai:latest.
   - 'Repository Not Found' / gated: export HF_TOKEN before running.
EOF
}

vllm_load_hf_token

# MTP (phase 2) — vLLM speculative-config for the in-model MTP head.
MTP_FLAGS=""
if [ "${ENABLE_MTP}" = "1" ]; then
    MTP_FLAGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}"
fi

echo "=== spin-up-vllm-nemotron3-super-120b (NVFP4, single-box) ==="
echo "  model:     ${SUPER_MODEL}"
echo "  image:     ${IMAGE}"
echo "  port:      ${SUPER_PORT}"
echo "  gpu_util:  ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:   ${MAX_LEN}"
echo "  max_seqs:  ${MAX_SEQS}"
echo "  MTP:       ${ENABLE_MTP} (tokens=${MTP_TOKENS})"
echo ""

# 1. Reasoning parser plugin on the host (mounted into the container).
mkdir -p "${PLUGIN_DIR}"
if [ ! -s "${PLUGIN_DIR}/${PLUGIN_FILE}" ]; then
    echo "→ downloading reasoning parser plugin (super_v3)..."
    wget -q --show-progress -O "${PLUGIN_DIR}/${PLUGIN_FILE}" "${PLUGIN_URL}"
fi
echo "→ plugin present: ${PLUGIN_DIR}/${PLUGIN_FILE} ($(wc -c < "${PLUGIN_DIR}/${PLUGIN_FILE}") bytes)"
echo ""

vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

echo "→ starting ${CONTAINER_NAME} with ${SUPER_MODEL} (NVFP4 auto-detected)..."
# NGC image: invoke `vllm serve` explicitly via bash -c (matches the
# proven invocation in spin-up-vllm-2box-rdma.sh). NVFP4 is read from
# hf_quant_config.json — no --quantization flag.
docker run -d \
    --runtime nvidia --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${SUPER_PORT}:${SUPER_PORT}" \
    --ipc=host \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v "${PLUGIN_DIR}:/plugins:ro" \
    "${IMAGE}" \
    bash -c "vllm serve ${SUPER_MODEL} \
        --tensor-parallel-size 1 \
        --max-model-len ${MAX_LEN} \
        --max-num-seqs ${MAX_SEQS} \
        --gpu-memory-utilization ${GPU_UTIL} \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser-plugin /plugins/${PLUGIN_FILE} \
        --reasoning-parser super_v3 \
        ${MTP_FLAGS} \
        --host 0.0.0.0 --port ${SUPER_PORT}"

# 60-min budget: fresh run pulls ~70 GB NVFP4 weights, then ~22 min
# load + compile + CUDA-graph capture (forum-measured on a single Spark).
vllm_wait_ready "${CONTAINER_NAME}" 3600 \
    "Traceback \(most recent call last\)|CUDA error|CUDA out of memory|RuntimeError:|ValueError:|ImportError:|AttributeError:|OSError:|out of memory|404 Client Error|Repository Not Found|^error: unrecognized arguments|unknown reasoning parser|Unknown quantization" \
    super_failure_hints \
    80

# Reasoning model — give it think budget or content comes back empty.
vllm_smoke_test localhost "${SUPER_PORT}" "${SUPER_MODEL}" 2048

echo ""
echo "=== done ==="
echo ""
echo "Reasoning model: smoke used max_tokens=2048. Inspect BOTH 'content'"
echo "and the 'reasoning' field (leak: opencode drops 'reasoning')."
echo ""
echo "Next — verify tool calling with the reasoning parser:"
echo "  MODEL=${SUPER_MODEL} ./test-toolcall.sh"
echo ""
echo "Phase 2 — turn on MTP self-speculative decode and re-measure decode:"
echo "  ENABLE_MTP=1 bash ~/spin-up-vllm-nemotron3-super-120b.sh"
echo ""
echo "Revert to the working cross-box Qwen3.5 coder (from the WORKSTATION):"
echo "  PROFILE=qwen35 ./spin-up-vllm-2box-rdma.sh"
