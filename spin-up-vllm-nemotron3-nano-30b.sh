#!/usr/bin/env bash
#
# spin-up-vllm-nemotron3-nano-30b.sh — replace vllm-chat on the DGX
# Spark with NVIDIA Nemotron 3 Nano 30B A3B (BF16). Hybrid Mamba-2 +
# Transformer MoE: 23 Mamba-2 + 23 MoE + 6 attention layers, ~30B total
# / ~3.5B active per token. 256K context (1M with an env override).
#
# WHY THIS MODEL IS DIFFERENT FROM GEMMA 4 26B MoE
#   * Active params: 3.5B vs 4B — close enough that prefill should be
#     in the same ballpark, but
#   * GB10 (sm_121) IS officially supported by NVIDIA's vLLM recipe for
#     this model, including a "DGX Spark / Jetson Thor" command block.
#     Gemma 4 ran on the fallback fused-MoE kernel and TRITON_ATTN; this
#     model's recipe is the first one we've tried that NVIDIA tuned for
#     this exact hardware. Expect prefill to be materially better.
#   * Only 6 attention layers — the other 46 are Mamba-2 (constant
#     per-sequence state) or MoE FFN. Traditional KV cache scaling is
#     ~10× cheaper than a Llama-shaped model at the same context, which
#     is why the recipe goes straight to --max-model-len 262144 (256K).
#   * Reasoning is BAKED IN. The model emits <think>...</think> blocks
#     before final answers. The custom `nano_v3` reasoning parser
#     strips them out of `content` into a separate `reasoning_content`
#     field on the response. Clients that don't know about reasoning
#     will just see clean answers, but token costs go up vs Gemma 4
#     (you pay for the think budget even when the answer is short).
#   * Tool calling uses the `qwen3_coder` parser (NVIDIA reused it).
#     There is an open HF discussion noting tool-call + reasoning is
#     flaky in some configurations — probe before trusting it:
#       https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/discussions/3
#     A quick toolcall probe (model name patched) ought to be the first
#     thing run after this script declares ready.
#
# WHAT THIS DOES
#   1. Ensure ~/vllm-plugins/nano_v3_reasoning_parser.py exists; wget
#      it from the HF repo if not.
#   2. Stop + remove the existing vllm-chat container (Gemma 4 MoE, or
#      whatever else is sitting in the slot).
#   3. Start a vLLM container on port 8001 serving
#      nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 with:
#        - --max-model-len 262144 (256K, recipe default for DGX Spark)
#        - --max-num-seqs 8         (recipe default)
#        - --tensor-parallel-size 1
#        - --enable-auto-tool-choice --tool-call-parser qwen3_coder
#        - --reasoning-parser-plugin /plugins/nano_v3_reasoning_parser.py
#        - --reasoning-parser nano_v3
#        - --trust-remote-code
#      mounting ~/vllm-plugins:/plugins so the parser file is visible.
#   4. Wait for "Application startup complete" — 30 min budget. First
#      run pulls ~60 GB of BF16 weights from HF.
#   5. Smoke-test /v1/models and a tiny chat completion.
#
# USAGE
#   scp spin-up-vllm-nemotron3-nano-30b.sh lib-vllm-spinup.sh spark:~/
#   ssh spark 'bash ~/spin-up-vllm-nemotron3-nano-30b.sh'
#
# CONFIGURATION
#   NEMO_MODEL     HF model id; default nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
#                  Alternative: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
#                  (~30 GB weights, faster decode on Blackwell FP8 tensor
#                  cores, but FP8 quant slightly more lossy than BF16).
#                  When swapping, also flip KV_CACHE_DTYPE to "fp8".
#   NEMO_PORT      host port; default 8001 (REPLACES vllm-chat)
#   GPU_UTIL       --gpu-memory-utilization; default 0.80 (~102 GB of 128 GB)
#                  60 GB weights + KV/Mamba state. Drop to 0.75 if OOM.
#   MAX_LEN        --max-model-len; default 262144 (256K, recipe default)
#                  Bumping to 1M requires VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.
#   MAX_SEQS       --max-num-seqs; default 8 (recipe default)
#   KV_CACHE_DTYPE --kv-cache-dtype; default "auto" (BF16). Set "fp8"
#                  only with the FP8 model variant.
#
# REVERT TO GEMMA 4 26B MoE (current default before this swap)
#   bash ~/spin-up-vllm-gemma4-26b-moe-longctx.sh   # 128K variant
#   bash ~/spin-up-vllm-gemma4-26b-moe.sh           # 32K variant
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

NEMO_MODEL="${NEMO_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
NEMO_PORT="${NEMO_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.80}"
MAX_LEN="${MAX_LEN:-262144}"
MAX_SEQS="${MAX_SEQS:-8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
CONTAINER_NAME="vllm-chat"
IMAGE="vllm/vllm-openai:latest"
PLUGIN_DIR="${HOME}/vllm-plugins"
PLUGIN_FILE="nano_v3_reasoning_parser.py"
PLUGIN_URL="https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/${PLUGIN_FILE}"

nemotron_failure_hints() {
    cat <<'EOF'
  Common causes:
   - OOM at startup: drop MAX_LEN (try 131072 first) and/or
     GPU_UTIL (0.75 → 0.70). Mamba states + 256K KV may exceed
     the 0.80 budget on the first run; the recipe doesn't
     specify a util value, so this is a guess we may need to
     correct.
   - 'reasoning-parser' / 'plugin' errors: vLLM image too old
     to support --reasoning-parser-plugin. Pull a newer image
     or pin a tag that includes the plugin interface.
   - 'tool-call-parser qwen3_coder' unrecognized: same fix —
     newer vllm/vllm-openai image.
   - 'Repository Not Found': model may be gated; export HF_TOKEN
     and re-add it via -e HF_TOKEN=… to the docker run.
EOF
}

vllm_load_hf_token

echo "=== spin-up-vllm-nemotron3-nano-30b ==="
echo "  model:     ${NEMO_MODEL}"
echo "  port:      ${NEMO_PORT}"
echo "  gpu_util:  ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:   ${MAX_LEN}"
echo "  max_seqs:  ${MAX_SEQS}"
echo "  kv_dtype:  ${KV_CACHE_DTYPE}"
echo ""

# 1. Make sure the reasoning parser plugin is on the host where the
#    container can mount it.
mkdir -p "${PLUGIN_DIR}"
if [ ! -f "${PLUGIN_DIR}/${PLUGIN_FILE}" ]; then
    echo "→ downloading reasoning parser plugin..."
    wget -q --show-progress -O "${PLUGIN_DIR}/${PLUGIN_FILE}" "${PLUGIN_URL}"
fi
echo "→ plugin present: ${PLUGIN_DIR}/${PLUGIN_FILE}"
echo ""

vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

echo "→ starting ${CONTAINER_NAME} with ${NEMO_MODEL}..."
docker run -d \
    --runtime nvidia --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${NEMO_PORT}:${NEMO_PORT}" \
    --ipc=host \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v "${PLUGIN_DIR}:/plugins:ro" \
    "${IMAGE}" \
    "${NEMO_MODEL}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_LEN}" \
    --max-num-seqs "${MAX_SEQS}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin /plugins/"${PLUGIN_FILE}" \
    --reasoning-parser nano_v3 \
    --host 0.0.0.0 --port "${NEMO_PORT}"

# 30-min budget — first run pulls ~60 GB weights. Stricter error
# regex (anchored on actual exception classes) avoids matching the
# benign "plugin"/"reasoning_parser" mentions in the args-dump line.
vllm_wait_ready "${CONTAINER_NAME}" 1800 \
    "Traceback \(most recent call last\)|CUDA error|CUDA out of memory|RuntimeError:|ValueError:|ImportError:|AttributeError:|OSError:|out of memory|404 Client Error|Repository Not Found|^error: unrecognized arguments" \
    nemotron_failure_hints \
    80

# Reasoning models need budget for <think> before producing content.
vllm_smoke_test localhost "${NEMO_PORT}" "${NEMO_MODEL}" 2048

echo ""
echo "=== done ==="
echo ""
echo "NOTE: max_tokens=2048 above (not 10) because reasoning models burn"
echo "their think budget before producing the final answer; capping at"
echo "10 makes the smoke test return empty content. Inspect both the"
echo "'content' and 'reasoning_content' fields in the response."
echo ""
echo "Next step — verify tool calling works with the reasoning parser:"
echo "  MODEL=${NEMO_MODEL} ./test-toolcall.sh"
echo ""
echo "Reasoning parser known-flaky combo (HF discussion #3):"
echo "  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/discussions/3"
echo ""
echo "To revert: bash ~/spin-up-vllm-gemma4-26b-moe-longctx.sh"
