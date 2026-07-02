#!/usr/bin/env bash
#
# spin-up-vllm-ling-lite.sh — swap vllm-chat (port 8001) to inclusionAI/Ling-lite.
#
# WHY: a FEWER-ACTIVE MoE experiment for the pdf-translators batch job. Ling-lite
# is 16.8B total / ~2.75B active (vs Qwen3-Next-80B-A3B's ~3B active), so it
# prefills + decodes a touch faster per token. The bet is throughput; the risk is
# quality (~Qwen2.5-7B-Instruct tier, well below the 80B). SAFE TO TRY because the
# pdf-translator has a HARD validator gate (validate_adventure.py) — measure
# validated-docs/hour (batch_status.py), not raw tok/s.
#
# ⚠️ THIS TARGETS spark1's PRODUCTION vllm-chat SLOT. It stops the live
#    Qwen3-Next-80B that MemPalace / llm_wiki / CampaignGenerator / opencode
#    point at, and changes the served model id (pinned clients 404 until
#    repointed). Revert is one command (bottom of file).
#
# DEPLOY:
#   scp spin-up-vllm-ling-lite.sh lib-vllm-spinup.sh spark:~/
#   ssh spark 'bash ~/spin-up-vllm-ling-lite.sh'
#
# KEY FACTS (verified 2026-06-24):
#   - arch bailing_moe (BailingMoeForCausalLM); needs --trust-remote-code.
#   - BF16 ONLY (no FP8 checkpoint). Weights ≈ 34 GB — tiny vs the 80B's ~76 GB.
#     Do NOT --quantization fp8: this is Ling 1.0 (FP8-native training is Ling 2.0),
#     so PTQ here is an unmeasured quality risk for no memory need.
#   - FULL-ATTENTION MoE: every layer carries KV (unlike Qwen3-Next's hybrid).
#     KV pool — not weights — bounds concurrency. AFTER startup, read the engine
#     log's "GPU KV cache size" and size MAX_SEQS / MAX_LEN against it (the
#     standing KV-pool rule), don't trust the seqs default blindly.
#   - No tool-call parser wired (the render job is system+user → JSON text, no
#     function calling). Set TOOL_PARSER=... to opt in, but vLLM may not ship a
#     bailing_moe parser — verify before trusting tool-calling clients.
#
# ENV KNOBS:
#   LING_MODEL     default inclusionAI/Ling-lite (the -0415 build on main)
#   GPU_UTIL       --gpu-memory-utilization; default 0.75 (~96 GB cap; leaves
#                  host headroom AND a usable KV pool for the full-attn MoE)
#   MAX_LEN        --max-model-len; default 32768 (32K — the checkpoint's TRUE
#                  native max_position_embeddings). The model card's "128K" needs
#                  YaRN rope-scaling NOT enabled in this default-branch config;
#                  forcing it (VLLM_ALLOW_LONG_MAX_MODEL_LEN=1) on a RoPE model
#                  yields NaN/OOB past 32K — do NOT. NOTE for pdf-translators:
#                  32K < the converter's default 40K --prompt-cap, so lower
#                  --prompt-cap for this endpoint (batch_convert already routes
#                  oversized docs to the 256K box).
#   MAX_SEQS       --max-num-seqs; default 16 (throughput; revisit vs logged KV pool)
#   KV_CACHE_DTYPE --kv-cache-dtype; default fp8
#   TOOL_PARSER    --tool-call-parser; default "" (off — plain chat)
#   IMAGE          default vllm/vllm-openai:v0.22.0-aarch64 (GB10-proven tag)
#
# REVERT spark1 to the production Qwen3-Next-80B:
#   ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b.sh'

set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

LING_MODEL="${LING_MODEL:-inclusionAI/Ling-lite}"
LING_PORT="${LING_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.75}"
MAX_LEN="${MAX_LEN:-32768}"
MAX_SEQS="${MAX_SEQS:-16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
TOOL_PARSER="${TOOL_PARSER:-}"
CONTAINER_NAME="vllm-chat"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.0-aarch64}"

ling_failure_hints() {
    cat <<'EOF'
  Common causes:
   - 'BailingMoeForCausalLM'/'bailing_moe' not recognized: the aarch64 image
     build may lack this arch. Pull a newer vllm/vllm-openai tag, or confirm
     bailing_moe is in this image (Ling 1.0 support landed ~vLLM 0.7).
   - trust-remote-code prompt/abort: bailing_moe ships custom modeling code;
     --trust-remote-code is set, but a sandbox may still block the HF fetch.
   - OOM at startup: unlikely (34 GB BF16 weights), but the full-attn KV pool
     grows with MAX_LEN×MAX_SEQS. Drop MAX_SEQS first, then MAX_LEN.
   - 'fp8' kv-cache-dtype unsupported for this attn impl: set KV_CACHE_DTYPE=auto
     (BF16 KV, 2× memory — also drop MAX_LEN/MAX_SEQS).
EOF
}

# Optional tool parser. Off by default (plain chat for the render job).
TOOL_ARGS=()
if [ -n "${TOOL_PARSER}" ]; then
    TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER}")
fi

vllm_load_hf_token

echo "=== spin-up-vllm-ling-lite ==="
echo "  model:       ${LING_MODEL}"
echo "  port:        ${LING_PORT}"
echo "  gpu_util:    ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:     ${MAX_LEN}"
echo "  max_seqs:    ${MAX_SEQS}"
echo "  kv_dtype:    ${KV_CACHE_DTYPE}"
echo "  tool_parser: ${TOOL_PARSER:-<off>}"
echo "  ⚠ replaces the production Qwen3-Next-80B on this box"
echo ""

vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

echo "→ starting ${CONTAINER_NAME} with ${LING_MODEL}..."
docker run -d \
    --runtime nvidia --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${LING_PORT}:${LING_PORT}" \
    --ipc=host \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${IMAGE}" \
    "${LING_MODEL}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_LEN}" \
    --max-num-seqs "${MAX_SEQS}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --trust-remote-code \
    "${TOOL_ARGS[@]}" \
    --host 0.0.0.0 --port "${LING_PORT}"

# 40-min budget — first run pulls ~34 GB BF16 weights.
vllm_wait_ready "${CONTAINER_NAME}" 2400 \
    "Traceback \(most recent call last\)|CUDA error|CUDA out of memory|RuntimeError:|ValueError:|ImportError:|AttributeError:|OSError:|out of memory|404 Client Error|Repository Not Found|^error: unrecognized arguments|not recognized|unknown architecture" \
    ling_failure_hints \
    80

vllm_smoke_test localhost "${LING_PORT}" "${LING_MODEL}"

echo ""
echo "=== done ==="
echo ""
echo "KV pool check — size MAX_SEQS/MAX_LEN against the REAL pool, not the guess:"
echo "  ssh spark 'docker logs vllm-chat 2>&1 | grep -i \"GPU KV cache size\\|maximum concurrency\"'"
echo ""
echo "Context check:"
echo "  curl -sS http://localhost:${LING_PORT}/v1/models | python3 -m json.tool | grep -i max_model_len"
echo ""
echo "Revert to production Qwen3-Next-80B:"
echo "  ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b.sh'"
