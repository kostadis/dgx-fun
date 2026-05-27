#!/usr/bin/env bash
#
# opencode-spark-longctx.sh — launch opencode against the long-context
# vllm-chat on the DGX Spark.
#
# CURRENT TARGET (2026-05-18)
#   vllm-chat is serving NVIDIA Nemotron 3 Nano 30B A3B BF16 at
#   --max-model-len 262144 (256K), with reasoning + tool calling on.
#   This wrapper points opencode at that deployment. To use the previous
#   Gemma 4 26B MoE @ 128K config instead, override MODEL_ID and MIN_CTX:
#
#     MODEL_ID=google/gemma-4-26b-a4b-it MIN_CTX=131072 \
#       ./opencode-spark-longctx.sh
#
#   (And on the Spark, swap vllm-chat back via
#    `bash ~/spin-up-vllm-gemma4-26b-moe-longctx.sh`.)
#
# REASONING-FIELD QUIRK
#   The custom nano_v3 reasoning parser emits the reasoning trace in a
#   field called `reasoning` (NOT the OpenAI-convention `reasoning_content`).
#   opencode's openai-compatible provider does not surface it specially,
#   so reasoning tokens just consume budget silently. Pad `max_tokens`
#   accordingly. See current-setup.md §3 for the parser details.
#
# WHAT THIS WRAPPER DOES
#   opencode's openai-compatible provider reads max_model_len from
#   GET /v1/models, so once the server is on the right model it picks
#   up the ceiling automatically. This wrapper just
#     1. confirms vllm-chat is actually serving the expected model at
#        >= MIN_CTX before launching (so you don't silently spend a
#        session on the wrong container);
#     2. exports the env vars opencode expects;
#     3. exec's opencode with any args you pass through.
#
# USAGE
#   ./opencode-spark-longctx.sh             # verify + launch opencode
#   ./opencode-spark-longctx.sh --help      # args pass through to opencode
#   source ./opencode-spark-longctx.sh      # just set env in current shell
#
# CONFIGURATION
#   SPARK_HOST    default 192.168.1.147
#   SPARK_PORT    default 8001
#   MODEL_ID      default nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.
#                 Override to point the wrapper at a different model id
#                 (e.g. google/gemma-4-26b-a4b-it on a Gemma 4 server).
#   MIN_CTX       minimum max_model_len to accept; default 262144 (256K).
#                 Set MIN_CTX=131072 for a Gemma 4 longctx server,
#                 MIN_CTX=32768 for the high-concurrency Gemma 4 variant,
#                 MIN_CTX=0 to skip the precheck entirely.
#
set -euo pipefail

source "$(dirname "$0")/lib-precheck.sh"

SPARK_HOST="${SPARK_HOST:-192.168.1.147}"
SPARK_PORT="${SPARK_PORT:-8001}"
MIN_CTX="${MIN_CTX:-262144}"
MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
ENDPOINT="http://${SPARK_HOST}:${SPARK_PORT}/v1"

# Precheck: is the server up, on the right model, and at >= MIN_CTX?
if ! precheck_vllm_endpoint "${ENDPOINT}" "${MODEL_ID}" "${MIN_CTX}"; then
    cat >&2 <<EOF

    To swap vllm-chat to the expected model on the Spark:
      # Nemotron 3 Nano 30B A3B (256K):
      ssh kostadis@${SPARK_HOST} 'bash ~/spin-up-vllm-nemotron3-nano-30b.sh'
      # Gemma 4 26B MoE longctx (128K):
      ssh kostadis@${SPARK_HOST} 'bash ~/spin-up-vllm-gemma4-26b-moe-longctx.sh'

    Or to accept whatever the server is serving (skip precheck):
      MODEL_ID='<served_id>' MIN_CTX=0 $0

    To check what's actually running:
      curl -sS ${ENDPOINT}/models | python3 -m json.tool
      ssh kostadis@${SPARK_HOST} 'docker ps | grep vllm-chat'
EOF
    exit 1
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_API_BASE="${ENDPOINT}"
export OPENAI_MODEL="${MODEL_ID}"

echo "→ env exported:"
echo "    OPENAI_API_BASE=${OPENAI_API_BASE}"
echo "    OPENAI_MODEL=${OPENAI_MODEL}"
echo "    OPENAI_API_KEY=${OPENAI_API_KEY}"
echo ""

# If sourced (not executed), leave the env in the caller's shell and return.
(return 0 2>/dev/null) && return 0

if ! command -v opencode >/dev/null 2>&1; then
    echo "  ✗ 'opencode' binary not on PATH. Install it, then re-run."
    exit 1
fi

echo "→ launching opencode $*"
exec opencode "$@"
