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

SPARK_HOST="${SPARK_HOST:-192.168.1.147}"
SPARK_PORT="${SPARK_PORT:-8001}"
MIN_CTX="${MIN_CTX:-262144}"
MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
ENDPOINT="http://${SPARK_HOST}:${SPARK_PORT}/v1"

# Precheck: is the server up, on the right model, and at >= MIN_CTX?
if [ "${MIN_CTX}" -gt 0 ]; then
    echo "→ checking ${ENDPOINT}/models ..."
    models_json="$(curl -sS --max-time 5 "${ENDPOINT}/models" 2>&1 || true)"
    if ! printf '%s' "${models_json}" | grep -q '"data"'; then
        echo "  ✗ ${ENDPOINT}/models did not respond with a model list."
        echo "    response was:"
        printf '%s\n' "${models_json}" | head -5
        echo "    is vllm-chat running on the Spark? try:"
        echo "      ssh kostadis@${SPARK_HOST} 'docker ps | grep vllm-chat'"
        exit 1
    fi

    served_model="$(printf '%s' "${models_json}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
ids = [m.get("id") for m in data.get("data", [])]
print(ids[0] if ids else "")
' 2>/dev/null || true)"
    if [ "${served_model}" != "${MODEL_ID}" ]; then
        echo "  ✗ expected model ${MODEL_ID}, got '${served_model}'."
        echo "    swap vllm-chat to the matching model first. For the default"
        echo "    (Nemotron 3 Nano 30B A3B):"
        echo "      ssh kostadis@${SPARK_HOST} 'bash ~/spin-up-vllm-nemotron3-nano-30b.sh'"
        echo "    Or to use the served model, re-run with:"
        echo "      MODEL_ID='${served_model}' MIN_CTX=0 $0"
        exit 1
    fi

    ctx="$(printf '%s' "${models_json}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
items = data.get("data", [])
print(items[0].get("max_model_len", 0) if items else 0)
' 2>/dev/null || echo 0)"
    if [ "${ctx}" -lt "${MIN_CTX}" ]; then
        echo "  ✗ max_model_len=${ctx} is below MIN_CTX=${MIN_CTX}."
        echo "    the server is running a shorter-context spin-up. To get 256K:"
        echo "      ssh kostadis@${SPARK_HOST} 'bash ~/spin-up-vllm-nemotron3-nano-30b.sh'"
        echo "    or rerun this wrapper with a lower MIN_CTX to accept the current server."
        exit 1
    fi
    echo "  ✓ ${served_model} @ max_model_len=${ctx}"
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
