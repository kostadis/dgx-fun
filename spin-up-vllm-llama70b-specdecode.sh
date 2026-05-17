#!/usr/bin/env bash
#
# spin-up-vllm-llama70b-specdecode.sh — same as spin-up-vllm-llama70b.sh
# but with speculative decoding: a small draft model proposes K tokens,
# the 70B target verifies them in one forward pass. Output distribution
# is unchanged (zero quality loss); decode throughput should rise 1.5-3×
# on workloads where the draft is right most of the time.
#
# Also enables tool calling (--enable-auto-tool-choice + Llama 3 parser)
# so OpenAI-compatible clients that send `tool_choice: "auto"` (opencode,
# Claude-style agents, etc.) don't get a 400. Tool-call flags are inert
# for non-tool requests, so this is free for chat-only callers.
#
# WHAT THIS DOES
#   1. Stop + remove the existing vllm-chat container.
#   2. Start a vLLM container on port 8001 serving Llama 3.3 70B Instruct AWQ
#      as the target, with a small Llama 3.2 1B draft model attached.
#   3. Wait for "Application startup complete" in the container logs.
#      First run pulls ~40 GB (target) + ~2.5 GB (draft) from HF; 25-min budget.
#   4. Confirm speculative decoding actually engaged (it's possible for vLLM
#      to silently fall back to plain decoding if the flag is rejected).
#   5. Smoke-test /v1/models and a tiny chat completion.
#   6. Print the spec-decode acceptance rate from /metrics.
#
# A/B COMPARISON
#   Compare against the baseline by alternating:
#     bash spin-up-vllm-llama70b.sh            # baseline (no spec decode)
#     bash spin-up-vllm-llama70b-specdecode.sh # this script
#   …and timing the same chat completion against each.
#
# USAGE
#   scp spin-up-vllm-llama70b-specdecode.sh kostadis@192.168.1.147:~/
#   ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-llama70b-specdecode.sh'
#
# CONFIGURATION
#   LLAMA_MODEL    target HF model id; default casperhansen/llama-3.3-70b-instruct-awq
#   DRAFT_MODEL    draft HF model id; default unsloth/Llama-3.2-1B-Instruct
#                  (non-gated mirror of meta-llama/Llama-3.2-1B-Instruct — same
#                   weights, same tokenizer, no HF_TOKEN required)
#   SPEC_TOKENS    --num-speculative-tokens (K); default 5.
#                  If acceptance rate >70%, try K=7 or 8.
#                  If acceptance rate <40%, drop to K=3.
#   LLAMA_PORT     host port; default 8001 (REPLACES vllm-chat)
#   GPU_UTIL       --gpu-memory-utilization; default 0.65
#                  (up from 0.6 in the baseline script to leave room for
#                   the ~2.5 GB draft weights)
#   MAX_LEN        --max-model-len; default 65536
#
# DRAFT/TARGET COMPATIBILITY
#   Draft and target MUST share a tokenizer or spec decode is broken /
#   useless. Llama 3.2 1B and Llama 3.3 70B both use the Llama 3.1+
#   tokenizer (128k vocab) — compatible. Don't mix families
#   (e.g. Qwen draft + Llama target).
#
# TOOL CALL PARSER
#   --tool-call-parser llama3_json is correct for Llama 3.1 / 3.3 Instruct.
#   If you swap the target to a different family, change the parser:
#     Llama 3.1 / 3.3 Instruct → llama3_json
#     Llama 3.2 (1B/3B text)   → pythonic
#     Qwen2.5 Instruct         → hermes
#     Mistral Instruct         → mistral
#   If tool calls come back as plain text in the assistant message
#   (instead of populating tool_calls), the parser is wrong OR the
#   model's built-in chat template doesn't render the tool format —
#   pass --chat-template /workspace/examples/tool_chat_template_llama3.1_json.jinja
#   (path inside the vllm container) as a fallback.
#
# REVERTING
#   bash ~/spin-up-vllm-llama70b.sh    # baseline 70B, no spec decode
#
set -euo pipefail

LLAMA_MODEL="${LLAMA_MODEL:-casperhansen/llama-3.3-70b-instruct-awq}"
DRAFT_MODEL="${DRAFT_MODEL:-unsloth/Llama-3.2-1B-Instruct}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"
LLAMA_PORT="${LLAMA_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.65}"
MAX_LEN="${MAX_LEN:-65536}"
CONTAINER_NAME="vllm-chat"
IMAGE="vllm/vllm-openai:latest"

# Build the --speculative-config JSON. printf %s/%d gives us proper escaping.
SPEC_CONFIG=$(printf '{"model": "%s", "num_speculative_tokens": %d}' \
    "${DRAFT_MODEL}" "${SPEC_TOKENS}")

echo "=== spin-up-vllm-llama70b-specdecode ==="
echo "  target:    ${LLAMA_MODEL}"
echo "  draft:     ${DRAFT_MODEL}"
echo "  K:         ${SPEC_TOKENS}"
echo "  port:      ${LLAMA_PORT}"
echo "  gpu_util:  ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:   ${MAX_LEN}"
echo "  spec_cfg:  ${SPEC_CONFIG}"
echo ""

# Stop existing vllm-chat.
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "→ stopping existing ${CONTAINER_NAME}..."
    docker stop "${CONTAINER_NAME}" 2>&1 || true
    docker rm "${CONTAINER_NAME}" 2>&1 || true
fi

# GPU healthcheck.
echo "→ GPU status pre-launch:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | head -3
echo ""

# Start the new container with speculative decoding.
echo "→ starting ${CONTAINER_NAME} with spec decode (${LLAMA_MODEL} + ${DRAFT_MODEL})..."
docker run -d \
    --runtime nvidia --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${LLAMA_PORT}:${LLAMA_PORT}" \
    --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${IMAGE}" \
    "${LLAMA_MODEL}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --speculative-config "${SPEC_CONFIG}" \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --host 0.0.0.0 --port "${LLAMA_PORT}"

# Wait for healthy. 25-min budget — target download + draft download + warm-up.
echo ""
echo "→ waiting for 'Application startup complete' in container logs..."
echo "  (first run includes ~40 GB target + ~2.5 GB draft download)"
echo ""
DEADLINE=$(( $(date +%s) + 1500 ))  # 25 min
while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
    if docker logs "${CONTAINER_NAME}" 2>&1 | grep -q "Application startup complete"; then
        echo "  ✓ ready"
        break
    fi
    if docker logs "${CONTAINER_NAME}" 2>&1 | grep -qE "Error|CUDA error|out of memory|Traceback|404 Client Error|Repository Not Found|unrecognized arguments"; then
        echo ""
        echo "  ✗ container errored during startup. Last 50 log lines:"
        docker logs --tail 50 "${CONTAINER_NAME}" 2>&1
        echo ""
        echo "  Common causes:"
        echo "   - 'unrecognized arguments: --speculative-config': vLLM image is too old."
        echo "       Fix: docker pull vllm/vllm-openai:latest && re-run this script,"
        echo "       OR switch to the old flag form by replacing the --speculative-config"
        echo "       line with: --speculative-model \"${DRAFT_MODEL}\" \\"
        echo "                  --num-speculative-tokens \"${SPEC_TOKENS}\""
        echo "   - 'Repository Not Found' for draft: try meta-llama/Llama-3.2-1B-Instruct"
        echo "       with HF_TOKEN passed via '-e HF_TOKEN=...' in docker run."
        echo "   - 'tokenizer mismatch' / 'vocab_size' error: draft and target tokenizers"
        echo "       diverged. Stick to same Llama 3.1+ family."
        echo "   - OOM: lower MAX_LEN or GPU_UTIL."
        exit 1
    fi
    sleep 15
    elapsed=$(( 1500 - (DEADLINE - $(date +%s)) ))
    last=$(docker logs --tail 1 "${CONTAINER_NAME}" 2>&1 | tr -d '\r' | cut -c1-100)
    echo "  [${elapsed}s] ${last}"
done

if [ "$(date +%s)" -ge "${DEADLINE}" ]; then
    echo ""
    echo "  ✗ 25-minute startup budget exceeded. Last 50 log lines:"
    docker logs --tail 50 "${CONTAINER_NAME}" 2>&1
    exit 1
fi

# Verify spec decode actually engaged.
echo ""
echo "→ verifying spec decode is active (not silently disabled)..."
if docker logs "${CONTAINER_NAME}" 2>&1 | grep -qiE "speculative.*(enabled|config|model)"; then
    echo "  ✓ found spec decode mention in startup logs:"
    docker logs "${CONTAINER_NAME}" 2>&1 | grep -iE "speculative" | head -5 | sed 's/^/    /'
else
    echo "  ⚠ no 'speculative' lines in startup logs — spec decode may not be engaged."
    echo "    Check 'docker logs ${CONTAINER_NAME} | grep -i spec' manually."
fi

# Smoke-test endpoints.
echo ""
echo "→ smoke-test: GET /v1/models"
curl -sS --max-time 5 "http://localhost:${LLAMA_PORT}/v1/models" | python3 -m json.tool 2>&1 | head -20

echo ""
echo "→ smoke-test: tiny chat completion (warms up spec decode for the metrics check)"
curl -sS --max-time 60 "http://localhost:${LLAMA_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(cat <<EOF
{
  "model": "${LLAMA_MODEL}",
  "messages": [{"role": "user", "content": "Write one short paragraph about coffee."}],
  "max_tokens": 200
}
EOF
)" | python3 -m json.tool 2>&1 | head -40

# Check the spec-decode acceptance rate from /metrics.
echo ""
echo "→ checking spec-decode metrics on /metrics..."
SPEC_METRICS=$(curl -sS --max-time 5 "http://localhost:${LLAMA_PORT}/metrics" | grep -i "spec_decode" || true)
if [ -n "${SPEC_METRICS}" ]; then
    echo "${SPEC_METRICS}" | head -20 | sed 's/^/  /'
    echo ""
    echo "  Acceptance rate is the headline number. Target: >0.5 for any meaningful win."
else
    echo "  ⚠ no spec_decode metrics found on /metrics."
    echo "    Either spec decode isn't engaged, or this vLLM build uses different metric names."
    echo "    Try: curl -sS http://localhost:${LLAMA_PORT}/metrics | grep -i draft"
fi

echo ""
echo "=== done ==="
echo ""
echo "Spec decode A/B comparison:"
echo "  # Time a long completion against this container:"
echo "  time curl -sS http://localhost:${LLAMA_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${LLAMA_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a 500-word essay on the history of bread.\"}],\"max_tokens\":700}' >/tmp/spec.json"
echo ""
echo "  # Then revert to baseline and time the same prompt:"
echo "  bash ~/spin-up-vllm-llama70b.sh"
echo "  time curl -sS http://localhost:${LLAMA_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${LLAMA_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a 500-word essay on the history of bread.\"}],\"max_tokens\":700}' >/tmp/base.json"
echo ""
echo "  # Compare elapsed times. Spec decode should be 1.5-3× faster on prose;"
echo "  # closer to 2-3× on code-like / structured outputs."
echo ""
echo "Tune K (currently ${SPEC_TOKENS}) based on acceptance rate:"
echo "   >0.7 → SPEC_TOKENS=7 bash $0   (more aggressive, leave less on the table)"
echo "   <0.4 → SPEC_TOKENS=3 bash $0   (less wasted draft work)"
echo ""
echo "To revert to plain 70B: bash ~/spin-up-vllm-llama70b.sh"
