#!/usr/bin/env bash
#
# spin-up-vllm-deepseek-r1-distill.sh — replace vllm-chat on a DGX
# Spark (typically spark2) with DeepSeek R1 Distill Qwen 32B AWQ.
# Reasoning model: emits <think>...</think> before final answers,
# routed into `reasoning_content` by the standard `deepseek_r1` parser.
#
# WHY THIS MODEL EXISTS ON SPARK2
#   spark2 is the experimental sidecar (see current-setup.md §"Two-box
#   layout"). DeepSeek R1 Distill Qwen 32B was chosen because:
#     * Reasoning trace uses the **standard** `reasoning_content` field
#       — opencode surfaces it correctly, unlike Nemotron's `nano_v3`
#       parser which routed it through a non-standard `reasoning` key
#       opencode dropped on the floor.
#     * Tool calling via the `hermes` parser (same family as Qwen3-Next
#       on spark1, well-tested in this repo).
#     * AWQ 4-bit weights (~18 GB) leave plenty of room for KV cache
#       on the 128 GB unified-memory box.
#     * llm_wiki still can't strip <think> blocks, so this model isn't
#       wired into production clients — it lives on spark2 for
#       opencode sandboxing only.
#
# WHAT THIS DOES
#   1. Stop + remove the existing vllm-chat container.
#   2. Start a vLLM container on port 8001 serving
#      casperhansen/deepseek-r1-distill-qwen-32b-awq with:
#        - --max-model-len 65536   (4× the prior 16K default; well
#                                   inside the Qwen2.5-32B 131K config
#                                   ceiling, no rope-scaling needed)
#        - --gpu-memory-utilization 0.85  (~109 GB of 128 GB; spark2
#                                          runs nothing else, so we
#                                          can afford the headroom)
#        - --max-num-seqs 4
#        - --quantization awq_marlin --dtype float16
#        - --enable-auto-tool-choice --tool-call-parser hermes
#        - --reasoning-parser deepseek_r1
#   3. Wait for "Application startup complete". First run pulls
#      ~18 GB AWQ weights; subsequent restarts are ~30–60s.
#   4. Smoke-test /v1/models and a tiny chat completion (max_tokens
#      padded for the reasoning trace).
#
# USAGE
#   scp spin-up-vllm-deepseek-r1-distill.sh lib-vllm-spinup.sh spark2:~/
#   ssh spark2 'bash ~/spin-up-vllm-deepseek-r1-distill.sh'
#
# CONFIGURATION
#   DS_MODEL    HF model id; default casperhansen/deepseek-r1-distill-qwen-32b-awq
#   DS_PORT     host port; default 8001 (REPLACES vllm-chat)
#   GPU_UTIL    --gpu-memory-utilization; default 0.85
#   MAX_LEN     --max-model-len; default 65536. Qwen2.5-32B config
#               supports up to 131072 natively; going past 65536 may
#               want explicit --rope-scaling YaRN settings.
#   MAX_SEQS    --max-num-seqs; default 4
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

DS_MODEL="${DS_MODEL:-casperhansen/deepseek-r1-distill-qwen-32b-awq}"
DS_PORT="${DS_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_LEN="${MAX_LEN:-65536}"
MAX_SEQS="${MAX_SEQS:-4}"
CONTAINER_NAME="vllm-chat"
IMAGE="vllm/vllm-openai:latest"

deepseek_failure_hints() {
    cat <<'EOF'
  Common causes:
   - OOM at startup: drop MAX_LEN (try 32768) and/or GPU_UTIL
     (0.85 → 0.75). AWQ weights are small (~18 GB) so OOM here
     almost always means KV cache too big for max_num_seqs *
     max_model_len.
   - 'max_model_len > max_position_embeddings': model config caps
     context below MAX_LEN. Add --rope-scaling with YaRN settings,
     or drop MAX_LEN below the model's native ceiling.
   - 'tool-call-parser hermes' or '--reasoning-parser deepseek_r1'
     unrecognised: vllm/vllm-openai image too old. Pull latest.
   - 'Repository Not Found': model gated. Export HF_TOKEN.
EOF
}

vllm_load_hf_token

echo "=== spin-up-vllm-deepseek-r1-distill ==="
echo "  model:     ${DS_MODEL}"
echo "  port:      ${DS_PORT}"
echo "  gpu_util:  ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:   ${MAX_LEN}"
echo "  max_seqs:  ${MAX_SEQS}"
echo ""

vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

echo "→ starting ${CONTAINER_NAME} with ${DS_MODEL}..."
docker run -d \
    --runtime nvidia --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${DS_PORT}:${DS_PORT}" \
    --ipc=host \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${IMAGE}" \
    "${DS_MODEL}" \
    --quantization awq_marlin \
    --dtype float16 \
    --max-model-len "${MAX_LEN}" \
    --max-num-seqs "${MAX_SEQS}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser deepseek_r1 \
    --host 0.0.0.0 --port "${DS_PORT}"

# 15-min budget — AWQ weights are ~18 GB; first pull faster than the
# 60 GB BF16 models. Subsequent warm restarts hit cache and finish in
# ~30–60s.
vllm_wait_ready "${CONTAINER_NAME}" 900 \
    "Traceback \(most recent call last\)|CUDA error|CUDA out of memory|RuntimeError:|ValueError:|ImportError:|AttributeError:|OSError:|out of memory|404 Client Error|Repository Not Found|^error: unrecognized arguments|max_model_len.*greater than" \
    deepseek_failure_hints \
    80

# Reasoning model: needs token budget for <think> before producing
# the final OK. 1024 is enough for trivial prompts; reasoning_content
# can run hundreds of tokens even when the answer is one word.
vllm_smoke_test localhost "${DS_PORT}" "${DS_MODEL}" 1024

echo ""
echo "=== done ==="
echo ""
echo "Next step — verify tool calling works alongside the reasoning"
echo "parser (the combination has been flaky on other reasoning models):"
echo "  MODEL=${DS_MODEL} ./test-toolcall.sh"
echo ""
echo "Reasoning trace lands in response.choices[0].message.reasoning_content"
echo "(opencode surfaces this; llm_wiki does not strip it — keep the"
echo " model on spark2 only, not wired into production clients)."
