#!/usr/bin/env bash
#
# spin-up-vllm-qwen3-embed-0.6b.sh — restore the vllm-embed slot on
# spark1 (port 8000) with Qwen3-Embedding-0.6B, replacing nomic-embed-text
# as the embedding model.
#
# WHY THIS MODEL
#   nomic-embed-text-v1.5 (137M, 768-dim, early-2024) is the current
#   embedding baseline (served via Ollama 11434 while this slot was down).
#   Qwen3-Embedding-0.6B is the current SOTA family for self-hosted
#   embeddings: decoder-based, instruction-aware, Matryoshka dims, 1024-dim
#   native, 32K context. vLLM shipped first-class Qwen3-Embedding support
#   in Q1 2026. This is the "best effort-to-payoff" upgrade — ~1.2 GB
#   weights, fits alongside the 80 GB chat model with room to spare.
#
# ⚠️ BLOCKED AS-IS (measured 2026-06-11): with vllm-chat at gpu-util 0.88,
#   only ~2.4 GB of the 128 GB UNIFIED (CPU+GPU) memory is free — this
#   container OOMs at engine init even at GPU_UTIL 0.07. To use the vLLM
#   embed slot you must FIRST free memory by dropping vllm-chat's util
#   (e.g. re-spin chat at GPU_UTIL=0.82 — the KV pool has ample headroom)
#   and re-launch chat, THEN run this. Until then, the embedder lives on
#   Ollama (`ollama pull qwen3-embedding:0.6b`, served on 11434) — same
#   layer as nomic, shares memory dynamically, no chat bounce. This script
#   is kept for the freed-util scenario / batched-throughput upgrade.
#
# SHARES THE BOX WITH vllm-chat
#   spark1's vllm-chat (Qwen3-Coder-Next FP8) already reserves
#   ~113 GB (gpu-util 0.88) of the 128 GB unified memory. This embedder
#   must take a TINY slice. GPU_UTIL default 0.07 (~9 GB) — well under the
#   ~15 GB free, with margin. Embedding models have NO autoregressive KV
#   cache (pooling/encode-only), so 9 GB is plenty for a 0.6B model.
#   vLLM uses torch.cuda.mem_get_info (not nvidia-smi, which reports N/A
#   on this GB10/WSL box) so its free-memory check works correctly here.
#
# IMPORTANT — POOLING RUNNER
#   Qwen3-Embedding-0.6B is a Qwen3 DECODER repurposed as an embedder.
#   vLLM may auto-detect it as a generative model. Force embedding mode
#   with --runner pooling (vLLM 0.22 flag; replaced the old --task embed).
#   vLLM has a registered pooler for Qwen3-Embedding (last-token / EOS
#   pooling) that activates in pooling mode. If --runner is rejected on an
#   older image, drop it and let auto-detect try (see failure hints).
#
# DIMENSION CHANGE = RE-INDEX
#   nomic = 768-dim, Qwen3-Embedding-0.6B = 1024-dim. Every corpus that
#   was indexed under nomic (mempalace/turbovecdb, llm_wiki) must be
#   RE-EMBEDDED before queries work — you cannot mix dims in one index.
#   Do the end-to-end hit@k A/B (NOT index-recall) BEFORE committing to a
#   full re-index. Qwen3-Embedding also wants an INSTRUCTION PREFIX on
#   QUERIES (not documents) — see EMBED USAGE below.
#
# USAGE — spark1 (standard path)
#   scp spin-up-vllm-qwen3-embed-0.6b.sh lib-vllm-spinup.sh spark:~/
#   ssh spark 'bash ~/spin-up-vllm-qwen3-embed-0.6b.sh'
#
# USAGE — spark2 alongside vllm-2box Ray worker (2026-06-15)
#   Only ~13 GB is free after the Ray worker; this script's default
#   GPU_UTIL=0.07 OOMs (profiling spike). Run directly with docker:
#
#   ssh spark2 'docker run -d --runtime nvidia --gpus all \
#     --name vllm-embed --entrypoint "" -p 8000:8000 --ipc=host \
#     --restart unless-stopped \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     local/vllm-ray:26.05 \
#     vllm serve Qwen/Qwen3-Embedding-0.6B \
#     --trust-remote-code --gpu-memory-utilization 0.05 \
#     --runner pooling --enforce-eager --host 0.0.0.0 --port 8000'
#
#   --entrypoint "" bypasses nvidia_entrypoint.sh (NGC image pattern).
#   --enforce-eager skips the CUDA-graph profiling spike that OOMs at 0.07.
#   --gpu-memory-utilization 0.05 gives just enough KV headroom above weights.
#
# CONFIGURATION
#   EMBED_MODEL  HF id; default Qwen/Qwen3-Embedding-0.6B
#                (4B = Qwen/Qwen3-Embedding-4B, 2560-dim, ~8 GB — bump
#                 GPU_UTIL to ~0.10 and expect it to compete more with chat)
#   EMBED_PORT   host port; default 8000 (the vllm-embed slot)
#   GPU_UTIL     --gpu-memory-utilization; default 0.07. Raise to 0.10 for
#                the 4B variant; drop to 0.05 if chat is memory-starved.
#   RUNNER       --runner; default "pooling". Set "" to let vLLM
#                auto-detect (only if the flag is rejected).
#   IMAGE        default vllm/vllm-openai:v0.22.0-aarch64 (present on spark1,
#                new enough for Qwen3-Embedding)
#
# REVERT TO nomic-embed-text (prior embed model)
#   ssh spark 'docker rm -f vllm-embed'   # back to Ollama nomic on 11434
#   # or re-create vllm-embed with nomic — see current-setup.md §2 history.
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
EMBED_PORT="${EMBED_PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.07}"
RUNNER="${RUNNER:-pooling}"
CONTAINER_NAME="vllm-embed"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.0-aarch64}"

qwen3_embed_failure_hints() {
    cat <<'EOF'
  Common causes:
   - '--runner: invalid choice' / unrecognized: the image is older than
     vLLM 0.22. Re-run with RUNNER="" to let auto-detect handle it (it
     may load Qwen3-Embedding as generative — verify the /v1/embeddings
     probe returns a vector, not a 400).
   - Loaded as a generative model (no /v1/embeddings, only /v1/completions):
     auto-detect picked the wrong runner. Force RUNNER=pooling (default).
   - OOM at startup: vllm-chat already holds ~113 GB. Drop GPU_UTIL to
     0.05. If still OOM, vllm-chat's 0.88 left too little — that's a box
     budgeting problem, not this container's fault.
   - 'Repository Not Found' / gated: Qwen3-Embedding repos are public;
     check HF reachability. HF_TOKEN is picked up automatically if set.
   - Slow first boot: ~1.2 GB FP/BF16 weight pull from HF on first run.
EOF
}

vllm_load_hf_token

echo "=== spin-up-vllm-qwen3-embed-0.6b (spark1 vllm-embed slot) ==="
echo "  model:    ${EMBED_MODEL}"
echo "  port:     ${EMBED_PORT}"
echo "  gpu_util: ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.1f", u*128}') GB — tiny, shares box with vllm-chat)"
echo "  runner:   ${RUNNER:-<auto-detect>}"
echo "  image:    ${IMAGE}"
echo ""

RUNNER_ARGS=()
if [ -n "${RUNNER}" ]; then
    RUNNER_ARGS=(--runner "${RUNNER}")
fi

vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

echo "→ starting ${CONTAINER_NAME} with ${EMBED_MODEL}..."
docker run -d \
    --runtime nvidia --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${EMBED_PORT}:${EMBED_PORT}" \
    --ipc=host \
    --restart unless-stopped \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${IMAGE}" \
    "${EMBED_MODEL}" \
    --trust-remote-code \
    --gpu-memory-utilization "${GPU_UTIL}" \
    "${RUNNER_ARGS[@]}" \
    --host 0.0.0.0 --port "${EMBED_PORT}"

# 15-min budget — small weight pull + load. Embed containers also print
# "Application startup complete" (same uvicorn layer).
vllm_wait_ready "${CONTAINER_NAME}" 900 \
    "Traceback \(most recent call last\)|CUDA error|CUDA out of memory|out of memory|RuntimeError:|ValueError:|ImportError:|OSError:|404 Client Error|Repository Not Found|invalid choice|unrecognized arguments|not recognized" \
    qwen3_embed_failure_hints \
    80

echo ""
echo "→ embeddings probe: POST /v1/embeddings (expect a 1024-dim vector)"
printf '%s' "{\"model\":\"${EMBED_MODEL}\",\"input\":\"hello world\"}" > /tmp/req_embed.json
curl -sS --max-time 30 "http://localhost:${EMBED_PORT}/v1/embeddings" \
    -H 'Content-Type: application/json' -d @/tmp/req_embed.json \
    | python3 -c "import sys,json; d=json.load(sys.stdin); v=d['data'][0]['embedding']; print('  ✓ dim:', len(v), '| first 3:', [round(x,4) for x in v[:3]])" \
    || { echo "  ✗ embeddings probe failed"; qwen3_embed_failure_hints; exit 1; }

echo ""
echo "=== done ==="
echo ""
echo "EMBED USAGE (instruction-aware — apply to QUERIES, not documents):"
echo "  query input:    'Instruct: Given a search query, retrieve relevant passages\\nQuery: <text>'"
echo "  document input: '<text>'   (no prefix)"
echo "Skipping the query instruction leaves quality on the table."
echo ""
echo "Next: end-to-end hit@k A/B vs nomic BEFORE re-indexing any corpus."
echo "To revert to nomic (Ollama 11434): ssh spark 'docker rm -f vllm-embed'"
