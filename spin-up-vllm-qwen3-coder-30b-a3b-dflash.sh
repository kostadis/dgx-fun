#!/usr/bin/env bash
#
# spin-up-vllm-qwen3-coder-30b-a3b-dflash.sh — swap vllm-chat to
# Qwen/Qwen3-Coder-30B-A3B-Instruct with DFlash block-diffusion speculative
# decoding. Step 1 of the DFlash calibration experiment (step 2 is the
# Qwen3-Coder-Next pairing, separate script).
#
# WHAT DFLASH IS (vs the qwen3_next_mtp spec decode already used on this box)
#   MTP drafts 1-2 tokens AUTOREGRESSIVELY from the target's own MTP head,
#   one at a time. DFlash instead runs a small (0.5B) diffusion drafter that
#   proposes an entire BLOCK of up to num_speculative_tokens future tokens in
#   ONE non-causal forward pass, which the target then verifies in a single
#   batched pass. Ceiling is much higher per verification step (up to 15
#   tokens here vs MTP's 2), but it's a SEPARATE checkpoint (not baked into
#   the target's weights) and needs non-causal attention support.
#
# CONFIRMED ON THIS IMAGE BEFORE WRITING THIS (2026-06-30, spark2)
#   * vllm/vllm-openai:v0.22.0-aarch64 (already cached on spark2) ships
#     dflash NATIVELY — vllm/v1/spec_decode/dflash.py and a generic
#     DFlashQwen3ForCausalLM drafter class are already installed. No git
#     clone, no pip install, no image rebuild, no vLLM downgrade.
#   * FlashAttention is present as vLLM's OWN bundled kernels
#     (vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so, _vllm_fa3_C.abi3.so) even
#     though the standalone `flash-attn` PyPI package is NOT installed —
#     don't be fooled by `pip show flash-attn` coming up empty. Select it
#     via VLLM_ATTENTION_BACKEND=FLASH_ATTN (env var, NOT a `vllm serve`
#     CLI flag on this build — the z-lab README's `--attention-backend`
#     example is not this vLLM version's syntax).
#   * dflash.py asserts attn_metadata.causal is False for every layer and
#     points at "a different attention backend, such as FlashAttention" if
#     it isn't — this is a hard runtime requirement, not a perf hint.
#
# MODEL / DRAFTER PAIR (official z-lab matched pair, not a mismatch test)
#   Target:  Qwen/Qwen3-Coder-30B-A3B-Instruct — Qwen3MoeForCausalLM, 48
#            layers, ALL full attention (no hybrid linear-attention layers
#            like the 80B-A3B you're used to — every layer here carries KV).
#            128 experts / 8 active, native max_position_embeddings 262144
#            (real 256K, no YaRN games — verified via config.json, unlike
#            the Ling-lite 32K surprise). BF16 only, no FP8 checkpoint found
#            — weights ~60 GB.
#   Drafter: z-lab/Qwen3-Coder-30B-A3B-DFlash — 0.5B, BF16, ~1 GB. vLLM
#            pulls it via --speculative-config's "model" key the same way
#            it pulls the target, through the same HF cache mount below.
#
# KV IS THE TIGHT CONSTRAINT HERE, NOT WEIGHTS (opposite of the hybrid 80B)
#   Every one of 48 layers carries KV (num_key_value_heads=4, head_dim=128):
#   ~96 KB/token at bf16 KV, ~48 KB/token at fp8 KV. A single 256K-token
#   request eats ~12-25 GB of pool depending on KV dtype. AFTER startup,
#   read the real "GPU KV cache size" from the engine log and size
#   MAX_SEQS/MAX_LEN against it — don't trust the defaults below blindly
#   (same rule Ling-lite's script and memory `feedback_size_context_by_kv_pool`
#   already established for this repo).
#
# WHAT THIS DOES
#   1. Stop + remove the existing vllm-chat container on spark2 (currently
#      the MTP-2 Qwen3-Next-80B — this REPLACES it for the duration of the
#      experiment; spark2 isn't wired into production clients, so nothing
#      downstream breaks).
#   2. Start vLLM on port 8001 serving Qwen3-Coder-30B-A3B-Instruct with
#      --speculative-config '{"method":"dflash","model":"z-lab/Qwen3-Coder-
#      30B-A3B-DFlash","num_speculative_tokens":15}' and
#      VLLM_ATTENTION_BACKEND=FLASH_ATTN.
#   3. Wait for "Application startup complete" — 40 min budget (first run
#      pulls ~60 GB target + ~1 GB drafter fresh; nothing this size is
#      cached on spark2 yet).
#   4. Smoke-test /v1/models + a coherence check, then print where to read
#      draft acceptance rate.
#
# USAGE (spark2 — the experimental box)
#   scp spin-up-vllm-qwen3-coder-30b-a3b-dflash.sh lib-vllm-spinup.sh spark2:~/
#   ssh spark2 'bash ~/spin-up-vllm-qwen3-coder-30b-a3b-dflash.sh'
#
# CONFIGURATION
#   TARGET_MODEL   default Qwen/Qwen3-Coder-30B-A3B-Instruct
#   DRAFT_MODEL    default z-lab/Qwen3-Coder-30B-A3B-DFlash
#   DFLASH_PORT    host port; default 8001 (REPLACES vllm-chat)
#   GPU_UTIL       default 0.80 (repo-wide default — 0.88 starves host RAM
#                  on unified memory, see feedback_gpu_util_080_default)
#   MAX_LEN        default 262144 (native max — real, not YaRN-gamed).
#                  Drop to 131072 or 65536 if the logged KV pool is tight.
#   MAX_SEQS       default 4 — full-attention KV is the bottleneck here,
#                  don't over-batch until the real pool is measured.
#   SPEC_TOKENS    --speculative-config num_speculative_tokens; default 15
#                  (DFlash's documented block size for this pair).
#   KV_CACHE_DTYPE default "auto" (bf16) — CONFIRMED 2026-06-30 on spark2:
#                  fp8 KV is INCOMPATIBLE with DFlash on this vLLM build.
#                  The drafter needs a non-causal attention backend
#                  (use_non_causal=True); FlashAttention is the only
#                  backend that supports non-causal at all, but it rejects
#                  fp8 KV specifically ("kv_cache_dtype not supported").
#                  Every other backend (FlashInfer, Triton, FlexAttention,
#                  TurboQuant) fails the non-causal check outright. Net:
#                  no backend satisfies non-causal + fp8 KV together, so
#                  engine init hard-fails with "No valid attention backend
#                  found for cuda". Don't set this to fp8 for any dflash run.
#   MAX_BATCHED_TOKENS  chunked-prefill cap; default 32768 (z-lab's own
#                  documented example for this pair).
#   TOOL_PARSER    default "qwen3_coder" (Coder-family tool format).
#   ATTN_BACKEND   VLLM_ATTENTION_BACKEND value; default "FLASH_ATTN".
#                  Required by dflash's non-causal drafter attention —
#                  don't unset this.
#
# REVERT spark2 to the production MTP-2 Qwen3-Next-80B:
#   ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-Coder-30B-A3B-DFlash}"
DFLASH_PORT="${DFLASH_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.80}"
MAX_LEN="${MAX_LEN:-262144}"
MAX_SEQS="${MAX_SEQS:-4}"
SPEC_TOKENS="${SPEC_TOKENS:-15}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-32768}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
ATTN_BACKEND="${ATTN_BACKEND:-FLASH_ATTN}"
CONTAINER_NAME="vllm-chat"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.0-aarch64}"

SPEC_CONFIG="{\"method\": \"dflash\", \"model\": \"${DRAFT_MODEL}\", \"num_speculative_tokens\": ${SPEC_TOKENS}}"

dflash_failure_hints() {
    cat <<'EOF'
  Common causes:
   - "does not have non-causal support" assertion in dflash.py: the
     attention backend didn't switch to FlashAttention. Confirm
     VLLM_ATTENTION_BACKEND=FLASH_ATTN reached the container (docker
     inspect vllm-chat, or check the engine startup log for which backend
     it picked).
   - "No valid attention backend found for cuda" / "kv_cache_dtype not
     supported" for FLASH_ATTN + "non-causal attention not supported" for
     everything else: you set KV_CACHE_DTYPE=fp8. fp8 KV is incompatible
     with DFlash on this vLLM build — CONFIRMED 2026-06-30. Use
     KV_CACHE_DTYPE=auto (the default in this script).
   - OOM at startup: this model's KV is full-attention (every layer), much
     heavier per token than the hybrid 80B. First try MAX_LEN=131072, then
     MAX_SEQS=2, then GPU_UTIL=0.75.
   - Drafter 404 / Repository Not Found on z-lab/Qwen3-Coder-30B-A3B-DFlash:
     verify HF_TOKEN is exported in ~/.bashrc (gated repos only — this one
     is likely public, but check the error).
   - "dflash" method not recognized: shouldn't happen on this image (already
     confirmed present) — if it does, the cached image tag drifted; re-pull
     vllm/vllm-openai:v0.22.0-aarch64 explicitly.
   - Slow / stalled download: fresh ~60 GB target + ~1 GB drafter pull, no
     cache hit on spark2. Check `docker logs -f vllm-chat` for download
     progress, not just silence.
EOF
}

vllm_load_hf_token

echo "=== spin-up-vllm-qwen3-coder-30b-a3b-dflash ==="
echo "  target:      ${TARGET_MODEL}"
echo "  drafter:     ${DRAFT_MODEL}"
echo "  port:        ${DFLASH_PORT}"
echo "  gpu_util:    ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:     ${MAX_LEN}"
echo "  max_seqs:    ${MAX_SEQS}"
echo "  spec_tokens: ${SPEC_TOKENS}  (dflash block size)"
echo "  kv_dtype:    ${KV_CACHE_DTYPE}"
echo "  attn_backend:${ATTN_BACKEND}"
echo "  tool_parser: ${TOOL_PARSER}"
echo "  spec_config: ${SPEC_CONFIG}"
echo "  ⚠ replaces spark2's MTP-2 Qwen3-Next-80B for the duration of this experiment"
echo ""

vllm_stop_container "${CONTAINER_NAME}"
vllm_gpu_healthcheck

echo "→ starting ${CONTAINER_NAME} with ${TARGET_MODEL} + DFlash..."
docker run -d \
    --runtime nvidia --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${DFLASH_PORT}:${DFLASH_PORT}" \
    --ipc=host \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e VLLM_ATTENTION_BACKEND="${ATTN_BACKEND}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${IMAGE}" \
    "${TARGET_MODEL}" \
    --max-model-len "${MAX_LEN}" \
    --max-num-seqs "${MAX_SEQS}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --speculative-config "${SPEC_CONFIG}" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_PARSER}" \
    --host 0.0.0.0 --port "${DFLASH_PORT}"

# 40-min budget — fresh ~60 GB target + ~1 GB drafter pull, nothing cached.
vllm_wait_ready "${CONTAINER_NAME}" 2400 \
    "Traceback \(most recent call last\)|CUDA error|CUDA out of memory|RuntimeError:|ValueError:|ImportError:|AttributeError:|OSError:|out of memory|404 Client Error|Repository Not Found|^error: unrecognized arguments|not recognized|unknown architecture|does not have.*non-causal support" \
    dflash_failure_hints \
    80

vllm_smoke_test localhost "${DFLASH_PORT}" "${TARGET_MODEL}"

echo ""
echo "=== COHERENCE CHECK ==="
COHERENCE=$(curl -sS "http://localhost:${DFLASH_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${TARGET_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one grammatical English sentence about the ocean.\"}],\"max_tokens\":40,\"temperature\":0.2}" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || echo "<<PARSE FAILED>>")
echo "  model said: ${COHERENCE}"
echo ""
echo "=== KV pool check — size MAX_SEQS/MAX_LEN against the REAL pool ==="
echo "  docker logs vllm-chat 2>&1 | grep -i \"GPU KV cache size\\|maximum concurrency\""
echo ""
echo "=== DFlash draft acceptance rate (the number that determines the speedup) ==="
echo "  docker logs vllm-chat 2>&1 | grep -iE 'accept|draft|spec' | tail -20"
echo "  curl -sS http://localhost:${DFLASH_PORT}/metrics | grep -iE 'spec_decode|accept|draft'"
echo ""
echo "=== done ==="
echo "Revert to production MTP-2 Qwen3-Next-80B: ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'"
