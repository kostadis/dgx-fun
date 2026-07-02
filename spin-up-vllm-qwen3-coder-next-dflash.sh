#!/usr/bin/env bash
#
# spin-up-vllm-qwen3-coder-next-dflash.sh — swap vllm-chat to Qwen3-Coder-Next
# with DFlash block-diffusion speculative decoding. Step 2 of the DFlash
# calibration experiment (step 1 was Qwen3-Coder-30B-A3B, see the sibling
# script — that run confirmed dflash works natively on this vLLM image and
# that KV_CACHE_DTYPE must be "auto", not "fp8" — see that script's header
# for the full "No valid attention backend found" postmortem).
#
# WHY THIS MODEL: same ~80B-A3B hybrid backbone (Gated DeltaNet + full
# attention + MoE) as the production Qwen3-Next-80B-A3B-Instruct this box
# normally runs — closer to a real production read on DFlash than step 1's
# full-attention 30B was. Unlike step 1, an FP8 checkpoint exists, so this
# also avoids step 1's BF16-bandwidth confound IF the FP8/drafter pairing
# works (see risk below).
#
# ⚠️ UNTESTED PAIRING: z-lab's own model card for this drafter documents
#    pairing with the BF16 base `Qwen/Qwen3-Coder-Next`, not the FP8
#    checkpoint this script defaults to. The drafter shares embed_tokens
#    and lm_head weights DIRECTLY from the target at load time (confirmed
#    in step 1's logs: "Detected EAGLE model without its own embed_tokens
#    ... Sharing target model embedding weights with the draft model") —
#    if the target's shared tensors are FP8-quantized and the drafter
#    expects BF16, this can crash at load (dtype mismatch) rather than
#    just under-perform. We're trying FP8 first because it's what
#    production actually runs and it's half the download; if it crashes
#    on a dtype/shape mismatch, fall back to the officially-documented
#    BF16 pairing (env override below).
#
# WHAT THIS DOES
#   1. Stop + remove the existing vllm-chat container on spark2.
#   2. Start vLLM on port 8001 serving Qwen3-Coder-Next-FP8 with
#      --speculative-config '{"method":"dflash","model":"z-lab/Qwen3-Coder-
#      Next-DFlash","num_speculative_tokens":15}', VLLM_ATTENTION_BACKEND=
#      FLASH_ATTN, and --kv-cache-dtype auto (fp8 KV is incompatible with
#      dflash on this vLLM build — see step 1 script header).
#   3. Wait for "Application startup complete" — 40 min budget (fresh ~80 GB
#      FP8 target pull; this exact checkpoint isn't cached on spark2, only
#      on spark1).
#   4. Smoke-test /v1/models + a coherence check, then print where to read
#      draft acceptance rate.
#
# USAGE (spark2 — the experimental box)
#   scp spin-up-vllm-qwen3-coder-next-dflash.sh lib-vllm-spinup.sh spark2:~/
#   ssh spark2 'bash ~/spin-up-vllm-qwen3-coder-next-dflash.sh'
#
# FALLBACK TO THE OFFICIALLY-DOCUMENTED BF16 PAIRING (if FP8 crashes)
#   TARGET_MODEL=Qwen/Qwen3-Coder-Next bash ~/spin-up-vllm-qwen3-coder-next-dflash.sh
#   (fresh ~160 GB download — check disk headroom first: spark2 has ~2.8 TB free)
#
# CONFIGURATION
#   TARGET_MODEL   default Qwen/Qwen3-Coder-Next-FP8 (production-matching,
#                  UNTESTED with this drafter — see risk note above)
#   DRAFT_MODEL    default z-lab/Qwen3-Coder-Next-DFlash
#   DFLASH_PORT    host port; default 8001 (REPLACES vllm-chat)
#   GPU_UTIL       default 0.80
#   MAX_LEN        default 262144 (native max — hybrid architecture, only
#                  the periodic full-attn layers carry KV, so this should
#                  have much more concurrency headroom than step 1's
#                  full-attention 30B; verify against the logged KV pool)
#   MAX_SEQS       default 4
#   SPEC_TOKENS    --speculative-config num_speculative_tokens; default 15
#   KV_CACHE_DTYPE default "auto" — MUST stay auto, fp8 crashes dflash
#                  engine init on this vLLM build (see step 1 script).
#   MAX_BATCHED_TOKENS  chunked-prefill cap; default 32768 (z-lab's own
#                  documented example for this pair).
#   TOOL_PARSER    default "qwen3_coder" (matches spin-up-vllm-qwen3-coder-
#                  next.sh — the non-dflash sibling for this model).
#   ATTN_BACKEND   VLLM_ATTENTION_BACKEND value; default "FLASH_ATTN".
#
# REVERT spark2 to the production MTP-2 Qwen3-Next-80B:
#   ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'
#
set -euo pipefail

source "$(dirname "$0")/lib-vllm-spinup.sh"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-Coder-Next-FP8}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-Coder-Next-DFlash}"
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

dflash_coder_next_failure_hints() {
    cat <<'EOF'
  Common causes:
   - dtype/shape mismatch loading the drafter's shared embed_tokens/lm_head
     (RuntimeError/size mismatch during "Sharing target model embedding
     weights with the draft model"): this IS the untested-pairing risk
     documented at the top of this script — the FP8 target's shared
     tensors don't match what the BF16-trained drafter expects. Fall back:
       TARGET_MODEL=Qwen/Qwen3-Coder-Next bash ~/spin-up-vllm-qwen3-coder-next-dflash.sh
   - "No valid attention backend found for cuda": you set KV_CACHE_DTYPE=fp8.
     Must be "auto" for any dflash run on this vLLM build (see step 1
     script's postmortem). Don't override this.
   - OOM at startup: hybrid architecture, so KV should be generous (few
     full-attn layers carry it) — if it still OOMs, weights are the
     culprit. Try MAX_LEN=131072, then GPU_UTIL=0.75.
   - Drafter 404 / Repository Not Found: verify HF_TOKEN is exported in
     ~/.bashrc.
   - Slow / stalled download: fresh ~80 GB FP8 pull (this exact checkpoint
     isn't cached on spark2, only on spark1). Check `docker logs -f
     vllm-chat` for shard-download progress, not just silence.
EOF
}

vllm_load_hf_token

echo "=== spin-up-vllm-qwen3-coder-next-dflash ==="
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
if [ "${TARGET_MODEL}" = "Qwen/Qwen3-Coder-Next-FP8" ]; then
    echo "  ⚠ UNTESTED PAIRING: drafter's own card documents BF16 Qwen/Qwen3-Coder-Next, not this FP8 checkpoint"
fi
echo "  ⚠ replaces spark2's current vllm-chat for the duration of this experiment"
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

# 40-min budget — fresh ~80 GB FP8 target pull (not cached on spark2).
vllm_wait_ready "${CONTAINER_NAME}" 2400 \
    "Traceback \(most recent call last\)|CUDA error|CUDA out of memory|RuntimeError:|ValueError:|ImportError:|AttributeError:|OSError:|out of memory|404 Client Error|Repository Not Found|^error: unrecognized arguments|not recognized|unknown architecture|does not have.*non-causal support|size mismatch" \
    dflash_coder_next_failure_hints \
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
