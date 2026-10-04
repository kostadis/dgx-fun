#!/usr/bin/env bash
#
# spin-up-vllm-qwen38-flash-next.sh
#
# Bring up Qwen3.8-Flash-Next (NVFP4 + fp8 side layers, "hybrid") on ONE box
# (spark1 by default), serving the chat slot on :8001.
#
# Run this FROM THE WORKSTATION (it SSHes to $HOST).
#
#     ./spin-up-vllm-qwen38-flash-next.sh              # spark1
#     HOST=spark2 ./spin-up-vllm-qwen38-flash-next.sh  # spark2 (same model, both boxes)
#
# Each box is an INDEPENDENT single-box endpoint — this is TP=1, not a cross-box
# pair, so running it on both boxes gives two identical endpoints, not one
# bigger one. The preflight below is what tells you a box is not yet stocked
# (recipe / image / checkpoint); see Â§8 of current-setup.md for how spark2
# was stocked from spark1 over the 10.100.16.x cable rather than re-downloading
# 126 GiB from HuggingFace.
#
# This is a THIN WRAPPER around the blazux recipe, which is cloned on the box
# at ~/qwen3.8-Flash-DGX:
#
#     https://github.com/blazux/qwen3.8-Flash-DGX
#
# The recipe's own scripts/serve.sh owns the docker invocation, the vLLM flags
# and the ten image patches. This wrapper owns only OUR choices on top of it
# (which box, which port, which profile) and the preflight/verify that the rest
# of this repo's spin-up-vllm-*.sh scripts do. Do NOT fork serve.sh — when the
# upstream recipe moves, `git -C ~/qwen3.8-Flash-DGX pull && docker build` on
# the box is the whole update.
#
# ---------------------------------------------------------------------------
# WHAT THIS REPLACES
#   The chat slot on $HOST:8001. As of the swap this script was written for
#   (2026-09-10) that was the cross-box `vllm-dspark`
#   (deepseek-ai/DeepSeek-V4-Flash-0731) which consumed BOTH boxes — so
#   bringing this up necessarily tore that down on spark1 AND spark2.
#   Ollama on :11434 is a systemd service and is NOT touched: the
#   qwen3-embedding:0.6b MemPalace path keeps working across the swap.
#
#   ⚠ Served model id is `qwen3.8-flash-next` — different from every id that
#   came before it. Clients that pin a model id (MemPalace, llm_wiki,
#   CampaignGenerator, opencode, openclaw) MUST be repointed. See §7 of
#   current-setup.md.
#
# REVERT
#   To the two-box Qwen3-Next world (weights cached on both boxes — a load,
#   not a download):
#     ssh spark  'PREFIX_CACHING=1 MAX_SEQS=3 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'
#     ssh spark2 'PREFIX_CACHING=1 MAX_SEQS=8 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'
#   To the cross-box DSpark pair (checkpoint still on disk on both boxes):
#     ./spin-up-vllm-dspark-2box.sh
#   Stop this container first either way (on whichever box you brought it up on):
#     ssh spark  'docker rm -f qwen38-flash'
#     ssh spark2 'docker rm -f qwen38-flash'
#
# ---------------------------------------------------------------------------
# WHY THESE KNOBS (all measured by blazux on a GX10 unless noted)
#
#   MODE=hybrid       NVFP4 routed experts as published + the dense side layers
#                     (GDN in/out proj, QSA q/k/v/o, shared experts, ~15 GiB of
#                     bf16) rewritten as blockwise fp8-e4m3. Those side layers
#                     are read in full on EVERY decoded token, so halving them
#                     is where the tokens/s come from: +20% decode, +8% KV,
#                     identical score on their 17-scenario agentic tournament.
#                     Needs the one-time scripts/prepare-hybrid.sh.
#
#   GPU_MEM=0.80      Their default AND our rule (memory
#                     `feedback_gpu_util_080_default`;
#                     gpu-reservation-and-kv-tradeoffs.md). Two independent
#                     paths to the same number. They additionally measured 0.85
#                     drifting into swap after a day and 0.875 getting
#                     OOM-killed on a 300k-token prefill. Do NOT raise it:
#                     on this model the host page cache is not spare capacity,
#                     it is what serves the 48 GiB n-gram table (below).
#
#   PREFIX_CACHE=1    Correct ONLY on this image. vLLM's EngineCore overwrites
#                     cache_config.block_size with the smallest KV-group block
#                     size (8 tokens at MTP=2, the QSA raw-key ring) while the
#                     Mamba state block is 1600; two call sites used the former
#                     as the latter, so a prefix hit computed the state slot as
#                     (3200-1)//8 = 399 instead of 1, read past the block table
#                     row, and restored an ALL-ZERO Mamba state. Result: no
#                     crash, no warning, silently different answers on cache
#                     hits — invisible to a coherence smoke test. The image
#                     carries the two-line fix (Dockerfile patch 4); with it,
#                     cold and cache-hit outputs are bit-identical.
#                     ⚠ Do NOT enable prefix caching for this model on any
#                     OTHER qwen38-flash-next image without checking for that
#                     fix first.
#
#   DET_TOPK=1        The GB10 QSA persistent_topk kernel is non-deterministic
#                     (identical greedy requests diverge 2 runs in 4) and drops
#                     legitimate attention candidates — vllm#51782. This is
#                     @jschmied's deterministic replacement kernel (vllm#55122),
#                     compiled into the image: identical greedy output at FULL
#                     prefill speed. EXACT_TOPK=1 is the torch.topk fallback,
#                     also deterministic but -20-40% on long prefill.
#
#   DRAFT_VOCAB=1     The MTP drafter scores the 65,536 most frequent tokens
#                     instead of 248,320 (320 MiB of head read per draft step
#                     instead of 1.27 GiB). The target verifies every drafted
#                     token, so emitted output is unchanged BY CONSTRUCTION.
#                     +20-23% decode, best tournament score they recorded.
#
#   MTP=2             Their tournament winner. MTP=3 is +7% decode but cost a
#                     point (44 vs 45/51), so it stays an option here too.
#
#   CTX=262144        The model's NATIVE context, and what the DSpark slot it
#                     replaced served (256K) — so no rope-scaling variable in
#                     the bring-up. YARN=1 CTX=500000 is validated upstream
#                     (needle-in-a-haystack at 414k) if the extra context is
#                     worth re-testing for.
#
#   SEQS=8            Recipe default. NOTE their warning: a low --max-num-seqs
#                     is indistinguishable from saturation if you only look at
#                     tok/s (at seqs 2 an aggregate sweep flatlined at ~33 tok/s
#                     while request_queue_time climbed to 142s). Do not
#                     benchmark this box at seqs 1-2.
#
#   PORT=8001         OUR override (recipe default is 18300) — this repo's chat
#                     slot, so client base URLs keep working. serve.sh maps
#                     host $PORT -> container 8000.
#
# NOT SET, deliberately:
#   KV_DTYPE=fp8_e4m3   x1.9 KV / 1M ctx, but -10% decode, -30% prefill and a
#                       measured quality loss. Not worth it at 262K.
#   PAD_M4=1            A no-op while PREFIX_CACHE=1 (chunks are 1600-aligned).
#   EXTRA=--long-prefill-token-threshold 1024
#                       The multi-client slider. With several interactive agents
#                       on the box, a DECODING client drops to ~0.2 tok/s for a
#                       minute or two while another client prefills a cold long
#                       prompt (vLLM chunked prefill: every step carrying a
#                       prefill chunk carries exactly one token per decoder).
#                       Not a bug and not fixable, only tradeable: 1024 lifts the
#                       stalled client to ~1.0 tok/s but costs 36% of an 8k TTFT.
#                       Left OFF for the single-main-user case; turn it on if
#                       two agents are routinely live at once.
#
# ---------------------------------------------------------------------------
set -euo pipefail

HOST="${HOST:-spark}"
RECIPE_DIR="${RECIPE_DIR:-\$HOME/qwen3.8-Flash-DGX}"
CONTAINER="${CONTAINER:-qwen38-flash}"

# LAN address for the summary at the end. `spark`/`spark2` are ~/.ssh/config
# aliases on the workstation and do NOT resolve in WSL2 — clients need the IP.
case "$HOST" in
  spark|spark1) HOST_IP="${HOST_IP:-192.168.1.147}" ;;
  spark2)       HOST_IP="${HOST_IP:-192.168.1.121}" ;;
  *)            HOST_IP="${HOST_IP:-$HOST}" ;;
esac

# Our profile. Every one of these is passed through to the recipe's serve.sh.
MODE="${MODE:-hybrid}"
PORT="${PORT:-8001}"
CTX="${CTX:-262144}"
YARN="${YARN:-0}"
GPU_MEM="${GPU_MEM:-0.80}"
MTP="${MTP:-2}"
SEQS="${SEQS:-8}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
DET_TOPK="${DET_TOPK:-1}"
DRAFT_VOCAB="${DRAFT_VOCAB:-1}"
KV_DTYPE="${KV_DTYPE:-auto}"
PREWARM="${PREWARM:-0}"
EXTRA="${EXTRA:-}"

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"

say() { printf '\n\033[0;34m>> %s\033[0m\n' "$*"; }
die() { printf '\n\033[0;31m!! %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preflight — fail before tearing anything down.
# ---------------------------------------------------------------------------
say "preflight on ${HOST}"
# Quoted heredoc: nothing here is expanded locally. MODE/RECIPE_DIR arrive as $1/$2,
# so $HOME and friends resolve on the BOX. (Do not switch this back to an
# interpolated double-quoted string — that is how the $HOME-quoting bug got in.)
ssh -o ConnectTimeout=10 "${HOST}" "bash -s -- '${MODE}' '${RECIPE_DIR}'" <<'PREFLIGHT' || die "preflight failed on ${HOST}"
set -e
MODE="$1"; RECIPE_DIR="$2"
eval "RECIPE_DIR=$RECIPE_DIR"   # let a literal $HOME in the default expand here
[ -d "$RECIPE_DIR" ] || { echo "!! recipe not cloned: git clone https://github.com/blazux/qwen3.8-Flash-DGX.git ~/qwen3.8-Flash-DGX"; exit 1; }
docker image inspect qwen38-flash-dgx >/dev/null 2>&1 || { echo "!! image qwen38-flash-dgx missing: (cd $RECIPE_DIR && docker build -t qwen38-flash-dgx .)"; exit 1; }
REPO="$HOME/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4"
[ -d "$REPO" ] || { echo "!! checkpoint missing: (cd $RECIPE_DIR && scripts/download-weights.sh)"; exit 1; }
if [ "$MODE" != nvfp4 ]; then
  SUF=-fp8hybrid; [ "$MODE" = hybrid-mtp ] && SUF=-fp8hybrid-mtpnvfp4
  ls "$REPO"/snapshots/*"$SUF"/.prepared >/dev/null 2>&1 || { echo "!! $MODE checkpoint not prepared: (cd $RECIPE_DIR && scripts/prepare-hybrid.sh)"; exit 1; }
fi
echo "preflight OK"
PREFLIGHT

# ---------------------------------------------------------------------------
# Drop the page cache. On this box vLLM does not reliably release its
# unified-memory footprint after shutdown, and the 48 GiB PLE table is served
# THROUGH the page cache — a stale one both starves the new run and slows it.
# ---------------------------------------------------------------------------
say "dropping page cache on ${HOST}"
ssh -o ConnectTimeout=10 "${HOST}" '
  docker rm -f '"${CONTAINER}"' >/dev/null 2>&1 || true
  sleep 2
  if sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; then
    echo "page cache dropped (sudo)"
  else
    IMG=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -v "<none>" | head -1)
    docker run --rm --privileged --pid=host --entrypoint sh "$IMG" \
      -c "sync; echo 3 > /proc/sys/vm/drop_caches" >/dev/null 2>&1 \
      && echo "page cache dropped (privileged container)" \
      || echo "!! WARNING: could not drop page cache — the first boot may fail to allocate"
  fi
  free -g | head -2
'

# ---------------------------------------------------------------------------
# Serve.
# ---------------------------------------------------------------------------
say "starting ${CONTAINER} on ${HOST}:${PORT} (mode=${MODE}, ctx=${CTX}, mtp=${MTP}, seqs=${SEQS}, util=${GPU_MEM})"
ssh -o ConnectTimeout=10 "${HOST}" "
  cd ${RECIPE_DIR} &&
  MODE='${MODE}' PORT='${PORT}' CTX='${CTX}' YARN='${YARN}' GPU_MEM='${GPU_MEM}' \
  MTP='${MTP}' SEQS='${SEQS}' PREFIX_CACHE='${PREFIX_CACHE}' DET_TOPK='${DET_TOPK}' \
  DRAFT_VOCAB='${DRAFT_VOCAB}' KV_DTYPE='${KV_DTYPE}' PREWARM='${PREWARM}' \
  EXTRA='${EXTRA}' NAME='${CONTAINER}' \
  scripts/serve.sh
" || die "serve.sh failed on ${HOST}"

# ---------------------------------------------------------------------------
# Health. First boot loads ~76 GiB of weights: 8-13 min is normal.
# ---------------------------------------------------------------------------
say "waiting up to ${HEALTH_TIMEOUT}s for http://${HOST}:${PORT}/health"
ssh -o ConnectTimeout=10 "${HOST}" "
  deadline=\$(( \$(date +%s) + ${HEALTH_TIMEOUT} ))
  while [ \$(date +%s) -lt \$deadline ]; do
    if ! docker ps -q --filter 'name=^/${CONTAINER}\$' | grep -q .; then
      echo '!! container exited during startup — last 40 lines:'
      docker logs --tail 40 '${CONTAINER}' 2>&1
      exit 1
    fi
    curl -sf -m 5 http://127.0.0.1:${PORT}/health >/dev/null 2>&1 && { echo 'HEALTHY'; exit 0; }
    sleep 15
  done
  echo '!! not healthy within ${HEALTH_TIMEOUT}s — last 40 lines:'
  docker logs --tail 40 '${CONTAINER}' 2>&1
  exit 1
" || die "${CONTAINER} did not become healthy"

# ---------------------------------------------------------------------------
# Verify. Read the numbers out of the engine's own log rather than trusting
# the flags we passed (`feedback_size_context_by_kv_pool`: size context from
# the MEASURED KV pool, never from a slot guess).
# ---------------------------------------------------------------------------
say "verifying"
ssh -o ConnectTimeout=10 "${HOST}" "
  echo '--- served id ---'
  curl -sS -m 10 http://127.0.0.1:${PORT}/v1/models |
    python3 -c 'import sys,json; d=json.load(sys.stdin)[\"data\"][0]; print(d[\"id\"], \"| max_model_len:\", d[\"max_model_len\"])'
  echo '--- engine config (from its own log, not our flags) ---'
  docker logs '${CONTAINER}' 2>&1 | grep -oE \
    'GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|enable_prefix_caching=[A-Za-z]+|num_speculative_tokens=[0-9]+|max_num_seqs=[0-9]+|kv_cache_dtype=[a-z0-9_]+' | sort -u
  echo '--- host memory ---'
  free -g | head -2
"

say "up. Next: (cd ${RECIPE_DIR} && PORT=${PORT} scripts/smoke-test.sh) on ${HOST}"
cat <<EOF

  Box             : ${HOST} (${HOST_IP})
  Served model id : qwen3.8-flash-next     <-- clients MUST use this
  Endpoint        : http://${HOST_IP}:${PORT}/v1
  Container       : ${CONTAINER}  (--restart unless-stopped, survives a reboot)
  Logs            : ssh ${HOST} 'docker logs -f ${CONTAINER}'

  Remember to sync current-setup.md and dgxlib/models.yaml in the SAME change.
EOF
