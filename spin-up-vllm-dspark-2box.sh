#!/usr/bin/env bash
#
# spin-up-vllm-dspark-2box.sh
#
# Bring up a DeepSeek-V4-Flash checkpoint across spark1 + spark2 (TP=2, `mp`
# backend — NO Ray) serving the chat slot on spark1:8001. Default MODEL is
# now `deepseek-ai/DeepSeek-V4-Flash-0731` (2026-08-03 — see "2026-08-03
# UPGRADE" below); override with MODEL=... to run any other DeepSeek-V4-Flash
# checkpoint (e.g. the original DSpark preview) that's staged in the HF cache.
#
# Run this FROM THE WORKSTATION (it SSHes to `spark` and `spark2`).
#
# Plan/runbook: deepseek-v4-flash-dspark-2box-plan.md
#
# ---------------------------------------------------------------------------
# WHAT THIS REPLACES
#   Stops whatever is currently in the `vllm-dspark` (or `vllm-chat`) slot on
#   BOTH boxes and starts $MODEL there. That is the model MemPalace/llm_wiki/
#   CampaignGenerator point at, so this is a service outage for the chat
#   endpoint. Ollama on :11434 is a systemd service and is NOT touched — the
#   qwen3-embedding path keeps working.
#
# REVERT
#   Nearest: rerun this script with the previous checkpoint, e.g.
#     MODEL=deepseek-ai/DeepSeek-V4-Flash-DSpark SPEC_TOKENS=3 ./spin-up-vllm-dspark-2box.sh
#   (that checkpoint is left on disk on both boxes on purpose — this is a
#   container restart, not a re-download.)
#   Furthest, back to Qwen3-Next-80B:
#   ssh spark  'bash ~/spin-up-vllm-qwen3-next-80b.sh'
#   ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'
#   (weights are cached — a load, not a download)
#
# ---------------------------------------------------------------------------
# 2026-08-03 UPGRADE: DSpark preview -> DeepSeek-V4-Flash-0731
#   0731 is DeepSeek's official V4-Flash release (2026-07-31), same
#   architecture/quant format as the DSpark preview (byte-identical
#   config.json: 284B/13B active, fp8/e4m3/ue8m0, 43 layers, dspark_block_size
#   5, same fused draft module) — a retrain, not a new architecture. Verified
#   against the live preview container's own logs/code before swapping, not
#   assumed from the community recipes (which disagreed with each other and,
#   in two cases, targeted a different image lineage entirely). Two changes
#   made here as a result:
#     - SPEC_TOKENS default 3 -> 5, matching config.json's dspark_block_size
#       (the drafter emits exactly that many tokens per pass; the original
#       preview bring-up's "3, not the card's 7" note undersold the real
#       constraint — 3 boots and serves but isn't the checkpoint's native
#       block size).
#     - VLLM_USE_BREAKABLE_CUDAGRAPH=0 added — a real, recognized flag in
#       this image (confirmed via the installed vllm/config/vllm.py, which
#       auto-enables it otherwise), reported to cost 13-29% decode if left
#       unset. Unverified ON THIS IMAGE at the time of this change — carried
#       over from a community report on a different image lineage.
#   NOT changed: tool-call parser (deepseek_v4, unchanged), image tag
#   (0.1.1 already meets 0731's vLLM>=0.25.0 requirement), gpu-memory-
#   utilization (0.80 unchanged). Checked and NOT an issue: omitted
#   `reasoning_effort` (what every current client sends) passes through this
#   image's tokenizer wrapper as None, unaffected by the low/high/max scheme
#   0731 introduces — only an explicit `reasoning_effort="low"` would be
#   mis-mapped to "high" by this image's pre-0731 wrapper, and nothing sends
#   that today.
#
# ---------------------------------------------------------------------------
# CORRECTIONS TO THE PUBLISHED RECIPES (verified against the image, not
# assumed — `vllm serve --help=all` + reading the installed vllm source):
#
#   1. The recipes say "identical command on both nodes". THAT IS WRONG for
#      this image. `--node-rank` must differ (0 = head, 1 = worker), and the
#      worker MUST get `--headless`. Proof: in
#      vllm/entrypoints/cli/serve.py, the multi-node TP worker branch
#      (`node_rank_within_dp > 0` -> MultiprocExecutor) lives inside
#      run_headless(), which is only reached when api_server_count < 1 —
#      and only `--headless` sets that to 0. Without `--headless` the worker
#      would try to stand up its own API server + engine instead of joining.
#
#   2. Node discovery is via the `--master-addr` / `--master-port` CLI flags
#      (ParallelConfig fields, defaults 127.0.0.1:29501), not a MASTER_PORT
#      env var.
#
#   3. Five of the nine "not optional" env vars in the recipe notes do not
#      exist anywhere in this image (0 files match under the vllm package):
#      WORKER_VLLM_HOST_IP, VLLM_USE_B12X_WO_PROJECTION,
#      VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK, VLLM_DSPARK_REPLICATE_MARKOV_W1,
#      VLLM_USE_B12X_FP8_GEMM. They belong to the *other* recipe's
#      Stage-C build. Setting them here would be cargo-culting, so we don't.
#      (Corollary: the "never set VLLM_USE_B12X_FP8_GEMM=1" warning is moot
#      for this image — there is nothing to read it.)
#      The ones that ARE real and ARE set: VLLM_HOST_IP, VLLM_USE_B12X_MOE,
#      VLLM_USE_FLASHINFER_SAMPLER.
#
#   4. Confirmed valid enum values in this build: `--kv-cache-dtype
#      nvfp4_ds_mla`, `--moe-backend flashinfer_b12x` (its help text says
#      "for SM12x (RTX Pro 6000 / DGX Spark)"), and speculative
#      method "dspark" with draft_sample_method "probabilistic"
#      (DraftSampleMethod = Literal["greedy","probabilistic"]).
#
# ---------------------------------------------------------------------------
# DELIBERATE DEVIATIONS FROM THE RECIPE
#
#   GPU_UTIL=0.80 not 0.85. `feedback_gpu_util_080_default`: 0.88 left ~15 GB
#   host and starved sshd's fork on spark2 badly enough to need a physical
#   reboot. 0.80 is our house default. The two recipes disagree anyway
#   (anemll 0.85, tonyd2wild 0.80).
#
#   MAX_LEN=262144 not 1048576 on the first bring-up. The recipe's 1M figure
#   comes with util 0.85; at 0.80 the KV pool is ~6 GB smaller and 1M may
#   simply not fit, turning bring-up into a boot failure for no measurement
#   benefit. Per `feedback_size_context_by_kv_pool`, read the real
#   "GPU KV cache size" from the log FIRST, then decide whether to push to
#   1M. 256K also covers the entire §6 measurement plan (8K/32K/128K).
#
#   RDMA=0 by default. The published 60-67 tok/s numbers were achieved over
#   plain TCP sockets on the cable; adding RoCE on the first bring-up would
#   add a variable to a config that has never booted here. RDMA=1 wires up
#   the IB path (same knobs as spin-up-vllm-2box-rdma.sh) as a follow-on
#   experiment once the baseline works.
#
#   No --trust-remote-code. This image has NATIVE DSpark support
#   (DSparkDraftModel / model_type deepseek_v4 in config/speculative.py), so
#   pulling the checkpoint's bundled inference/ code would risk shadowing the
#   optimized path. Add it only if load actually demands it.
#
# ---------------------------------------------------------------------------
# USAGE
#   ./spin-up-vllm-dspark-2box.sh
#   MAX_LEN=1048576 ./spin-up-vllm-dspark-2box.sh   # after checking KV pool
#   RDMA=1 ./spin-up-vllm-dspark-2box.sh            # follow-on experiment
#   SPEC=0 ./spin-up-vllm-dspark-2box.sh            # disable spec decode
#
set -euo pipefail

# ---- knobs ------------------------------------------------------------------
HEAD=spark                              # ssh alias, spark1 (LAN 192.168.1.147)
WORKER=spark2                           # ssh alias, spark2 (LAN 192.168.1.121)
HEAD_LAN=192.168.1.147
HEAD_IP=10.100.16.1                     # cable IP, spark1 enp1s0f0np0
WORKER_IP=10.100.16.2                   # cable IP, spark2 enp1s0f0np0
CABLE_IF=enp1s0f0np0
IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1
CONTAINER=vllm-dspark
OLD_CONTAINER=vllm-chat                 # what we are displacing on both boxes
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}"
HF_CACHE=/home/kostadis/.cache/huggingface
CHAT_PORT="${CHAT_PORT:-8001}"
MASTER_PORT="${MASTER_PORT:-25440}"
SHM="${SHM:-8589934592}"                # 8g
GPU_UTIL="${GPU_UTIL:-0.80}"
MAX_LEN="${MAX_LEN:-262144}"
MAX_SEQS="${MAX_SEQS:-6}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
KV_DTYPE="${KV_DTYPE:-nvfp4_ds_mla}"
MOE_BACKEND="${MOE_BACKEND:-flashinfer_b12x}"
SPEC="${SPEC:-1}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"         # 5, matching config.json's
                                        # dspark_block_size (the drafter emits
                                        # exactly this many tokens per pass).
                                        # NOT the card's 7 (4xGB300 stack, a
                                        # different serving config). The
                                        # original preview default of 3 also
                                        # boots and serves, just not at the
                                        # checkpoint's native block size.
RDMA="${RDMA:-0}"
TOOL_PARSER="${TOOL_PARSER:-deepseek_v4}"  # HEAD only. Without --enable-auto-tool-choice
                                        # + a parser, vLLM 400s ANY request carrying
                                        # tools ('"auto" tool choice requires ...'),
                                        # so every agent client dies before inference.
                                        # This image offers deepseek_v3/_v31/_v32/_v4.
BOOT_BUDGET="${BOOT_BUDGET:-3600}"      # 60 min: ~78 GB/box cold load + compile

run_remote() { ssh -o ConnectTimeout=10 "$1" "${2}"; }

# ---- 0. preflight -----------------------------------------------------------
echo ">>> [0/6] Preflight..."
for box in "$HEAD" "$WORKER"; do
  run_remote "$box" "test -d ${HF_CACHE}/hub || { echo 'no HF cache on '\$(hostname); exit 1; }"
done
# Cable must be real: ICMP is not proof (feedback_test_tcp_not_just_ping_rdma).
run_remote "$HEAD" "ping -c2 -W2 ${WORKER_IP} >/dev/null && echo '    cable ping OK'"
run_remote "$HEAD" "timeout 5 bash -c 'cat < /dev/null > /dev/tcp/${WORKER_IP}/22' \
  && echo '    cable TCP:22 OK' || echo '    !! cable TCP:22 FAILED'"

# ---- speculative + rdma arg assembly ---------------------------------------
SPEC_ARG=""
if [[ "$SPEC" == "1" ]]; then
  # The docker command below is a STRING re-parsed by the REMOTE shell, so
  # this JSON must arrive single-quoted. Learned the hard way on the first
  # bring-up: unquoted, the remote shell brace-expands
  # {"method":"dspark","num_speculative_tokens":3,...} on its commas into
  # separate words, and vLLM sees the fragment `method:dspark` ->
  #   error: argument --speculative-config: Value method:dspark cannot be
  #   converted to <function loads>
  SPEC_ARG="--speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\"}'"
  echo ">>> Spec decode: dspark, ${SPEC_TOKENS} tokens, probabilistic"
else
  echo ">>> Spec decode: DISABLED"
fi

RDMA_ENV=(); RDMA_FLAGS=()
if [[ "$RDMA" == "1" ]]; then
  RDMA_ENV=( -e NCCL_IB_HCA=rocep1s0f0:1 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_DISABLE=0 )
  RDMA_FLAGS=( --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 )
  echo ">>> Transport: RDMA/RoCE"
else
  echo ">>> Transport: TCP sockets on ${CABLE_IF}"
fi

# Env real in THIS image (see correction #3 above).
COMMON_ENV=(
  -e VLLM_USE_B12X_MOE=1
  -e VLLM_USE_FLASHINFER_SAMPLER=1
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0        # added 2026-08-03 for 0731 — see
                                            # the UPGRADE note at top of file
  -e HF_HUB_OFFLINE=1
  -e NCCL_SOCKET_IFNAME=${CABLE_IF}
  -e GLOO_SOCKET_IFNAME=${CABLE_IF}
  -e TP_SOCKET_IFNAME=${CABLE_IF}
  -e NCCL_DEBUG=WARN
)

COMMON_SERVE=(
  "${MODEL}"
  --tensor-parallel-size 2
  --distributed-executor-backend mp
  --nnodes 2
  --master-addr "${HEAD_IP}"
  --master-port "${MASTER_PORT}"
  --kv-cache-dtype "${KV_DTYPE}"
  --block-size "${BLOCK_SIZE}"
  --max-model-len "${MAX_LEN}"
  --max-num-seqs "${MAX_SEQS}"
  --max-num-batched-tokens "${MAX_BATCHED}"
  --gpu-memory-utilization "${GPU_UTIL}"
  --moe-backend "${MOE_BACKEND}"
  --async-scheduling
  --enable-chunked-prefill
  --generation-config vllm
)

echo ">>> Config: util=${GPU_UTIL} max_len=${MAX_LEN} seqs=${MAX_SEQS} kv=${KV_DTYPE} moe=${MOE_BACKEND}"

# ---- 1. DESTRUCTIVE: stop the Qwen chat slot on both boxes ------------------
echo ">>> [1/6] Stopping ${OLD_CONTAINER} on ${HEAD} and ${WORKER} (chat outage begins)..."
# --restart unless-stopped means an explicit stop sticks.
run_remote "$HEAD"   "docker stop ${OLD_CONTAINER} 2>/dev/null || true"
run_remote "$WORKER" "docker stop ${OLD_CONTAINER} 2>/dev/null || true"
run_remote "$HEAD"   "docker rm -f ${CONTAINER} 2>/dev/null || true"
run_remote "$WORKER" "docker rm -f ${CONTAINER} 2>/dev/null || true"
echo ">>> Waiting 20s for GPU memory to be released..."
sleep 20
run_remote "$HEAD"   "nvidia-smi --query-gpu=memory.used --format=csv,noheader || true"
run_remote "$WORKER" "nvidia-smi --query-gpu=memory.used --format=csv,noheader || true"

# ---- 2. WORKER first (node-rank 1, headless) --------------------------------
echo ">>> [2/6] Starting WORKER on ${WORKER} (node-rank 1, headless)..."
run_remote "$WORKER" "docker run -d --name ${CONTAINER} --network host \
  --runtime nvidia --gpus all --shm-size=${SHM} --ipc=host \
  ${RDMA_FLAGS[*]} ${COMMON_ENV[*]} ${RDMA_ENV[*]} \
  -e VLLM_HOST_IP=${WORKER_IP} \
  -v ${HF_CACHE}:/root/.cache/huggingface \
  ${IMAGE} ${COMMON_SERVE[*]} ${SPEC_ARG} \
  --node-rank 1 --headless"

sleep 15

# ---- 3. HEAD second (node-rank 0, serves the API) --------------------------
echo ">>> [3/6] Starting HEAD on ${HEAD}:${CHAT_PORT} (node-rank 0)..."
run_remote "$HEAD" "docker run -d --name ${CONTAINER} --network host \
  --runtime nvidia --gpus all --shm-size=${SHM} --ipc=host \
  ${RDMA_FLAGS[*]} ${COMMON_ENV[*]} ${RDMA_ENV[*]} \
  -e VLLM_HOST_IP=${HEAD_IP} \
  -v ${HF_CACHE}:/root/.cache/huggingface \
  ${IMAGE} ${COMMON_SERVE[*]} ${SPEC_ARG} \
  --node-rank 0 --host 0.0.0.0 --port ${CHAT_PORT} \
  --enable-auto-tool-choice --tool-call-parser ${TOOL_PARSER}"

# ---- 4. wait for the endpoint ----------------------------------------------
echo ">>> [4/6] Waiting for ${HEAD_LAN}:${CHAT_PORT} (budget ${BOOT_BUDGET}s)..."
ok=0
deadline=$(( $(date +%s) + BOOT_BUDGET ))
while [[ $(date +%s) -lt $deadline ]]; do
  if curl -sS --max-time 5 "http://${HEAD_LAN}:${CHAT_PORT}/v1/models" 2>/dev/null | grep -qF "${MODEL##*/}"; then
    echo "    Endpoint live."
    ok=1; break
  fi
  # Fail fast on a dead container rather than burning the whole budget.
  if ! run_remote "$HEAD" "docker ps --format '{{.Names}}' | grep -q '^${CONTAINER}$'"; then
    echo "!!! HEAD container exited. Last 40 lines:"
    run_remote "$HEAD" "docker logs --tail 40 ${CONTAINER} 2>&1 || true"
    break
  fi
  sleep 15
done

# ---- 5. KV pool + spec-decode boot signature -------------------------------
echo ">>> [5/6] Boot signature (read the REAL numbers, don't trust the recipe):"
run_remote "$HEAD" "docker logs ${CONTAINER} 2>&1 | grep -aE 'GPU KV cache size|Maximum concurrency|Speculative|dspark|graph capturing|Application startup' | tail -12 || true"
echo ">>> Weight-mapping sanity (should be 0 on both ranks — a >0 count means"
echo ">>> draft-model tensors are being silently dropped, see the 2026-08-03"
echo ">>> UPGRADE note at top of file):"
for box in "$HEAD" "$WORKER"; do
  n=$(run_remote "$box" "docker logs ${CONTAINER} 2>&1 | grep -c 'Skipping unknown' || true")
  echo "    ${box}: ${n} 'Skipping unknown' warnings"
done

# ---- 6. smoke: check MEANING, not HTTP 200 ---------------------------------
if [[ "$ok" != "1" ]]; then
  echo "!!! Endpoint did not come up."
  echo "!!! HEAD log:   ssh ${HEAD} 'docker logs --tail 80 ${CONTAINER}'"
  echo "!!! WORKER log: ssh ${WORKER} 'docker logs --tail 80 ${CONTAINER}'"
  echo "!!! REVERT:     ssh ${HEAD} 'bash ~/spin-up-vllm-qwen3-next-80b.sh'"
  exit 1
fi

echo ">>> [6/6] Coherence smoke test (gibberish is a KNOWN failure mode here)..."
t0=$(date +%s.%N)
resp=$(curl -sS --max-time 300 "http://${HEAD_LAN}:${CHAT_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Name the capital of France, then count from 1 to 10.\"}],\"max_tokens\":128,\"temperature\":0,\"stream\":false}")
t1=$(date +%s.%N)
echo "--- raw response ---"
echo "$resp" | head -c 1200; echo
ct=$(echo "$resp" | grep -o '"completion_tokens":[0-9]*' | grep -o '[0-9]*' || echo 0)
el=$(echo "$t1 - $t0" | bc)
echo "    completion_tokens=${ct:-?} elapsed=${el}s ~$(echo "scale=1; ${ct:-0}/${el}" | bc) tok/s (incl. prefill)"
if echo "$resp" | grep -qi "paris"; then
  echo "    COHERENCE: PASS (found 'Paris')"
else
  echo "    COHERENCE: *** FAIL *** — no 'Paris'. Suspect the gibberish failure mode."
fi

echo ">>> Done. Next: measure cold TTFT at 8K/32K/128K (the go/no-go), then"
echo ">>> update current-setup.md + dgxlib/models.yaml + a new"
echo ">>> deepseek-v4-flash-dspark-observations.md per this repo's CLAUDE.md."
