#!/usr/bin/env bash
#
# spin-up-vllm-2box-rdma.sh
#
# Rebuild the cross-box (spark1 + spark2) vLLM Ray cluster that serves the
# `vllm-2box` slot on spark1:8001, switching the NCCL tensor-parallel
# transport from TCP sockets to RDMA/RoCE over the direct DAC cable.
#
# Run this FROM THE WORKSTATION (it SSHes to `spark` and `spark2`).
#
# Why: TP=2 decode was ~11 tok/s, bottlenecked by per-token NCCL all-reduce
# over TCP sockets on the cable. The cable is RoCE-capable (HCA rocep1s0f0,
# port 1 ACTIVE, RoCE v2 GID index 3). Moving NCCL to IB verbs attacks the
# per-token latency directly.
#
# The current containers have NO RDMA access (no /dev/infiniband device, no
# IPC_LOCK cap, no memlock ulimit), so we must RECREATE both containers.
#
# REVERT to the socket-only path: re-run with RDMA=0
#   RDMA=0 ./spin-up-vllm-2box-rdma.sh
# (brings the cluster back exactly as it was before this change.)
#
# SELECT the model in the slot with PROFILE (default: minimax):
#   PROFILE=qwen35 ./spin-up-vllm-2box-rdma.sh   # Qwen3.5-122B-A10B-FP8
#   PROFILE=minimax ./spin-up-vllm-2box-rdma.sh  # nvidia/MiniMax-M2.7-NVFP4
#
set -euo pipefail

# ---- knobs ------------------------------------------------------------------
RDMA="${RDMA:-1}"                       # 1 = RoCE/IB transport, 0 = TCP sockets (revert)
HEAD=spark                              # ssh alias, spark1 (LAN 192.168.1.147)
WORKER=spark2                           # ssh alias, spark2 (on wifi for mgmt; cable for TP)
IMAGE=local/vllm-ray:26.05
CONTAINER=vllm-2box
HF_CACHE=/home/kostadis/.cache/huggingface
SHM=10995116277                         # 10.24g, matches prior run
HEAD_IP=10.100.16.1                     # cable IP, spark1 enp1s0f0np0
WORKER_IP=10.100.16.2                   # cable IP, spark2 enp1s0f0np0
RAY_PORT=6379
CHAT_PORT=8001

# Which model occupies the cross-box slot. Each profile carries its own
# model id + tool/reasoning parsers + context length; everything else
# (TP=2, Ray backend, RoCE transport, gpu-util) is shared below.
#   minimax  - nvidia/MiniMax-M2.7-NVFP4 (230B/10B NVFP4; minimax_m2 parsers)
#   qwen35   - Qwen/Qwen3.5-122B-A10B-FP8 (122B/10B FP8; qwen3 parsers, no
#              think-leak into content, no path-corruption bug)
PROFILE="${PROFILE:-minimax}"
case "$PROFILE" in
  minimax)
    MODEL=nvidia/MiniMax-M2.7-NVFP4
    MODEL_FLAGS="--tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think --max-model-len 65536"
    ;;
  qwen35)
    MODEL=Qwen/Qwen3.5-122B-A10B-FP8
    MODEL_FLAGS="--tool-call-parser qwen3_coder --reasoning-parser qwen3 --max-model-len 131072"
    ;;
  *)
    echo "!!! Unknown PROFILE='$PROFILE' (expected: minimax | qwen35)" >&2
    exit 1
    ;;
esac
echo ">>> Profile: ${PROFILE}  Model: ${MODEL}"

SERVE_FLAGS="--tensor-parallel-size 2 --distributed-executor-backend ray \
  ${MODEL_FLAGS} \
  --enable-auto-tool-choice --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port ${CHAT_PORT}"

# ---- shared env for both containers ----------------------------------------
# Bootstrap / OOB pinning (unchanged from the socket build):
COMMON_ENV=(
  -e RAY_memory_monitor_refresh_ms=0
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0
  -e TP_SOCKET_IFNAME=enp1s0f0np0
  -e UCX_NET_DEVICES=enp1s0f0np0
  -e OMPI_MCA_btl_tcp_if_include=enp1s0f0np0
  -e MASTER_ADDR=${HEAD_IP}
)

# RDMA-specific env + docker run flags (only when RDMA=1):
RDMA_ENV=()
RDMA_FLAGS=()
if [[ "$RDMA" == "1" ]]; then
  RDMA_ENV=(
    -e NCCL_IB_HCA=rocep1s0f0:1     # cable HCA, port 1 (MTU 4096)
    -e NCCL_IB_GID_INDEX=3          # RoCE v2 / IPv4 GID
    -e NCCL_IB_DISABLE=0
    -e NCCL_DEBUG=INFO              # so we can prove NET/IB was selected
  )
  RDMA_FLAGS=(
    --device /dev/infiniband
    --cap-add IPC_LOCK
    --ulimit memlock=-1:-1
  )
  echo ">>> Transport: RDMA/RoCE (NCCL_IB_HCA=rocep1s0f0:1, GID 3)"
else
  RDMA_ENV=( -e NCCL_DEBUG=INFO )
  echo ">>> Transport: TCP sockets (RDMA=0, revert mode)"
fi

run_remote() { ssh -o ConnectTimeout=10 "$1" "${2}"; }

# ---- 1. tear down both containers ------------------------------------------
echo ">>> [1/6] Removing existing ${CONTAINER} on ${HEAD} and ${WORKER}..."
run_remote "$HEAD"   "docker rm -f ${CONTAINER} 2>/dev/null || true"
run_remote "$WORKER" "docker rm -f ${CONTAINER} 2>/dev/null || true"

# ---- 2. start Ray HEAD on spark1 -------------------------------------------
echo ">>> [2/6] Starting Ray head on ${HEAD} (${HEAD_IP})..."
run_remote "$HEAD" "docker run -d --name ${CONTAINER} --network host \
  --gpus all --shm-size=${SHM} \
  ${RDMA_FLAGS[*]} \
  ${COMMON_ENV[*]} ${RDMA_ENV[*]} -e VLLM_HOST_IP=${HEAD_IP} \
  -v ${HF_CACHE}:/root/.cache/huggingface \
  ${IMAGE} bash -c \
  'ray start --head --node-ip-address=${HEAD_IP} --port=${RAY_PORT} --dashboard-host=0.0.0.0 --block'"

echo ">>> Waiting for Ray GCS to bind (cold bind ~55s on this box)..."
for i in $(seq 1 24); do
  if run_remote "$HEAD" "docker exec ${CONTAINER} ray status >/dev/null 2>&1"; then
    echo "    GCS up after ~$((i*5))s."
    break
  fi
  sleep 5
done

# ---- 3. start Ray WORKER on spark2 -----------------------------------------
echo ">>> [3/6] Starting Ray worker on ${WORKER} (${WORKER_IP})..."
run_remote "$WORKER" "docker run -d --name ${CONTAINER} --network host \
  --gpus all --shm-size=${SHM} \
  ${RDMA_FLAGS[*]} \
  ${COMMON_ENV[*]} ${RDMA_ENV[*]} -e VLLM_HOST_IP=${WORKER_IP} \
  -v ${HF_CACHE}:/root/.cache/huggingface \
  ${IMAGE} bash -c \
  'ray start --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${WORKER_IP} --block'"

# ---- 4. wait for 2-node cluster --------------------------------------------
echo ">>> [4/6] Waiting for 2-node / 2-GPU Ray cluster..."
for i in $(seq 1 24); do
  nodes=$(run_remote "$HEAD" "docker exec ${CONTAINER} ray status 2>/dev/null | grep -c ' node_' || true")
  if [[ "${nodes:-0}" -ge 2 ]]; then
    echo "    ${nodes} nodes registered."
    run_remote "$HEAD" "docker exec ${CONTAINER} ray status 2>/dev/null | grep -E 'GPU|node_' | head"
    break
  fi
  sleep 5
done

# ---- 5. launch vllm serve into the head (detached) -------------------------
echo ">>> [5/6] Launching vllm serve (${MODEL}) on ${HEAD}:${CHAT_PORT}..."
run_remote "$HEAD" "docker exec -d ${CONTAINER} bash -c \
  'vllm serve ${MODEL} ${SERVE_FLAGS} > /tmp/vllm-serve.log 2>&1'"

# Cold load is SLOW on this box: the FP8/FP4 checkpoint exceeds available
# RAM, so vLLM disables auto-prefetch on EXT4 and reads shards serially —
# rank-0 weight load alone measured ~875 s for qwen35, plus compile +
# CUDA-graph capture. Wait up to 40 min before declaring failure; the
# detached `vllm serve` keeps loading regardless of this loop.
echo ">>> Waiting for the chat endpoint to answer (cold load is slow: ~875s rank-0 weight load + compile; up to 40 min)..."
ok=0
for i in $(seq 1 240); do
  if curl -sS --max-time 5 "http://192.168.1.147:${CHAT_PORT}/v1/models" 2>/dev/null | grep -q "${MODEL}"; then
    echo "    Endpoint live after ~$((i*10))s."
    ok=1
    break
  fi
  sleep 10
done

# ---- 6. verify transport + smoke + rough decode ----------------------------
echo ">>> [6/6] Verifying NCCL transport..."
echo "----- NCCL NET selection (expect NET/IB ... rocep1s0f0, NOT NET/Socket) -----"
run_remote "$HEAD" "docker exec ${CONTAINER} grep -aE 'NCCL INFO (NET/|Using network|NET/IB)' /tmp/vllm-serve.log 2>/dev/null | head -8 || echo '(no NCCL lines yet)'"

if [[ "$ok" == "1" ]]; then
  echo "----- smoke + rough decode (max_tokens=128) -----"
  t0=$(date +%s.%N)
  resp=$(curl -sS --max-time 180 "http://192.168.1.147:${CHAT_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 30.\"}],\"max_tokens\":128,\"stream\":false}")
  t1=$(date +%s.%N)
  ct=$(echo "$resp" | grep -o '"completion_tokens":[0-9]*' | grep -o '[0-9]*')
  el=$(echo "$t1 - $t0" | bc)
  echo "    completion_tokens=${ct:-?}  elapsed=${el}s  ~$(echo "scale=1; ${ct:-0}/${el}" | bc) tok/s (incl. prefill)"
else
  echo "!!! Endpoint did not come up. Check: ssh ${HEAD} 'docker exec ${CONTAINER} tail -40 /tmp/vllm-serve.log'"
  echo "!!! To revert to the working socket path: RDMA=0 $0"
  exit 1
fi

echo ">>> Done. If decode improved, drop NCCL_DEBUG and update current-setup.md + qwen35-122b-2box-observations.md."
