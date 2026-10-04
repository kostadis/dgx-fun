#!/usr/bin/env bash
#
# spin-up-clef.sh
#
# Bring up Cloudflare's Clef + Clef-flash decision models (Jev/SystemOne API)
# on ONE box (spark2 by default), on :8002.
#
# Run this FROM THE WORKSTATION (it SSHes to $HOST).
#
#     STOP_CHAT=1 ./spin-up-clef.sh                    # spark2, both models
#     STOP_CHAT=1 CLEF_MODELS=clef-flash ./spin-up-clef.sh   # flash only
#
# ---------------------------------------------------------------------------
# WHAT THIS IS
#   Clef (Qwen3.8-27B backbone, 55 GB BF16) and Clef-flash (Qwen3.5-9B, 19 GB)
#   are NOT generative: one prefill pass, then a small "joint schema head"
#   scores every allowed answer of every typed question. vLLM cannot serve
#   them (the head is custom code; `vllm serve Cloudflare/clef` on the HF card
#   is HuggingFace's auto-generated widget and would serve a bare chat
#   backbone). So: plain transformers + the release's own joint_schema_model.py,
#   behind clef/server.py. Image built from clef/Dockerfile on the box:
#       scp clef/Dockerfile clef/server.py $HOST:~/clef/ && ssh $HOST 'cd ~/clef && docker build -t clef-server .'
#
# WHAT THIS REPLACES
#   Both weights (~74 GB) do not fit beside qwen38-flash (~97 GB reservation,
#   ~13 GB host free), so this needs the box to itself. STOP_CHAT=1 is the
#   explicit opt-in to `docker rm -f qwen38-flash`. Ollama :11434 is untouched.
#
# REVERT (checkpoint + image still on the box — a load, not a download):
#     ssh spark2 'docker rm -f clef'; HOST=spark2 ./spin-up-vllm-qwen38-flash-next.sh
# ---------------------------------------------------------------------------
set -euo pipefail

HOST="${HOST:-spark2}"
PORT="${PORT:-8002}"
CLEF_MODELS="${CLEF_MODELS:-clef-flash,clef}"
CLEF_MAX_LENGTH="${CLEF_MAX_LENGTH:-16384}"
STOP_CHAT="${STOP_CHAT:-0}"

case "$HOST" in
  spark|spark1) IP=192.168.1.147 ;;
  spark2)       IP=192.168.1.121 ;;
  *) echo "unknown HOST=$HOST" >&2; exit 1 ;;
esac

echo "== preflight on $HOST"
ssh "$HOST" 'docker image inspect clef-server >/dev/null' \
  || { echo "image clef-server missing on $HOST — build it (see header)" >&2; exit 1; }
for m in ${CLEF_MODELS//,/ }; do
  ssh "$HOST" "ls ~/.cache/huggingface/hub/models--Cloudflare--$m/snapshots/*/joint_head.safetensors >/dev/null 2>&1" \
    || { echo "checkpoint Cloudflare/$m not downloaded on $HOST" >&2; exit 1; }
done

if ssh "$HOST" 'docker ps --format "{{.Names}}" | grep -qx qwen38-flash'; then
  if [[ "$STOP_CHAT" != 1 ]]; then
    echo "qwen38-flash is running on $HOST and both cannot fit. Re-run with STOP_CHAT=1." >&2
    exit 1
  fi
  echo "== stopping qwen38-flash on $HOST"
  ssh "$HOST" 'docker rm -f qwen38-flash'
fi

echo "== starting clef ($CLEF_MODELS) on $HOST:$PORT"
ssh "$HOST" "docker rm -f clef >/dev/null 2>&1 || true; docker run -d --name clef \
  --gpus all --restart unless-stopped \
  -p $PORT:$PORT \
  -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
  -e HF_HUB_OFFLINE=1 \
  -e CLEF_MODELS=$CLEF_MODELS -e CLEF_MAX_LENGTH=$CLEF_MAX_LENGTH -e PORT=$PORT \
  clef-server"

echo "== waiting for /health"
for _ in $(seq 1 120); do
  if curl -fsS "http://$IP:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! ssh "$HOST" 'docker ps --format "{{.Names}}" | grep -qx clef'; then
    echo "container exited:" >&2; ssh "$HOST" 'docker logs --tail 40 clef' >&2; exit 1
  fi
  sleep 10
done
curl -fsS "http://$IP:$PORT/health"; echo

echo "== smoke"
for m in ${CLEF_MODELS//,/ }; do
  curl -sS -D - "http://$IP:$PORT/v1/systemone" -H 'content-type: application/json' -d '{
    "model": "'"$m"'",
    "state": "Checkout has been failing for every customer for the last hour.",
    "questions": {
      "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
      "team": {"type": "choice", "instructions": "Which team should handle this request?",
               "criteria": {"billing": "Payments, invoices, and refunds", "technical": "Outages, errors, and configuration", "sales": "Plans and upgrades"}},
      "severity": {"type": "score", "instructions": "How severe is the customer impact?",
                   "criteria": ["No impact", "Minor", "Major", "Critical"]}
    }}' | grep -iE '^x-clef-latency|^\{'
  echo
done
