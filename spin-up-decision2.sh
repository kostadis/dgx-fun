#!/usr/bin/env bash
# Serve one Decision 2.0 model on spark2 with the vLLM Semantic Router runtime.
#   MODEL=Lux-9B PORT=8003 ./spin-up-decision2.sh      # also: Kai-0.6B, Nox-4B, Sol-2B, Eos-0.8B
#   BUILD=1 ...   rebuilds the image first (copies decision2/ and the pinned runtime source)
# API: POST http://$IP:$PORT/v1/systemone  (same body as Clef), GET /health, /v1/models, /metrics
set -euo pipefail
HOST=${HOST:-192.168.1.121}; MODEL=${MODEL:?set MODEL, e.g. Lux-9B}; PORT=${PORT:?set PORT}
PROFILE=${PROFILE:-exact}
NAME="decision2-$(echo "$MODEL" | tr 'A-Z.' 'a-z-')"
SR_SRC=${SR_SRC:?path to a semantic-router checkout (src/model-runtime is copied)}
if [[ "${BUILD:-0}" == 1 ]]; then
  rsync -a --delete "$(dirname "$0")/decision2/" "$HOST:~/decision2-build/"
  rsync -a --delete --exclude tests "$SR_SRC/src/model-runtime/" "$HOST:~/decision2-build/model-runtime/"
  ssh "$HOST" 'cd ~/decision2-build && docker build -q -t decision2-runtime .'
fi
ssh "$HOST" "docker rm -f $NAME >/dev/null 2>&1 || true; docker run -d --name $NAME --restart unless-stopped \
  --gpus all --ipc=host -p $PORT:$PORT -v \$HOME/.cache/huggingface:/root/.cache/huggingface \
  decision2-runtime vllm-sr/Decision-2.0-$MODEL --device cuda --host 0.0.0.0 --port $PORT --profile $PROFILE"
echo "started $NAME on $HOST:$PORT (profile $PROFILE); waiting for /health (downloads + golden-answer check)…"
for i in $(seq 1 120); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then curl -s "http://$HOST:$PORT/health"; echo; exit 0; fi
  ssh "$HOST" "docker ps --format '{{.Names}}' | grep -qx $NAME" || { ssh "$HOST" "docker logs --tail 40 $NAME"; exit 1; }
  sleep 10
done
echo "not healthy after 20 min"; ssh "$HOST" "docker logs --tail 40 $NAME"; exit 1
