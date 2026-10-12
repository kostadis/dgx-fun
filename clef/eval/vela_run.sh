#!/usr/bin/env bash
# Frozen E3/E4 outputs for Vela 2.0 members, one at a time on spark2:8006 (serve_vela.py in the vela2-runtime image =
# decision2-runtime + transformers 5.17.0). Each member is checked against its release's recorded card output first.
# Writes $DATA/results/<part>/vela-*.json via run.py (single-stream, like every SystemOne model).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; SCR="${SCR:-/tmp}"
scp -q "$HERE/../finetune/serve_vela.py" spark2:vela2/
HUB=/root/.cache/huggingface/hub
CODE_MD5="494a7779b79a7c9aa1c8c0fef8abcbe1 731c5234eb5adf71917544ffa9ef1e56 85260056a49194a77254b008004f96c8 f5700f4ebb07444c2576c6a31dab3efb"  # reviewed 2026-10-12
declare -A REV=([0.8B]=326b01d81f61fff7fd4ae570e54b87a49119e490 [4B]=d692394127f6f87fc581d8634c9558fe4776efce [9B]=068425bb145c05884a98b870c219047da22d9aea)
serve() {  # size abstain-mode
  until ssh spark2 "grep -q \"DL DONE\" ~/vela2/dl.log"; do sleep 30; done
  local snap=$HUB/models--vllm-sr--Vela-2.0-$1/snapshots/${REV[$1]}
  # integrity: every shipped file but the README (edited after SHA256SUMS) and the images (not downloaded)
  ssh spark2 "cd ~/.cache/huggingface/hub/models--vllm-sr--Vela-2.0-$1/snapshots/${REV[$1]} && \
    grep -v -E ' (README.md|assets/.*)$' SHA256SUMS | sha256sum -c --quiet && md5sum *.py" | { tee -a "${LOG:-/dev/null}"; } | \
    awk '{print $1}' | sort | tr '\n' ' ' | grep -q "$CODE_MD5" || { echo "code differs from the reviewed files"; exit 1; }
  ssh spark2 "docker rm -f vela2 >/dev/null 2>&1 || true; docker run -d --name vela2 --gpus all --ipc=host -p 8006:8006 \
    -v \$HOME/.cache/huggingface:/root/.cache/huggingface -v \$HOME/vela2:/v -e HF_HUB_OFFLINE=1 vela2-runtime \
    python3 /v/serve_vela.py $snap --abstain $2 --device cuda --host 0.0.0.0 --port 8006 --name Vela-2.0-$1" >/dev/null
  for i in $(seq 1 120); do curl -fsS -m3 http://192.168.1.121:8006/ready >/dev/null 2>&1 && break; sleep 5; done
  ssh spark2 "cat ~/.cache/huggingface/hub/models--vllm-sr--Vela-2.0-$1/snapshots/${REV[$1]}/card_examples.json" > "$SCR/card-$1.json"
}
for spec in 0.8B:default:vela-0.8b 4B:default:vela-4b 9B:default:vela-9b 4B:off:vela-4b-noabs; do
  IFS=: read -r size mode model <<< "$spec"
  echo "== $model ($size, abstain=$mode) $(date -Is)"
  serve "$size" "$mode"
  if [ "$mode" = default ]; then python3 "$HERE/vela_check.py" http://192.168.1.121:8006/v1/systemone "Vela-2.0-$size" "$SCR/card-$size.json"; fi
  python3 "$HERE/run.py" "$model" e3 e4
done
ssh spark2 "docker rm -f vela2" >/dev/null; echo "ALL DONE $(date -Is)"
