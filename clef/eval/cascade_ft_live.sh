#!/usr/bin/env bash
# Live check of the tuned-Nox cascade (cascade_ft.py gates DIR wrote the frozen gates + prereg.json first).
# For each (part, variant, half): serve Nox on spark2:8005 with that half's tuned head (serve_tuned.py),
# probe it against the replay, then run Nox alone and the cascade at c=1 and c=8 on that half's rows.
# Then qwen alone (spark1:8001, deployed config) on every row, c=1 and c=8, in the same session.
# Outputs: $DATA/cascade/live/<part>/<mode>-c<N>-ft-<variant>-h<H>.json and big-c<N>-ft.json
set -euo pipefail
DATA="${DATA:-$HOME/data/decision-eval/v1}"; E8="$DATA/results/e8"; HERE="$(cd "$(dirname "$0")" && pwd)"
scp -q "$HERE/../finetune/serve_tuned.py" spark2:ft-v1/
# the heads were written on spark2 by cascade_head.py; check they are the pre-registered ones
python3 - "$E8" <<'PY'
import json, subprocess, sys
reg = json.load(open(sys.argv[1] + '/prereg.json'))['runs']
for k, r in reg.items():
    if not r.get('head'): continue
    got = subprocess.check_output(['ssh', 'spark2', f"sha256sum ft-v1/{r['head']}"], text=True).split()[0]
    assert got == r['head_sha256'], (k, got); print('head ok', k, got[:12])
PY
serve() {  # head file
  ssh spark2 "docker rm -f nox-tuned >/dev/null 2>&1 || true; docker run -d --name nox-tuned --gpus all --ipc=host -p 8005:8005 \
    -v \$HOME/.cache/huggingface:/root/.cache/huggingface -v \$HOME/ft-v1:/ft -e HF_HUB_OFFLINE=1 --entrypoint python3 \
    decision2-runtime /ft/serve_tuned.py /ft/$1 vllm-sr/Decision-2.0-Nox-4B --device cuda --host 0.0.0.0 --port 8005 --profile exact" >/dev/null
  for i in $(seq 1 90); do curl -fsS http://192.168.1.121:8005/health 2>/dev/null | grep -q '"ready"' && break; sleep 10; done
  sleep 5; ssh spark2 "docker logs nox-tuned 2>&1 | grep -m1 'TUNED HEAD loaded'"
}
for spec in e3:triage-lenient e4_facts:typing-strict; do
  part=${spec%%:*}; var=${spec#*:}
  for h in 0 1; do
    echo "== $part $var h$h  $(date -Is)"
    serve "head-$var-h$h.safetensors"
    python3 "$HERE/ft_probe.py" "$part" "$var" "$h" 20
    g="$E8/live-gates-$part-$var-h$h.json"
    for c in 1 8; do for m in small cascade; do
      python3 "$HERE/cascade_live.py" $m "$part" --conc $c --gates "$g" --tag="-ft-$var-h$h"
    done; done
  done
done
for part in e3 e4_facts; do for c in 1 8; do
  python3 "$HERE/cascade_live.py" big "$part" --conc $c --gates "$E8/live-gates-$part-all.json" --tag=-ft
done; done
echo "ALL DONE $(date -Is)"
