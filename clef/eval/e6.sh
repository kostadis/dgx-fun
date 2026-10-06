#!/usr/bin/env bash
# E6: fine-tune Decision-2.0-Nox-4B on the frozen dataset's GM rulings, on spark2.
#
#     ./e6.sh            # DATA=~/data/decision-eval/v1 (default)
#
# Copies the frozen E6 items + clef/finetune/*.py to spark2:~/ft-<version>/ and runs each stage
# in a throwaway decision2-runtime container (same image/runtime as the server, so cached
# features reproduce served probabilities). Logs land in $DATA/results/e6/.
# Needs ~20-40 GB free on spark2 beside whatever is serving; run it with Clef stopped.
set -euo pipefail
DATA="${DATA:-$HOME/data/decision-eval/v1}"
HOST="${HOST:-spark2}"
V="$(basename "$DATA")"
HERE="$(cd "$(dirname "$0")" && pwd)"
HEAD=/root/.cache/huggingface/hub/models--vllm-sr--Decision-2.0-Nox-4B/snapshots/ce1bdc9d91333aae2bf496ec48c66e1a913eb0a0/decision_head.safetensors
OUT="$DATA/results/e6"; mkdir -p "$OUT"

ssh "$HOST" "mkdir -p ~/ft-$V"
scp -q "$HERE"/../finetune/{extract,train_head,train_lora}.py "$DATA"/e6_{triage,typing}_items.json "$HOST:ft-$V/"

run() {  # name, command
  local name="$1"; shift
  [ -s "$OUT/$name.log" ] && { echo "skip $name (log exists)"; return; }
  echo "== $name: $*"
  ssh "$HOST" "docker run --rm --gpus all --ipc=host -v \$HOME/.cache/huggingface:/root/.cache/huggingface \
      -v \$HOME/ft-$V:/ft -w /ft -e HF_HUB_OFFLINE=1 --entrypoint python3 decision2-runtime $*" 2>&1 \
    | grep -v -i 'warn' | tee "$OUT/$name.log.tmp"
  mv "$OUT/$name.log.tmp" "$OUT/$name.log"
}

run extract_triage extract.py e6_triage_items.json triage_feats.pt
run extract_typing extract.py e6_typing_items.json typing_feats.pt
run head_triage train_head.py triage_feats.pt --task triage --head $HEAD --folds 3 --lr 1e-4 --l2sp 0.1 --epochs 20
run head_typing train_head.py typing_feats.pt --task typing --head $HEAD --folds 5 --lr 1e-4 --l2sp 0.1 --epochs 20
run lora_triage train_lora.py e6_triage_items.json --task triage --folds 3 --rank 16 --epochs 2
run lora_typing train_lora.py e6_typing_items.json --task typing --folds 5 --rank 16 --epochs 1
scp -q "$HOST:ft-$V/typing_entity_preds.json" "$HOST:ft-$V/lora_*_preds.pt" "$OUT/" 2>/dev/null || true
echo "E6 done -> $OUT"
