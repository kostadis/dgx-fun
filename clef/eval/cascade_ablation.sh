#!/usr/bin/env bash
# Serving-config ablation for the cascade paper: is qwen's run-to-run disagreement (and its latency)
# due to MTP speculative decoding or prefix caching?  Run from the workstation, AFTER the baseline
# schedule (MTP=2, prefix caching on) has finished. Restarts spark1's qwen38-flash twice and restores
# the documented config (spin-up defaults: MTP=2, PREFIX_CACHE=1) at the end.
#
#   -rep          baseline config, qwen alone single-stream on e3 again (same-config repeat)
#   -mtp0         MTP=0, prefix caching on
#   -mtp0-apc0    MTP=0, prefix caching off
set -euo pipefail
cd "$(dirname "$0")"
LOG=${LOG:-$HOME/data/decision-eval/v1/cascade/ablation.log}
SPIN=../../spin-up-vllm-qwen38-flash-next.sh
PARTS="e4_ent e3 e4_facts e2"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
restore() { say "FAILED at line $1 -- restoring documented config"; "$SPIN" >> "$LOG" 2>&1 || say "!! restore failed"; say "ABLATION FAILED (restore attempted)"; }
trap 'restore $LINENO' ERR

until grep -q DONE "$HOME/data/decision-eval/v1/cascade/live.log"; do sleep 30; done
say "baseline schedule done"

python3 cascade_live.py big e3 --conc 1 --tag=-rep >> "$LOG" 2>&1

phase() {  # tag, env for spin-up
  local tag=$1; shift
  say "restart spark1 qwen38-flash with: $*"
  env "$@" "$SPIN" >> "$LOG" 2>&1
  ssh spark 'docker logs qwen38-flash 2>&1 | grep -oE "num_speculative_tokens.{0,4}|enable_prefix_caching=[A-Za-z]+" | sort -u' | tee -a "$LOG"
  for c in 1 8; do for m in big cascade; do
    python3 cascade_live.py $m $PARTS --conc $c --tag="$tag" >> "$LOG" 2>&1
  done; done
  python3 cascade_live.py big e3 --conc 1 --tag="$tag-rep" >> "$LOG" 2>&1
}
phase -mtp0      MTP=0
phase -mtp0-apc0 MTP=0 PREFIX_CACHE=0

say "restoring documented config (MTP=2, PREFIX_CACHE=1)"
"$SPIN" >> "$LOG" 2>&1
ssh spark 'docker logs qwen38-flash 2>&1 | grep -oE "num_speculative_tokens.{0,4}|enable_prefix_caching=[A-Za-z]+" | sort -u' | tee -a "$LOG"
say "ABLATION DONE"
