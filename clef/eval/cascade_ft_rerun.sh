#!/usr/bin/env bash
# Rerun the legs contaminated by a concurrent spark1 client (21:24-~22:20): E4 h1 small+cascade, qwen alone on both tasks.
# (Ran once, 2026-10-11/12. The load monitor now appends, so MARK lines survive; the original run split legs by rerun.log timestamps.)
set -euo pipefail
DATA=$HOME/data/decision-eval/v1; E8=$DATA/results/e8; L=$DATA/cascade/live; HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p $L/e3/contaminated $L/e4_facts/contaminated
mv $L/e4_facts/{small,cascade}-c{1,8}-ft-typing-strict-h1.json $L/e4_facts/contaminated/
mv $L/e3/big-c{1,8}-ft.json $L/e3/contaminated/; mv $L/e4_facts/big-c{1,8}-ft.json $L/e4_facts/contaminated/
# spark1 load monitor: running/waiting every 2s
( while true; do echo "$(date +%s) $(curl -s -m3 http://192.168.1.147:8001/metrics | grep -E '^vllm:num_requests_(running|waiting)\{' | awk '{print $2}' | tr '\n' ' ')"; sleep 2; done ) >> $E8/rerun-spark1-load.log &
MON=$!; trap "kill $MON" EXIT
mark() { echo "MARK $(date +%s) $*" | tee -a $E8/rerun-spark1-load.log; }
ssh spark2 "docker logs nox-tuned 2>&1 | grep -m1 'TUNED HEAD loaded'" | grep -q head-typing-strict-h1
python3 $HERE/ft_probe.py e4_facts typing-strict 1 20
g=$E8/live-gates-e4_facts-typing-strict-h1.json
for c in 1 8; do for m in small cascade; do
  mark "start e4 $m c$c"; python3 $HERE/cascade_live.py $m e4_facts --conc $c --gates $g --tag="-ft-typing-strict-h1"
done; done
for part in e3 e4_facts; do for c in 1 8; do
  mark "start $part big c$c"; python3 $HERE/cascade_live.py big $part --conc $c --gates $E8/live-gates-$part-all.json --tag=-ft
done; done
mark "ALL DONE"
