#!/usr/bin/env bash
# spark-tps.sh — live tokens/sec + in-flight requests for both DGX Sparks.
#
# Polls each box's vLLM Prometheus /metrics endpoint once per second and prints
# one refreshing line per Spark:
#
#   spark1 (192.168.1.147)   142.3 tok/s   2 running   0 waiting
#   spark2 (192.168.1.121)     0.0 tok/s   0 running   0 waiting
#
# tok/s is the delta of vllm:generation_tokens_total divided by the actual
# elapsed wall-clock between samples (so it self-corrects if a poll is slow).
# "running" / "waiting" are the live vllm:num_requests_{running,waiting} gauges.
#
# Usage:  ./spark-tps.sh            # both boxes, refresh every 1s
#         INTERVAL=2 ./spark-tps.sh # custom refresh
#         ./spark-tps.sh --once     # single sample line (no rates), for scripts
#
# Targets the vllm-chat slot on :8001. Edit BOXES below if ports/IPs change.

set -uo pipefail

# name|host:port
BOXES=(
  "spark1|192.168.1.147:8001"
  "spark2|192.168.1.121:8001"
)

INTERVAL="${INTERVAL:-1}"
CURL_TIMEOUT="${CURL_TIMEOUT:-5}"

# Pull /metrics for a box and emit "gen running waiting" on one line.
# gen = generated-token counter (summed across label series; "-" if unreachable).
scrape() {
  local hostport="$1"
  curl -sS --max-time "$CURL_TIMEOUT" "http://${hostport}/metrics" 2>/dev/null | awk '
    /^vllm:generation_tokens_total[ {]/ { gen += $NF; have_gen=1 }
    /^vllm:num_requests_running[ {]/     { run += $NF }
    /^vllm:num_requests_waiting[ {]/     { wait += $NF }
    END {
      if (have_gen) printf "%.0f %.0f %.0f\n", gen, run, wait
      else          print "- - -"
    }'
}

now() { date +%s.%N; }

# Previous sample state, indexed by box order.
declare -a PREV_GEN PREV_T
for i in "${!BOXES[@]}"; do PREV_GEN[$i]=""; PREV_T[$i]=""; done

once_mode=0
[[ "${1:-}" == "--once" ]] && once_mode=1

# Exit cleanly on Ctrl-C / SIGTERM (don't leave the cursor parked mid-redraw).
trap 'printf "\n"; exit 0' INT TERM

# ANSI: move cursor up N lines to redraw in place (skipped in --once mode).
N=${#BOXES[@]}
first=1

while :; do
  # Collect this round's lines into an array, then print as a block.
  declare -a LINES=()
  t_now=$(now)

  for i in "${!BOXES[@]}"; do
    IFS='|' read -r name hostport <<<"${BOXES[$i]}"
    read -r gen run wait <<<"$(scrape "$hostport")"

    if [[ "$gen" == "-" ]]; then
      LINES+=("$(printf '%-7s (%-15s)   %s' "$name" "${hostport%%:*}" "DOWN / unreachable")")
      PREV_GEN[$i]=""; PREV_T[$i]="$t_now"
      continue
    fi

    rate="--"
    if [[ -n "${PREV_GEN[$i]}" ]]; then
      dt=$(awk -v a="$t_now" -v b="${PREV_T[$i]}" 'BEGIN{print a-b}')
      dg=$(( gen - ${PREV_GEN[$i]} ))
      rate=$(awk -v dg="$dg" -v dt="$dt" 'BEGIN{ if (dt>0) printf "%.1f", dg/dt; else printf "0.0" }')
    fi
    PREV_GEN[$i]="$gen"; PREV_T[$i]="$t_now"

    LINES+=("$(printf '%-7s (%-15s)  %8s tok/s   %2s running   %2s waiting' \
      "$name" "${hostport%%:*}" "$rate" "$run" "$wait")")
  done

  if [[ $once_mode -eq 1 ]]; then
    printf '%s\n' "${LINES[@]}"
    break
  fi

  # Redraw block in place after the first paint.
  if [[ $first -eq 0 ]]; then printf '\033[%dA' "$N"; fi
  first=0
  for line in "${LINES[@]}"; do printf '\033[2K%s\n' "$line"; done

  sleep "$INTERVAL"
done
