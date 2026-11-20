#!/usr/bin/env bash
# spark-tps.sh — live tokens/sec + in-flight requests for both DGX Sparks.
#
# Polls each box's Prometheus /metrics endpoint once per second and prints one
# refreshing line per Spark:
#
#   spark1 (192.168.1.147)   142.3 tok/s   2 running   0 waiting
#   spark2 (192.168.1.121)    38.7 tok/s   0 running   0 waiting
#
# Backends are auto-detected per box, so a vLLM slot and a llama.cpp/Unsloth
# slot can be watched side by side:
#
#   vLLM       vllm:generation_tokens_total       (counter -> delta/dt)
#              vllm:num_requests_{running,waiting}
#   llama.cpp  llamacpp:tokens_predicted_total    (counter -> delta/dt)
#              llamacpp:predicted_tokens_seconds   (gauge, preferred when present)
#              llamacpp:requests_{processing,deferred}
#
# The llama.cpp gauge is averaged over the window between two scrapes, so it is
# already a live rate and is used in preference to the counter delta. vLLM has
# no such gauge, so there the rate is the delta of the generation-token counter
# divided by the actual elapsed wall-clock between samples (self-correcting if a
# poll is slow).
#
# IMPORTANT — llama.cpp needs --metrics to serve /metrics at all. Without it the
# endpoint returns HTTP 501 and this script prints "metrics disabled" instead of
# pretending the box is down. Start the server with:
#   llama-server ... --metrics          # or LLAMA_ARG_ENDPOINT_METRICS=1
#
# Usage:  ./spark-tps.sh            # both boxes, refresh every 1s (in-place)
#         INTERVAL=2 ./spark-tps.sh # custom refresh
#         ./spark-tps.sh --once     # single sample line (no rates), for scripts
#         ./spark-tps.sh --scroll   # append a new row each tick, one column per Spark
#
# In --scroll mode the display does not overwrite itself: every tick prints a
# fresh row that scrolls up, with each Spark rendered as its own column and a
# leading clock column. Good for keeping a scrollback history / piping to a log.
#
# Targets spark1 :8001 (vLLM) and spark2 :8002 (llama.cpp / Unsloth GGUF).
# Edit BOXES below if ports/IPs change.

set -uo pipefail

# name|host:port
BOXES=(
  "spark1|192.168.1.147:8001"
  "spark2|192.168.1.121:8002"
)

INTERVAL="${INTERVAL:-1}"
CURL_TIMEOUT="${CURL_TIMEOUT:-5}"

# Pull /metrics for a box and emit "status gen running waiting gauge".
#   status: ok | nometrics (endpoint disabled) | down (unreachable)
#   gen:    generated-token counter, summed across label series
#   gauge:  llamacpp:predicted_tokens_seconds if exposed, else "-"
# Only ONE request is made per tick: a llama.cpp /metrics scrape resets the
# averaging bucket, so a second probe would zero out the rate.
scrape() {
  local hostport="$1" raw code
  raw=$(curl -sS --max-time "$CURL_TIMEOUT" -w '\n__HTTPCODE__%{http_code}\n' \
        "http://${hostport}/metrics" 2>/dev/null) || true
  code=$(printf '%s\n' "$raw" | sed -n 's/^__HTTPCODE__//p' | tail -1)

  printf '%s\n' "$raw" | awk -v code="$code" '
    /^__HTTPCODE__/                                { next }
    /^vllm:generation_tokens_total[ {]/            { gen  += $NF; have=1 }
    /^llamacpp:tokens_predicted_total[ {]/         { gen  += $NF; have=1 }
    /^vllm:num_requests_running[ {]/               { run  += $NF }
    /^llamacpp:requests_processing[ {]/            { run  += $NF }
    /^vllm:num_requests_waiting[ {]/               { wait += $NF }
    /^llamacpp:requests_deferred[ {]/              { wait += $NF }
    /^llamacpp:predicted_tokens_seconds[ {]/       { gauge = $NF; have_gauge=1 }
    END {
      if (have)          printf "ok %.0f %.0f %.0f %s\n", gen, run, wait, (have_gauge ? gauge : "-")
      else if (code=="501") print "nometrics - - - -"
      else               print "down - - - -"
    }'
}

now() { date +%s.%N; }

# Previous sample state, indexed by box order.
declare -a PREV_GEN PREV_T
for i in "${!BOXES[@]}"; do PREV_GEN[$i]=""; PREV_T[$i]=""; done

once_mode=0
scroll_mode=0
case "${1:-}" in
  --once)   once_mode=1 ;;
  --scroll) scroll_mode=1 ;;
esac

# Exit cleanly on Ctrl-C / SIGTERM (don't leave the cursor parked mid-redraw).
trap 'printf "\n"; exit 0' INT TERM

# ANSI: move cursor up N lines to redraw in place (skipped in --once mode).
N=${#BOXES[@]}
first=1

# Per-box column width for --scroll mode (each cell, excluding the separator).
CELL_W=24

# Print the scroll-mode header row once (clock column + one column per box).
print_scroll_header() {
  printf '%-8s' "time"
  for i in "${!BOXES[@]}"; do
    IFS='|' read -r name hostport <<<"${BOXES[$i]}"
    printf ' │ %-*s' "$CELL_W" "$name (${hostport%%:*})"
  done
  printf '\n'
}

[[ $scroll_mode -eq 1 ]] && print_scroll_header

while :; do
  # Collect this round's lines into an array, then print as a block.
  declare -a LINES=()
  declare -a CELLS=()
  t_now=$(now)

  for i in "${!BOXES[@]}"; do
    IFS='|' read -r name hostport <<<"${BOXES[$i]}"
    read -r status gen run wait gauge <<<"$(scrape "$hostport")"
    ip="${hostport%%:*}"

    if [[ "$status" == "down" ]]; then
      LINES+=("$(printf '%-7s (%-15s)   %s' "$name" "$ip" "DOWN / unreachable")")
      CELLS+=("$(printf '%-*s' "$CELL_W" "DOWN")")
      PREV_GEN[$i]=""; PREV_T[$i]="$t_now"
      continue
    fi

    if [[ "$status" == "nometrics" ]]; then
      LINES+=("$(printf '%-7s (%-15s)   %s' "$name" "$ip" \
        "metrics disabled — restart llama-server with --metrics")")
      CELLS+=("$(printf '%-*s' "$CELL_W" "no --metrics")")
      PREV_GEN[$i]=""; PREV_T[$i]="$t_now"
      continue
    fi

    rate="--"
    if [[ "$gauge" != "-" ]]; then
      # Backend reports its own live rate (llama.cpp, averaged over the
      # scrape window). Use it directly; no warm-up sample needed.
      rate=$(awk -v g="$gauge" 'BEGIN{ printf "%.1f", g }')
    elif [[ -n "${PREV_GEN[$i]}" ]]; then
      dt=$(awk -v a="$t_now" -v b="${PREV_T[$i]}" 'BEGIN{print a-b}')
      dg=$(( gen - ${PREV_GEN[$i]} ))
      rate=$(awk -v dg="$dg" -v dt="$dt" 'BEGIN{ if (dt>0) printf "%.1f", dg/dt; else printf "0.0" }')
    fi
    PREV_GEN[$i]="$gen"; PREV_T[$i]="$t_now"

    LINES+=("$(printf '%-7s (%-15s)  %8s tok/s   %2s running   %2s waiting' \
      "$name" "$ip" "$rate" "$run" "$wait")")
    CELLS+=("$(printf '%8s t/s %2sr %2sw' "$rate" "$run" "$wait")")
  done

  if [[ $once_mode -eq 1 ]]; then
    printf '%s\n' "${LINES[@]}"
    break
  fi

  if [[ $scroll_mode -eq 1 ]]; then
    # Append one row: leading clock column, then each box as a column.
    printf '%-8s' "$(date +%H:%M:%S)"
    for cell in "${CELLS[@]}"; do printf ' │ %-*s' "$CELL_W" "$cell"; done
    printf '\n'
    sleep "$INTERVAL"
    continue
  fi

  # Redraw block in place after the first paint.
  if [[ $first -eq 0 ]]; then printf '\033[%dA' "$N"; fi
  first=0
  for line in "${LINES[@]}"; do printf '\033[2K%s\n' "$line"; done

  sleep "$INTERVAL"
done
