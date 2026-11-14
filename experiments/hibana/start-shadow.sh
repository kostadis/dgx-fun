#!/usr/bin/env bash
set -euo pipefail
profile=${1:-exact}
case "$profile" in exact|fast) ;; *) echo 'profile must be exact or fast' >&2; exit 2;; esac
trial_root="$HOME/hibana-trial"
test "$(hostname)" = gx10-3e5c
if docker ps --format '{{.Names}}' | grep -Eq '^(qwen38-flash|clef|clef-card-train)$'; then
  echo 'Stop the conflicting spark2 model deliberately before starting Hibana.' >&2
  exit 1
fi
if ss -lnt | grep -q ':30004 '; then
  echo 'Port 30004 is occupied.' >&2
  exit 1
fi
test "$(git -C "$trial_root/source" rev-parse HEAD)" = d474b74d5a892b2cd1cc8fc50cb17946e2a539bb
profile_env=()
if [ "$profile" = fast ]; then
  calibrated="$trial_root/results-fast/w3-checkpoints/flashnext-5c4c62a62853"
  test -f "$calibrated/manifest.json"
  profile_env+=(--setenv="HIBANA_W3_CHECKPOINT=$calibrated")
fi
systemd-run --user --unit="hibana-shadow-$profile" --collect --property=Restart=no \
  --working-directory="$trial_root/source" \
  --setenv="HIBANA_RESULTS=$trial_root/results-$profile" \
  --setenv=HIBANA_SSD_GIB=40 --setenv=HIBANA_HOST_FLOOR_GIB=16 \
  --setenv=HIBANA_KV_POOL_TOKENS=300032 --setenv=MAX_JOBS=2 \
  "${profile_env[@]}" \
  "$trial_root/bin/uv" run --locked hibana serve flashnext "--$profile" \
  --context-length 262144 --max-batch 4 --host 127.0.0.1 --port 30004
