#!/usr/bin/env bash
set -euo pipefail
test "$(hostname)" = gx10-3e5c
for unit in hibana-fast-trial hibana-shadow-exact hibana-shadow-fast; do
  if systemctl --user is-active --quiet "$unit"; then
    systemctl --user stop "$unit"
  fi
done
if ss -lnt | grep -q ':30004 '; then
  echo 'Shadow port is still occupied; Clef was not started.' >&2
  exit 1
fi
docker start clef-card-train
