#!/bin/bash
set -eu
trial="$HOME/inferssd-trial"
test -f "$trial/download-complete.json"
test -f "$trial/engine-resources.json"
if docker ps --format '{{.Names}}' | grep -qx qwen38-flash; then
    echo 'Stop qwen38-flash explicitly before running this trial.' >&2
    exit 1
fi
export PATH="$trial/venv/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$trial/deps/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export INFERSSD_HOME="$trial/state"
run="$(date -u +%Y%m%dT%H%M%SZ)"
exec "$trial/venv/bin/python" "$trial/launch.py" --execute \
    --resources "$trial/engine-resources.json" \
    --runtime-policy "$trial/runtime-policy.json" \
    --model "$trial/model/DeepSeek-V4.1-Flash-1017-Q8_0-00001-of-00010.gguf" \
    --context 65536 --batch 8192 --ubatch 8192 --port 8002 --prompt-cache off \
    --set host.headroom_mib=12288 \
    --output-dir "$trial/runs/$run" "$@"
