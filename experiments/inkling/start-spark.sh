#!/usr/bin/env bash
# Run on Spark 2. Keeps the existing Qwen and Clef services intact.
set -euo pipefail

INKLING_ROOT="${INKLING_ROOT:-$HOME/inkling-trial}"
INKLING_IMAGE="vllm/vllm-openai@sha256:ecdf0ccc663d86332459f5a50a38c30861a5079bc3924aa08fdc657ef10dd648"
INKLING_REVISION="b85c3cfa615fc413cc9af3379f491b963f3b586a"

if docker container inspect inkling2mini >/dev/null 2>&1; then
    echo 'inkling2mini already exists; inspect it before starting another instance.'
    exit 1
fi
python3 - <<'PY'
from pathlib import Path
mem = {line.split(':')[0]: int(line.split()[1]) for line in
       Path('/proc/meminfo').read_text().splitlines()}
if mem['MemAvailable'] < 46 * 1024 * 1024:
    raise SystemExit('Need at least 46 GiB currently available for this shared-host trial')
PY
mkdir -p "$INKLING_ROOT/cache" "$INKLING_ROOT/runtime"
docker run -d --name inkling2mini --gpus all --shm-size 8g \
    --memory 44g --memory-swap 44g --cpus 8 \
    -p 192.168.1.121:8004:8000 \
    -v "$INKLING_ROOT/cache:/hf" -e HF_HOME=/hf \
    -v "$INKLING_ROOT/runtime:/work" -e VLLM_CACHE_ROOT=/work/vllm-cache \
    "$INKLING_IMAGE" thinkingmachines/Inkling-2-Mini \
    --revision "$INKLING_REVISION" --served-model-name inkling-2-mini \
    --tensor-parallel-size 1 --max-model-len 65536 \
    --gpu-memory-utilization 0.26 --max-num-seqs 1 \
    --max-num-batched-tokens 4096 --enforce-eager \
    --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
    --limit-mm-per-prompt '{"image":0}' --host 0.0.0.0 --port 8000
