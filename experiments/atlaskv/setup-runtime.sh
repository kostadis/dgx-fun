#!/usr/bin/env bash
# Run inside the private Python 3.12 / CUDA 13 helper image, with /work mounted.
set -euo pipefail
mkdir -p /work/source
tar -xf /work/source.tar -C /work/source
patch --batch --forward -d /work/source -p1 < /work/pruning-index.patch
python3 -m venv --system-site-packages /work/venv
/work/venv/bin/python -m pip install --disable-pip-version-check -r /work/runtime-requirements.txt
/work/venv/bin/python -m pip install --disable-pip-version-check --no-deps -e /work/source
/work/venv/bin/python - <<'PY'
import json,sys,torch,transformers,sentence_transformers
from pathlib import Path
Path('/work/runtime.json').write_text(json.dumps({
    'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
    'transformers':transformers.__version__,
    'sentence_transformers':sentence_transformers.__version__},indent=2)+'\n')
print('runtime ready',flush=True)
PY
