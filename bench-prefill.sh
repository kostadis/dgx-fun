#!/usr/bin/env bash
#
# bench-prefill.sh — measure prefill cost (time-to-first-token) at
# varying input lengths against any OpenAI-compatible /v1/chat/completions
# endpoint. Designed to compare backends back-to-back on the same box.
#
# USAGE
#   bash bench-prefill.sh <model-id>
#
#   # examples:
#   bash bench-prefill.sh google/gemma-4-26b-a4b-it
#   bash bench-prefill.sh nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
#
# CONFIGURATION (env vars)
#   HOST     default 192.168.1.147 (the Spark)
#   PORT     default 8001          (vllm-chat)
#   LENGTHS  default "1024 8192 32768 65536 100000"
#            Space-separated approximate input token counts. Keep below
#            the binding --max-model-len (Gemma 4 longctx = 131072,
#            Nemotron = 262144).
#   OUTFILE  optional path to also tee the output to (default: stdout only)
#
# OUTPUT
#   Tab-aligned table:
#     input_tokens   TTFT_s
#             1024     0.42
#             8192     1.15
#            32768     3.89
#            ...
#
# NOTES
#   - "Input tokens" is approximate. Prompt is built as N random short
#     words, which tokenise ~1:1 for both Gemma 4 and Nemotron, ±10%.
#   - TTFT is measured from request send to first non-empty SSE delta
#     containing either `content` or `reasoning_content`. For reasoning
#     models (Nemotron) this captures "time until the model starts
#     emitting anything" — including the think phase — which is the
#     latency the user actually feels.
#   - Network adds ~1 ms LAN RTT — noise at the scales we care about.
#
set -euo pipefail

MODEL="${1:-}"
if [ -z "${MODEL}" ]; then
  echo "usage: bash bench-prefill.sh <model-id>" >&2
  echo "  e.g. bash bench-prefill.sh google/gemma-4-26b-a4b-it" >&2
  exit 2
fi

HOST="${HOST:-192.168.1.147}"
PORT="${PORT:-8001}"
LENGTHS="${LENGTHS:-1024 8192 32768 65536 100000}"
OUTFILE="${OUTFILE:-}"

run() {
  python3 -u - "${MODEL}" "${HOST}" "${PORT}" ${LENGTHS} <<'PY'
import json, random, sys, time, urllib.request

model, host, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
lengths = [int(x) for x in sys.argv[4:]]

print(f"# bench-prefill")
print(f"# model={model}")
print(f"# endpoint=http://{host}:{port}/v1/chat/completions")
print(f"# {'input_tokens':>12} {'TTFT_s':>10}")

WORDS = ['the','quick','brown','fox','jumps','over','lazy','dog',
         'pack','my','box','with','five','dozen','liquor','jugs']

for ilen in lengths:
    print(f"[bench-prefill] running input_tokens~={ilen}...", file=sys.stderr, flush=True)
    random.seed(ilen)
    prompt = ' '.join(random.choice(WORDS) for _ in range(ilen))
    body = json.dumps({
        "model": model,
        "messages": [{"role":"user","content": prompt + "\nReply with just OK."}],
        "max_tokens": 32,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=body, headers={"Content-Type":"application/json"})
    t0 = time.time()
    first = None
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                if not raw.startswith(b"data: "):
                    continue
                payload = raw[len(b"data: "):].strip()
                if payload == b"[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                try:
                    delta = obj["choices"][0]["delta"]
                except (KeyError, IndexError, TypeError):
                    continue
                # First non-empty token-bearing chunk. vLLM emits reasoning
                # tokens under different field names depending on the parser:
                #   - OpenAI convention: "reasoning_content"
                #   - nano_v3 parser (Nemotron 3 Nano): "reasoning"
                # Match all three so this script gives a fair "time to
                # first emitted token" reading across model families.
                text = (delta.get("content")
                        or delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or "")
                if text:
                    first = time.time()
                    break
    except Exception as e:
        print(f"  {ilen:>12} ERROR: {e}")
        continue

    if first is None:
        print(f"  {ilen:>12} {'NO_TOKEN':>10}")
    else:
        print(f"  {ilen:>12} {first-t0:>10.2f}")
PY
}

if [ -n "${OUTFILE}" ]; then
  run | tee "${OUTFILE}"
else
  run
fi
