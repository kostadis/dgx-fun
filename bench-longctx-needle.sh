#!/usr/bin/env bash
#
# bench-longctx-needle.sh — long-context "needle in a haystack" QUALITY
# probe for any OpenAI-compatible /v1/chat/completions endpoint.
#
# WHY THIS EXISTS
#   KV-cache quantization (fp8, TurboQuant turboquant_k8v4/4bit_nc/...)
#   only touches the KV cache, so it can ONLY degrade quality at long
#   context — short prompts barely populate the cache and always look
#   fine. To know whether a KV-quant config "worked" you must:
#     1. plant a unique fact deep inside a long prompt,
#     2. ask the model to recall it,
#     3. do it at several context lengths, and
#     4. compare the SAME run against the plain-fp8 config (revert + rerun).
#   If TurboQuant misses a needle that fp8 catches, the compression is
#   too aggressive for your use. If both recall it, quality held.
#
# This measures QUALITY (did it recall) + TTFT. Pair with bench-prefill.sh
# for the speed-only sweep.
#
# USAGE
#   bash bench-longctx-needle.sh <model-id>
#   # A/B example:
#   ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b-turboquant.sh'
#   bash bench-longctx-needle.sh Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 | tee tq.txt
#   ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b.sh'   # revert to fp8
#   bash bench-longctx-needle.sh Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 | tee fp8.txt
#   diff fp8.txt tq.txt
#
# CONFIGURATION (env vars)
#   HOST     default 192.168.1.147 (the Spark)
#   PORT     default 8001          (vllm-chat)
#   LENGTHS  default "8192 32768 65536 120000"  approx input tokens.
#            Keep below --max-model-len (128K here). 120000 also doubles
#            as the #41726 long-chunked-prefill CRASH probe.
#   DEPTHS   default "0.25 0.5 0.9"  where in the prompt to plant the
#            needle, as a fraction (start..end). Late depths stress the
#            cache hardest.
#
# OUTPUT (per length × depth)
#   len      depth  TTFT_s  verdict  answer
#   8192     0.50   1.12    PASS     CRIMSON-FALCON-7294
#   120000   0.90   58.30   FAIL     "I don't see a passphrase..."
#
# A run that ERRORs or NO_TOKENs at 120000 (and not at shorter lengths)
# is the #41726 chunked-prefill crash signature — confirm in the
# container logs and retry with ENFORCE_EAGER=1 / lower MAX_BATCHED.
#
set -euo pipefail

MODEL="${1:-}"
if [ -z "${MODEL}" ]; then
  echo "usage: bash bench-longctx-needle.sh <model-id>" >&2
  exit 2
fi

HOST="${HOST:-192.168.1.147}"
PORT="${PORT:-8001}"
LENGTHS="${LENGTHS:-8192 32768 65536 120000}"
DEPTHS="${DEPTHS:-0.25 0.5 0.9}"

python3 -u - "${MODEL}" "${HOST}" "${PORT}" "${LENGTHS}" "${DEPTHS}" <<'PY'
import json, random, sys, time, urllib.request

model, host, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
lengths = [int(x) for x in sys.argv[4].split()]
depths  = [float(x) for x in sys.argv[5].split()]

# Unique, training-set-novel secret so recall can't come from priors.
SECRET = "CRIMSON-FALCON-7294"
NEEDLE = f"\n\n>>> IMPORTANT: The secret passphrase is {SECRET}. Remember it. <<<\n\n"
QUESTION = ("\n\nQuestion: What is the secret passphrase mentioned somewhere "
            "above? Reply with ONLY the passphrase, nothing else.")

WORDS = ['the','quick','brown','fox','jumps','over','lazy','dog','pack','my',
         'box','with','five','dozen','liquor','jugs','amid','vast','silent','plains']

print(f"# bench-longctx-needle  model={model}")
print(f"# endpoint=http://{host}:{port}/v1/chat/completions  secret={SECRET}")
print(f"# {'len':>7} {'depth':>6} {'TTFT_s':>8} {'verdict':>8}  answer")

for ilen in lengths:
    random.seed(ilen)
    filler = [random.choice(WORDS) for _ in range(ilen)]
    for depth in depths:
        toks = list(filler)
        pos = max(0, min(len(toks), int(len(toks) * depth)))
        prompt = ' '.join(toks[:pos]) + NEEDLE + ' '.join(toks[pos:]) + QUESTION
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 40,
            "temperature": 0,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            f"http://{host}:{port}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"})
        t0 = time.time(); first = None; out = []
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                for raw in r:
                    if not raw.startswith(b"data: "):
                        continue
                    payload = raw[len(b"data: "):].strip()
                    if payload == b"[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"]
                    except Exception:
                        continue
                    text = delta.get("content") or ""
                    if text and first is None:
                        first = time.time()
                    out.append(text)
        except Exception as e:
            print(f"  {ilen:>7} {depth:>6.2f} {'ERROR':>8} {'ERROR':>8}  {e}")
            continue
        ans = ''.join(out).strip()
        ttft = f"{first - t0:.2f}" if first else "NO_TOK"
        verdict = "PASS" if SECRET in ans else "FAIL"
        print(f"  {ilen:>7} {depth:>6.2f} {ttft:>8} {verdict:>8}  {ans[:60]!r}")
PY
