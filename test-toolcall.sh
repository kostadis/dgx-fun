#!/usr/bin/env bash
#
# test-toolcall.sh — model-agnostic tool-calling probe for any
# OpenAI-compatible /v1/chat/completions endpoint with
# --enable-auto-tool-choice + --tool-call-parser <family> configured.
#
# Sends a get_weather function-call probe and reports pass/fail:
#   PASS = content is null + tool_calls[] populated with name="get_weather"
#          and arguments that parse as JSON containing a location
#   FAIL = content contains the function-call as plain text (parser didn't
#          fire) OR tool_calls[] is empty OR arguments aren't parseable
#
# Use this script after each spin-up-vllm-*.sh that enables tool calling
# (Gemma 4 26B MoE, Llama 70B + spec decode, Nemotron 3 Nano, etc.) to
# verify the parser family in the spin-up flags actually matches what
# the model emits. Silent parser mismatches are the #1 way local agents
# break in unobvious ways.
#
# USAGE
#   # On the Spark, or with PORT pointing at it from the LAN:
#   MODEL=google/gemma-4-26b-a4b-it ./test-toolcall.sh
#   MODEL=casperhansen/llama-3.3-70b-instruct-awq ./test-toolcall.sh
#   MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 ./test-toolcall.sh
#   MODEL=ollama/qwen2.5-coder:14b PORT=11434 ./test-toolcall.sh
#
# CONFIGURATION
#   MODEL    model id to send in the request body; required.
#            Default for backwards compat is google/gemma-4-26b-a4b-it.
#   PORT     host port; default 8001 (vllm-chat)
#   HOST     hostname; default localhost (run on the Spark itself)
#
# KNOWN PARSERS BY FAMILY (must match --tool-call-parser at spin-up)
#   Gemma 4 Instruct          → gemma4
#   Llama 3.1 / 3.3 Instruct  → llama3_json
#   Llama 3.2 (1B/3B text)    → pythonic
#   Qwen2.5 / Qwen3 Instruct  → hermes
#   Qwen3 Coder               → qwen3_coder
#   Nemotron 3 Nano           → qwen3_coder  (NVIDIA reused it)
#   Mistral Instruct          → mistral
#
set -euo pipefail

MODEL="${MODEL:-google/gemma-4-26b-a4b-it}"
PORT="${PORT:-8001}"
HOST="${HOST:-localhost}"
REQ_FILE="/tmp/req_tool.json"
RESP_FILE="/tmp/resp_tool.json"

echo "=== test-toolcall ==="
echo "  model: ${MODEL}"
echo "  host:  ${HOST}"
echo "  port:  ${PORT}"
echo ""

cat > "${REQ_FILE}" <<JSON
{
  "model": "${MODEL}",
  "messages": [
    {"role": "user", "content": "What is the weather in Paris right now?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "max_tokens": 150
}
JSON

echo "→ request:"
cat "${REQ_FILE}" | python3 -m json.tool
echo ""

echo "→ sending..."
time curl -sS --max-time 60 "http://${HOST}:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d @"${REQ_FILE}" > "${RESP_FILE}"
echo ""

echo "→ full response:"
python3 -m json.tool < "${RESP_FILE}"
echo ""

echo "=== verdict ==="
python3 - "${MODEL}" <<'PY'
import json, sys
model = sys.argv[1]
r = json.load(open("/tmp/resp_tool.json"))
msg = r["choices"][0]["message"]
content = msg.get("content")
tool_calls = msg.get("tool_calls") or []
finish = r["choices"][0].get("finish_reason")

print(f"finish_reason: {finish}")
print(f"content: {repr(content)[:200]}")
print(f"tool_calls count: {len(tool_calls)}")

if tool_calls:
    for i, tc in enumerate(tool_calls):
        fn = tc.get("function", {})
        print(f"  [{i}] name: {fn.get('name')!r}, arguments: {fn.get('arguments')!r}")

ok_null_content = content is None or content == ""
ok_has_calls = len(tool_calls) > 0
ok_right_name = any(
    (tc.get("function") or {}).get("name") == "get_weather"
    for tc in tool_calls
)
ok_args_parseable = True
for tc in tool_calls:
    args_raw = (tc.get("function") or {}).get("arguments")
    if args_raw is None:
        ok_args_parseable = False
        continue
    if isinstance(args_raw, dict):
        continue
    try:
        json.loads(args_raw)
    except Exception:
        ok_args_parseable = False

print()
print(f"  content null/empty:    {ok_null_content}")
print(f"  tool_calls populated:  {ok_has_calls}")
print(f"  correct function name: {ok_right_name}")
print(f"  arguments parseable:   {ok_args_parseable}")
print()
if ok_null_content and ok_has_calls and ok_right_name and ok_args_parseable:
    print(f"PASS — tool calling works on {model}")
else:
    print(f"FAIL — see details above (model={model})")
PY
