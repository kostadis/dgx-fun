#!/usr/bin/env bash
#
# test-gemma4-toolcall.sh — verify tool calling works on the Gemma 4 26B MoE
# vllm-chat container, with --tool-call-parser gemma4 enabled.
#
# Sends a get_weather function-call probe and reports pass/fail:
#   PASS = content is null + tool_calls[] is populated with name="get_weather"
#          and arguments containing a location
#   FAIL = content contains the function-call as plain text (parser didn't fire)
#
# USAGE
#   scp test-gemma4-toolcall.sh kostadis@192.168.1.147:~/
#   ssh kostadis@192.168.1.147 'bash ~/test-gemma4-toolcall.sh'
#
set -euo pipefail

MODEL="${MODEL:-google/gemma-4-26b-a4b-it}"
PORT="${PORT:-8001}"
REQ_FILE="/tmp/req_tool.json"
RESP_FILE="/tmp/resp_tool.json"

echo "=== test-gemma4-toolcall ==="
echo "  model: ${MODEL}"
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
time curl -sS --max-time 60 "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d @"${REQ_FILE}" > "${RESP_FILE}"
echo ""

echo "→ full response:"
python3 -m json.tool < "${RESP_FILE}"
echo ""

echo "=== verdict ==="
python3 - <<PY
import json
r = json.load(open("${RESP_FILE}"))
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
    print("PASS — tool calling works on Gemma 4 26B MoE")
else:
    print("FAIL — see details above")
PY
