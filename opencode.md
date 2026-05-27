# opencode against the DGX Spark

Notes for driving [opencode](https://opencode.ai) against the Spark's
`vllm-chat` container on `http://192.168.1.147:8001/v1`. Companion to
`desktop-chat-clients.md` (which covers GUI chat clients) — opencode
is the terminal coding-agent client, not a chat UI.

For what is *currently* served on port 8001 (model id, context length,
tool-call parser), `current-setup.md` is authoritative. This doc covers
the opencode-side wiring and the failure modes seen in practice.

---

## TL;DR

```bash
# On the Spark: bring up the model you want on port 8001.
ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b.sh'

# On the laptop: edit ~/.config/opencode/opencode.json to set
# "model" to the matching DGX entry, then:
opencode
```

No env vars, no wrapper script needed for the standard path. The
config file is the source of truth.

---

## Config: `~/.config/opencode/opencode.json`

opencode reads provider config from `~/.config/opencode/opencode.json`.
The DGX provider has five registered chat models, keyed by an
opencode-side short name. The literal file shipped on the laptop:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "dgx": {
      "api": "openai",
      "name": "DGX Spark (vLLM)",
      "options": {
        "baseURL": "http://192.168.1.147:8001/v1",
        "apiKey": "ignored"
      },
      "models": {
        "qwen3-next-80b": {
          "id": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
          "name": "Qwen3-Next 80B A3B Instruct FP8 @ 128K (hybrid attention, tools)",
          "limit": { "context": 131072, "output": 8192 },
          "tool_call": true,
          "temperature": true
        },
        "nemotron3-nano-30b": {
          "id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
          "name": "Nemotron 3 Nano 30B A3B BF16 (256K, reasoning+tools)",
          "limit": { "context": 262144, "output": 8192 },
          "tool_call": true,
          "reasoning": true,
          "temperature": true
        },
        "gemma-4-26b-moe-longctx": {
          "id": "google/gemma-4-26b-a4b-it",
          "name": "Gemma 4 26B MoE (A4B) BF16 @ 128K",
          "limit": { "context": 131072, "output": 8192 },
          "tool_call": true,
          "temperature": true
        },
        "gemma-4-26b-moe": {
          "id": "google/gemma-4-26b-a4b-it",
          "name": "Gemma 4 26B MoE (A4B) BF16 @ 32K (high concurrency)",
          "limit": { "context": 32768, "output": 8192 },
          "tool_call": true,
          "temperature": true
        },
        "llama-3.3-70b": {
          "id": "casperhansen/llama-3.3-70b-instruct-awq",
          "name": "Llama 3.3 70B Instruct AWQ",
          "limit": { "context": 65536, "output": 8192 },
          "tool_call": true,
          "temperature": true
        }
      }
    }
  },
  "model": "dgx/qwen3-next-80b"
}
```

### Field meanings

- **`id`**: the literal HuggingFace model id that vLLM serves at
  `GET /v1/models`. Must match exactly, or vLLM returns HTTP 400 on
  every call.
- **`limit.context` / `limit.output`**: opencode's *intent* for that
  model — it does **not** reconfigure the server. The vLLM container's
  `--max-model-len` is the real ceiling. The opencode value should be
  ≤ the server's max_model_len.
- **`tool_call: true`**: tells opencode the model can be driven through
  tool calls. The server must be launched with
  `--enable-auto-tool-choice --tool-call-parser <family>` or it will
  return assistant content with embedded JSON instead of structured
  `tool_calls` — opencode's agent loop will silently fail.
- **`reasoning: true`** (Nemotron only): tells opencode the model emits
  a separate reasoning trace. Useful only if the server's reasoning
  parser surfaces it via `reasoning_content` (OpenAI convention). The
  custom `nano_v3` parser emits it as `reasoning` instead — opencode
  doesn't surface that, so tokens are consumed silently. See the
  Nemotron gotcha below.

### Switching models

Flip the top-level `"model"` to one of the five `dgx/<short>` ids,
then run the matching spin-up script on the Spark:

| opencode `model` | Spark spin-up |
|---|---|
| `dgx/qwen3-next-80b` | `bash ~/spin-up-vllm-qwen3-next-80b.sh` |
| `dgx/nemotron3-nano-30b` | `bash ~/spin-up-vllm-nemotron3-nano-30b.sh` (on Spark only) |
| `dgx/gemma-4-26b-moe-longctx` | `bash ~/spin-up-vllm-gemma4-26b-moe-longctx.sh` |
| `dgx/gemma-4-26b-moe` | `bash ~/spin-up-vllm-gemma4-26b-moe.sh` |
| `dgx/llama-3.3-70b` | `bash ~/spin-up-vllm-llama70b.sh` |

The `id` in the opencode entry must match the vLLM `id` for the
container actually running on port 8001. opencode does not auto-detect
the served model — if the two disagree, every call 400s.

---

## Verifying before you launch

```bash
# Is the server up at all?
curl -sS http://192.168.1.147:8001/v1/models

# What model id is it actually serving?
curl -sS http://192.168.1.147:8001/v1/models | python3 -c \
  'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])'

# Smoke test the OpenAI-compatible chat endpoint:
curl -sS http://192.168.1.147:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<served_id>","messages":[{"role":"user","content":"Say OK"}],"max_tokens":8}'
```

If the `id` doesn't match the opencode entry's `id`, fix one side
before launching opencode — debugging the agent loop with a 400 in
flight is painful.

For a full end-to-end tool-call probe (the most common failure mode):

```bash
MODEL=<served_id> ./test-toolcall.sh
```

PASS = `tool_calls[get_weather]` with parseable JSON arguments and
null `content`. FAIL = content carries embedded JSON, or the response
is plain prose — server lacks the tool-call flags, opencode's agent
loop will not work.

---

## Wrapper script: `opencode-spark-longctx.sh`

`opencode-spark-longctx.sh` is an older alternate entry point that
predates the multi-model `opencode.json`. It:

1. prechecks that vllm-chat is serving the expected model id at
   ≥ `MIN_CTX` context (via `lib-precheck.sh`);
2. exports legacy `OPENAI_API_BASE` / `OPENAI_MODEL` env vars; and
3. `exec`s opencode.

Use it when you want the precheck guard rail (no silent fall-through
to a stale model on the server). Override `MODEL_ID` and `MIN_CTX` to
match whatever you're targeting:

```bash
# Qwen3-Next 80B @ 128K (current default)
MODEL_ID=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 MIN_CTX=131072 \
  ./opencode-spark-longctx.sh

# Gemma 4 26B MoE longctx @ 128K
MODEL_ID=google/gemma-4-26b-a4b-it MIN_CTX=131072 \
  ./opencode-spark-longctx.sh

# Skip the precheck entirely
MIN_CTX=0 ./opencode-spark-longctx.sh
```

The bare `opencode` invocation against `opencode.json` is the standard
path; the wrapper is only useful when you want the model-id assertion
before the agent loop starts.

---

## Known failure modes (from Phase B testing)

### 1. Tool-call flags missing on the server

vLLM will happily serve a tool-call-capable model **without**
`--enable-auto-tool-choice --tool-call-parser <family>`. The chat
endpoint returns content with the tool call JSON embedded as text, and
opencode's agent loop does nothing. Every current `spin-up-vllm-*.sh`
script sets the flags correctly, but if you hand-roll a container,
verify with `test-toolcall.sh` before invoking opencode.

Parser families seen in this repo:
- **Qwen3-Next**: `--tool-call-parser hermes` (verified end-to-end).
  `qwen3_coder` is a stricter alternate.
- **Gemma 4 26B MoE**: see `gemma4-26b-moe-runbook.md` and
  `gemma4-26b-moe-observations.md` for the parser-identification step.
- **Nemotron 3 Nano**: `--tool-call-parser qwen3_coder` (per the
  NVIDIA vLLM recipe).
- **Llama 3.3 70B AWQ**: `--tool-call-parser llama3_json`.

### 2. Model id mismatch (opencode.json vs. served)

Most common when swapping vllm-chat without updating `opencode.json`
(or vice versa). Symptom: every opencode call 400s with `model not
found`. Fix: align the two — flip `"model"` in `opencode.json` to the
entry whose `id` matches what `curl /v1/models` returns.

### 3. Nemotron reasoning-trace leakage

Nemotron 3 Nano 30B was tried on `vllm-chat` 2026-05-18 → 2026-05-19
(see `nemotron3-nano-30b-observations.md`). The Phase B verdict:

- The custom `nano_v3_reasoning_parser.py` emits the reasoning trace
  in a field called `reasoning`, **not** the OpenAI-convention
  `reasoning_content`.
- opencode's openai-compatible provider does not surface a custom
  `reasoning` field — those tokens consume budget silently.
- llm_wiki (a different client) has *no* reasoning-trace stripping at
  all, so the chat box filled with raw `<think>` blocks. This is what
  ultimately rejected the experiment.

opencode itself tolerates the leak better than llm_wiki (the agent
loop still sees structured `tool_calls`), but the `max_tokens` budget
on every turn needs to be padded substantially for the silent
reasoning consumption. If you re-enable the Nemotron entry as default,
expect to spend ~1.5–2× the output tokens you'd budget for an
instruction-tuned non-reasoning model.

### 4. `limit.context` lying to opencode

If `opencode.json` lists a larger `context` than the server's
`--max-model-len`, opencode happily packs a long prompt and the server
rejects the request at the boundary. Always set `limit.context` ≤ the
container's `--max-model-len`. The current entries match the active
spin-up scripts.

---

## Slot history (which opencode entry was the default, when)

Only the dated transitions below are sourced from `current-setup.md` /
git log; earlier defaults (Qwen 2.5 14B AWQ, Llama 3.3 70B with
spec-decode, the original Gemma 4 26B MoE @ 32K slot) existed but
their exact handover dates aren't recorded in this repo.

| date | default `model` | notes |
|---|---|---|
| up to 2026-05-17 | (various, undated) | Qwen 2.5 14B AWQ → Llama 3.3 70B AWQ+spec-decode → Gemma 4 26B MoE @ 32K, in some order |
| ~2026-05-17 → 05-18 | `dgx/gemma-4-26b-moe-longctx` | longctx variant in place before Nemotron swap |
| 2026-05-18 → 05-19 | `dgx/nemotron3-nano-30b` | rejected after Phase B (see above) |
| 2026-05-19 → 05-21 | `dgx/gemma-4-26b-moe-longctx` | revert pending Qwen3-Next |
| 2026-05-21 → current | `dgx/qwen3-next-80b` | hybrid attention, 128K, tools verified |

The full per-model experimental records live in the matching
`*-observations.md` files. This file is just the opencode-side wiring.

---

## See also

- `current-setup.md` §6 — the authoritative copy of `opencode.json`
  (this doc reproduces it; if they disagree, `current-setup.md` wins
  because that's the file the rebuild-from-scratch path uses).
- `desktop-chat-clients.md` — GUI chat clients (Jan, Cherry Studio,
  Open WebUI) for the same `vllm-chat` endpoint.
- `opencode-spark-longctx.sh` — the precheck-wrapper entry point.
- `lib-precheck.sh` — the precheck helper used by the wrapper.
- `test-toolcall.sh` — end-to-end tool-call probe, parametrised on
  `MODEL`.
- `gemma4-26b-moe-observations.md` — Phase B2 opencode + tool-calling
  validation against Gemma 4.
- `nemotron3-nano-30b-observations.md` — Phase B opencode + tool-call
  + reasoning validation against Nemotron (rejected).
