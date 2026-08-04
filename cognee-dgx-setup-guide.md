# Setting up Cognee against the local DGX Spark

**One line:** How to point Cognee's `remember()`/`recall()` (or the classic `add()`/`cognify()`/`search()`) at a local Spark vLLM endpoint + Ollama embeddings instead of a cloud API, plus an MCP server so you can query from Claude Code directly instead of writing scripts.

Validated end-to-end 2026-08-01 against current cognee `main` (well past v1.2.2) — see [`cognee-local-control-plane-RETEST-2026-08-01.md`](cognee-local-control-plane-RETEST-2026-08-01.md) for the retest that produced this guide, and [`cognee-local-control-plane-2026-07-04.md`](cognee-local-control-plane-2026-07-04.md) for the original hardware-calibration + failure-mode writeup this supersedes for setup purposes.

---

## Prerequisites

- A running OpenAI-compatible chat endpoint on a Spark box (vLLM, e.g. `http://192.168.1.147:8001/v1`). Check what's actually being served before wiring anything up:
  ```bash
  curl -sS http://192.168.1.147:8001/v1/models
  ```
- An embedding model served by Ollama on a Spark box (e.g. `qwen3-embedding:0.6b`):
  ```bash
  curl -sS http://192.168.1.121:11434/api/tags
  ```
- See `current-setup.md` in this repo for what's actually live right now — it changes.

## Install cognee

**Use an editable install from a current checkout of the `cognee` repo, not an old pinned PyPI version.** The 2026-08-01 retest found real fixes landed between v1.2.2 (2026-06-26) and current `main` that matter for local-backend robustness (a lock-race fix, general hardening, a new backend-aware overload/pacing policy). Pin an old version and you'll rediscover already-fixed bugs.

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e /home/kostadis/src/cognee
git -C /home/kostadis/src/cognee pull   # before installing, so the editable install is current
```

`ladybug` (the default graph backend, wraps Kuzu) is a core dependency — no extras needed for local-only use.

## `.env`

```bash
LLM_PROVIDER="custom"
LLM_MODEL="openai/<served-model-id>"        # "openai/" prefix required — see gotcha below
LLM_ENDPOINT="http://192.168.1.147:8001/v1"
LLM_API_KEY="sk-noauth-vllm"                 # vLLM usually doesn't check this; any string works

EMBEDDING_PROVIDER="openai_compatible"       # NOT "ollama" — see gotcha below
EMBEDDING_MODEL="qwen3-embedding:0.6b"
EMBEDDING_ENDPOINT="http://192.168.1.121:11434/v1"
EMBEDDING_API_KEY="ollama"
EMBEDDING_DIMENSIONS="1024"

ENABLE_BACKEND_ACCESS_CONTROL="false"        # single-user local, no auth needed
RAISE_INCREMENTAL_LOADING_ERRORS="false"     # validated: lets a batch finish past one bad doc

DATA_ROOT_DIRECTORY="/path/to/your/project/.cognee_data"
SYSTEM_ROOT_DIRECTORY="/path/to/your/project/.cognee_system"
```

**Don't add an `LLM_RATE_LIMIT_*` limiter.** Validated as a *net negative* on real fan-out workloads (all documents hit their full timeout with zero logged progress, worse than no limiter at all) — see the 2026-07-25 comment on [issue #3870](https://github.com/topoteretes/cognee/issues/3870). The moving-window RPM limiter isn't a substitute for a concurrency cap; it paces admission, not in-flight requests.

### Gotchas

1. **`import cognee` runs `dotenv.load_dotenv(override=True)`**, which clobbers any `os.environ[...]` you set *before* the import back to whatever `.env` says. Setting env vars in Python before `import cognee` does not work reliably — the `.env` file (or env vars set on the *process* before Python even starts, e.g. via an MCP server's `-e` flags) is the only reliable knob.
2. **`.env` loads relative to CWD.** `chdir()` into the directory holding your `.env` *before* `import cognee`, or you'll silently fall back to OpenAI defaults and get a confusing auth error. If launching from an MCP config or a wrapper script where CWD isn't guaranteed, prefer passing every value as a real environment variable on the process (see the MCP section below) rather than relying on `.env` discovery.
3. **`LLM_MODEL` needs the `openai/` prefix** for the `custom` provider — LiteLLM needs it to route to `LLM_ENDPOINT` (`api_base`), and strips it before sending the actual served model id.
4. **Embeddings: use `EMBEDDING_PROVIDER="openai_compatible"`, not `"ollama"`.** The `openai_compatible` tokenizer loader falls back to TikToken when the HF tokenizer id doesn't resolve, avoiding a `transformers` install / HF download. `ollama` hard-requires both.

## Smoke test before a real batch

Always validate end-to-end on one small doc before committing to a big ingest — cheap, and catches endpoint/config mistakes in ~2-3 minutes instead of losing a long run to them.

```python
import asyncio, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # before import cognee, per gotcha #2
import cognee

async def main():
    result = await cognee.remember("/path/to/one/small/doc.md", dataset_name="smoke", self_improvement=False)
    print(result)
    print(await cognee.recall("some question about that doc", datasets=["smoke"]))

asyncio.run(main())
```

## Real ingestion

```python
import litellm
litellm.request_timeout = 1800  # generous; set AFTER `import cognee`, not before

result = await cognee.remember(
    list_of_file_paths,           # remember() takes str, list[str] (file paths), or raw text
    dataset_name="your_dataset",
    self_improvement=False,       # skip the self-improvement loop for a bulk ingest
)
```

Validated on a real 106-file corpus (3 dense docs + 103 small ones, ~50k words total): completed in ~27 minutes, zero errors, zero retries, on current `main` against a local vLLM + Ollama backend.

## MCP server — query without writing scripts

`cognee-mcp/` in the cognee repo ships an official MCP server exposing `remember`, `recall`, and `forget` as tools. Register it pointed at the same config, passing every `.env` value as an explicit `-e` flag (sidesteps gotcha #2 above entirely — no CWD/`.env`-discovery dependency):

```bash
claude mcp add cognee-<name> \
  -e LLM_PROVIDER=custom \
  -e LLM_MODEL="openai/<served-model-id>" \
  -e LLM_ENDPOINT="http://192.168.1.147:8001/v1" \
  -e LLM_API_KEY="sk-noauth-vllm" \
  -e EMBEDDING_PROVIDER=openai_compatible \
  -e EMBEDDING_MODEL="qwen3-embedding:0.6b" \
  -e EMBEDDING_ENDPOINT="http://192.168.1.121:11434/v1" \
  -e EMBEDDING_API_KEY=ollama \
  -e EMBEDDING_DIMENSIONS=1024 \
  -e ENABLE_BACKEND_ACCESS_CONTROL=false \
  -e RAISE_INCREMENTAL_LOADING_ERRORS=false \
  -e DATA_ROOT_DIRECTORY="/path/to/.cognee_data" \
  -e SYSTEM_ROOT_DIRECTORY="/path/to/.cognee_system" \
  -- /path/to/venv/bin/python /home/kostadis/src/cognee/cognee-mcp/src/server.py
```

The server's own deps (`mcp>=1.12,<2.0.0`, `httpx`, `starlette`, `uvicorn` — `sse-starlette`/`httpx-sse` come along transitively) need installing into whatever venv you point at; a venv with cognee itself editable-installed already has everything else. Defaults to stdio transport, `-s local` scope (private to you, this project, this machine — not committed anywhere).

Once connected (`claude mcp list` shows `✔ Connected`), `recall`/`remember`/`forget` are available as normal tool calls instead of one-off Python scripts.
