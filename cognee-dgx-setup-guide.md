# Setting up Cognee against the local DGX Spark

**One line:** How to point Cognee's `remember()`/`recall()` (or the classic `add()`/`cognify()`/`search()`) at a local Spark vLLM endpoint + Ollama embeddings instead of a cloud API, plus an MCP server so you can query from Claude Code directly instead of writing scripts.

Validated end-to-end 2026-08-01 against current cognee `main` (well past v1.2.2) — see [`cognee-local-control-plane-RETEST-2026-08-01.md`](cognee-local-control-plane-RETEST-2026-08-01.md) for the retest that produced this guide, and [`cognee-local-control-plane-2026-07-04.md`](cognee-local-control-plane-2026-07-04.md) for the original hardware-calibration + failure-mode writeup this supersedes for setup purposes.

> **⚠️ 2026-09-10: this recipe is INCOMPLETE as originally written.** Three settings
> are now required that the August version does not mention. Without them a run
> either produces a graph of file paths or never finishes. See
> **[Retest 2026-09-10](#retest-2026-09-10-simplified-technical-english)** at the end.

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
AUTO_RATE_LIMIT="false"                      # cognee's OWN auto-limiter — see 2026-09-10 below

# REQUIRED on a reasoning model (e.g. qwen3.8-flash-next). Without it the model
# spends its whole completion budget thinking and writes zero content.
LLM_ARGS='{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'

# REQUIRED to ingest files outside CWD. Without it a path is silently ingested
# as literal TEXT and remember() still reports status='completed'.
COGNEE_ALLOWED_LOCAL_FILE_ROOTS="/home/kostadis/src:/path/to/project:/tmp"

DATA_ROOT_DIRECTORY="/path/to/your/project/.cognee_data"
SYSTEM_ROOT_DIRECTORY="/path/to/your/project/.cognee_system"
```

**Don't add an `LLM_RATE_LIMIT_*` limiter — and turn off `AUTO_RATE_LIMIT`, which now does it for you (see the 2026-09-10 section).** Validated as a *net negative* on real fan-out workloads (all documents hit their full timeout with zero logged progress, worse than no limiter at all) — see the 2026-07-25 comment on [issue #3870](https://github.com/topoteretes/cognee/issues/3870). The moving-window RPM limiter isn't a substitute for a concurrency cap; it paces admission, not in-flight requests.

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

---

## Retest 2026-09-10 (Simplified Technical English)

This section uses ASD-STE100. Configuration lines are verbatim. Copy them exactly.

### 1. The issue

The language model wrote no answer.

Cognee gave the model 16384 tokens for each answer. The model has a reasoning
function. The model uses this function to think before it writes. The model used
all 16384 tokens to think. Then the model stopped. It wrote zero characters.

Cognee got empty answers. So cognee could not build the graph.

### 2. The proof

We sent the same text two times to an idle server. The text was 4000 characters
of `campaign_state.md`. We used the cognee graph prompt.

| Reasoning | Time | Tokens used to think | Characters written |
|---|---|---|---|
| On | 251 s | 8000 | **0** |
| Off | 172 s | 0 | **25397** |

The model with reasoning on wrote nothing.

### 3. Why the first tests did not show the issue

The first test used a file of 981 bytes. This file is small. The model had enough
tokens to think **and** to write. So the test was satisfactory.

**Small files hide this issue. Large files show it.** Do the smoke test with one
LARGE file, not one small file.

### 4. The correction

Put this line in the `.env` file:

```
LLM_ARGS='{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'
```

This line stops the reasoning function. Cognee sends `llm_args` to
`litellm.acompletion` (`native_adapter.py:433`).

### 5. The results of the issue

The empty answers caused four more failures:

1. Each call was very slow. Eight calls shared one server. Each call needed
   approximately 23 minutes.
2. The calls went past the time limit. Cognee counted 493 timeouts.
3. Cognee has an automatic speed control (`AUTO_RATE_LIMIT`, default true). The
   timeouts started this control. The control made the run more slow. This is the
   same net-negative behaviour as the manual limiter in issue #3870.
4. Cognee sent the failed calls again. This added more load to the server.

**Caution: We tried to correct items 1 to 4 first. This was incorrect.** These
four items are results of the empty answers. They are not the cause. We used
approximately 5 hours of GPU time on them.

### 6. Two more issues

**Issue A — cognee ingests a file path as text.** Set
`COGNEE_ALLOWED_LOCAL_FILE_ROOTS`. If this variable is not set, cognee permits
only the current directory and `/tmp`. A path outside these roots is not an
error. Cognee puts the path STRING into the graph as text. Then `remember()`
reports `status='completed'`. You get a graph of file paths.

The variable REPLACES the defaults. So write all the roots, including `/tmp`.
Separate the roots with `:`.

**Issue B — the litellm time limit is three times the value you set.** Litellm
makes three internal attempts before it raises the error.

| You set | The error came after |
|---|---|
| 600 s | 1801 s |
| 1800 s | 5401 s |

Do not set a small time limit to fail more quickly. It does the opposite.

### 7. The three runs

| Item | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Reasoning | on | on | **off** |
| `data_per_batch` | 20 (default) | 8 | 8 |
| `AUTO_RATE_LIMIT` | true (default) | false | false |
| Timeouts | 493 | 14 | **0** |
| Result | stopped at 5 h, incomplete | stopped at 31 min, no data | **completed, 26.8 min** |

Run 3 result: `RememberResult(status='completed', items=104, elapsed=1610.1s)`.
This is 26.8 minutes. The August 2026 run needed 27 minutes. So the correct
configuration gives the same speed as before. The model is not slow. The
reasoning function was the problem.

### 7a. Answer quality (run 3)

We asked about a person the party has NEVER met. This is a difficult test. A
graph can invent a relation that does not exist.

Cognee did not invent a relation. It answered:

> The party has **never met Hedrack** directly. They first heard his name from
> **Lucius Graeme**, who referenced "Lord Hedrack" as a higher authority when
> selling them magic items.

We checked each statement against the source files. All statements are correct.
The quote is in `npcs/lucius_graeme.md:67`. Cognee also found the correct title
in `npcs/lucius_graeme.md:120`. Cognee kept party knowledge separate from
GM-only knowledge.

Three files in the ingested set name Hedrack. Cognee combined all three.

`data_per_batch` limits DOCUMENTS, not calls. Each document then makes one call
for each of its chunks. So 8 documents can make approximately 24 calls. The
vLLM server has 8 slots (`--max-num-seqs 8`).

### 8. Related cognee behaviour

Cognee does not slow down for vLLM. The file `llm/config.py:39` says vLLM
"absorbs concurrency like a cloud endpoint". This is correct for a large vLLM
installation. This is not correct for one Spark box with 8 slots. The box puts
the extra calls in a queue. Then the client time limit stops the calls.
