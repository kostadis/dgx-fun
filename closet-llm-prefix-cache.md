# Optimize closet_llm: lift the system instructions into a shared prefix

**Goal:** Restructure `closet_llm`'s prompt so vLLM's prefix cache can skip
re-processing the static instructions on every one of the 906 LLM calls
in a full-palace regen. Expected gain: 5–15% wallclock improvement plus
smoother prefill/decode contention.

**Status:** not done. This is the spec.

**Authors:** Kostadis + Claude.

## The problem in one paragraph

vLLM keeps a **prefix cache**: when consecutive requests share a leading
sequence of tokens, the KV cache for those tokens is reused instead of
re-computed. During the full-palace regen we just ran, the live vLLM
log reported:

```
Prefix cache hit rate: 3.5%
```

That's because `closet_llm` interpolates the source-file metadata into
the same string as the static instructions — so the *first* tokens of
each request differ. The shared instructions (the prompt template's
schema, rules, example pair) never enter the cache. vLLM has to
re-prefill them 906 times.

## Why this matters

The instruction block is ~400 tokens. Prefill on the Spark runs at
roughly 800 tok/s, so re-prefilling the static block costs ~0.5 s per
call. At 906 calls that's **~7.5 minutes of wasted GPU**.

The wallclock saving on a 125-minute run is modest (~6%). The bigger
benefit is that those 400 tokens of unavoidable-but-redundant prefill
**also displace decode slots** every step they're being processed. Less
prefill per request means more decode capacity per step, which means
smoother per-stream latency and less risk of the prefill-decode
contention pattern we hit with workers=16 earlier.

If we make this change, the next vLLM log should report:

```
Prefix cache hit rate: 90%+
```

…across the entire 906-call run, modulo the cache eviction window.

## What the fix actually is

### Current code shape

```python
# mempalace/closet_llm.py
PROMPT_TEMPLATE = """You are reading content filed in a memory palace...
[~400 tokens of instructions, rules, output schema, examples]

Source: {source_file}
Wing: {wing} | Room: {room}

CONTENT:
{content}

---

Output a JSON object with EXACTLY these fields:
{{ "index_sentences": [...], "quotes": [...], "summary": "..." }}
[rules block]
"""

prompt = PROMPT_TEMPLATE.format(
    source_file=source_file[:100],
    wing=wing,
    room=room,
    content=content[:MAX_CONTENT_CHARS],
)

body = {
    "model": cfg.model,
    "max_tokens": MAX_OUTPUT_TOKENS,
    "messages": [{"role": "user", "content": prompt}],
}
```

Problem: the *very first* tokens of the request include the prompt
preamble — but the per-source `source_file`, `wing`, `room`,
and `content` get embedded mid-template. Worse, the output-schema
instructions come *after* `{content}`, so even the schema can't be
prefix-cached.

### Target shape

Split into two messages. **System** = 100% identical across all 906
calls. **User** = only the per-source variable parts.

```python
# mempalace/closet_llm.py

SYSTEM_PROMPT = """You are reading content filed in a memory palace.
Generate an index of natural-language sentences that someone searching
for this content would naturally type. Each sentence will be independently
embedded and keyword-indexed.

[full instructions block]

Output a JSON object with EXACTLY these fields:

{
  "index_sentences": [
    "A complete sentence about one searchable aspect.",
    ...
  ],
  "quotes": ["[Speaker] verbatim quote", ...],
  "summary": "2-3 sentences describing what this content is about."
}

RULES:
- index_sentences: 8-15 entries...
[full rules block]
- Output valid JSON only. No code fences. No commentary.
"""

USER_PROMPT_TEMPLATE = """Source: {source_file}
Wing: {wing} | Room: {room}

CONTENT:
{content}"""

# in _call_llm:
user_prompt = USER_PROMPT_TEMPLATE.format(
    source_file=source_file[:100],
    wing=wing,
    room=room,
    content=content[:MAX_CONTENT_CHARS],
)

body = {
    "model": cfg.model,
    "max_tokens": MAX_OUTPUT_TOKENS,
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ],
}
```

vLLM tokenizes the messages into one sequence. Because the system
message is byte-identical across all 906 calls, its tokenized form is
identical, and the prefix cache catches it from call #2 onward.

### Language-instruction tail (lang_instruction)

Currently appended *after* the content:

```python
if lang_instruction and "english" not in lang_instruction.lower():
    prompt += f"\n\nLanguage instruction: {lang_instruction}"
```

This breaks the shared prefix when it's set. Move it into the **system
prompt** (it's static for a given palace anyway):

```python
SYSTEM_PROMPT_BASE = """...the main instructions..."""

def _build_system_prompt() -> str:
    try:
        from mempalace.i18n import t
        lang_instruction = t("aaak.instruction")
    except Exception:
        lang_instruction = ""
    if lang_instruction and "english" not in lang_instruction.lower():
        return SYSTEM_PROMPT_BASE + f"\n\nLanguage instruction: {lang_instruction}"
    return SYSTEM_PROMPT_BASE

# cached at module load so we build the string once, not per call:
SYSTEM_PROMPT = _build_system_prompt()
```

Now the system prompt is **fixed at import time**, identical for every
call within a process run, and fully prefix-cacheable.

## Changes by file

| file | what changes | LOC |
|---|---|---:|
| `mempalace/closet_llm.py` | Split `PROMPT_TEMPLATE` → `SYSTEM_PROMPT` + `USER_PROMPT_TEMPLATE`; update `_call_llm` to send two messages; move `lang_instruction` into the system prompt; one-time build at import. | ~40 |
| `tests/test_closet_llm.py` | Update `_call_llm` request-shape test to expect two messages instead of one. | ~10 |
| **total** | | **~50 lines** |

## Verification

The vLLM log is the empirical signal. Two ways to measure:

**A — quick sample bench (~1 min):**

```bash
MEMPALACE_WORKERS=8 python -m mempalace.closet_llm \
  --palace /home/kroussos/.mempalace/palaces/campaign-dev \
  --endpoint http://192.168.1.147:8001/v1 \
  --model Qwen/Qwen2.5-14B-Instruct-AWQ \
  --sample 16
```

Watch `docker logs -f vllm-chat` on the Spark while it runs. **Pass
criterion:** `Prefix cache hit rate` reported by vLLM rises from ~3.5%
to ≥80% during the steady-state portion of the run.

**B — full-palace bench for wallclock:**

Run the full 906-file regen (the one currently completing in our
session was ~125 min). Compare:

- Pre-fix wallclock: ~125 min (current run).
- Post-fix wallclock: should drop to ~115 min if the prediction holds.

**Diagnostic to check** if the gain is bigger than expected: the
**prefill throughput** numbers from vLLM logs. Currently we see
intermittent prefill spikes of 800-1000 tok/s. After the fix, prefill
spikes should be both **lower in magnitude** (most calls have no static
prefill to do) and **less frequent** (only for the genuinely-new content
tokens).

## Why this won't break anything

1. **Output is identical.** The model sees the same instructions and
   the same content — it just sees them in two separate messages
   instead of one big user message. Qwen2.5-14B-Instruct is trained
   for the system+user format; in fact this is closer to its intended
   prompt shape.
2. **Schema unchanged.** Output JSON shape is identical
   (`index_sentences`, `quotes`, `summary`).
3. **Idempotent.** A re-run after the fix overwrites old closet rows
   via `purge_file_closets` + `upsert_closet_lines` (same path as
   today). No migration needed.
4. **vLLM compatibility.** OpenAI-compat `/v1/chat/completions`
   accepts an arbitrary number of messages; this is the standard
   shape. No vLLM-specific feature is being used.

## Open questions to resolve at implementation

1. **Should `Source: {source_file}` be the very first tokens of the
   user message?** Probably yes — if the file is mined twice (renamed,
   moved) the second call has a different `source_file` and the user
   message diverges immediately. But all calls share the `"Source: "`
   prefix (2 tokens of cache hit). Probably worth ordering this way
   for clarity even if the cache gain is tiny.

2. **vLLM's prefix-cache eviction policy.** The cache is bounded; with
   16 active sequences each holding multi-thousand-token KV caches,
   prefix entries can get evicted. Need to verify the 400-token system
   prefix actually stays cached across 906 requests. The vLLM log's
   hit-rate metric is the authoritative signal — if it lands at e.g.
   50% instead of 95%, eviction is happening and we'd want to bump
   `--prefix-caching-hash-algo` or increase KV cache budget.

3. **Does Qwen2.5-14B-Instruct have any prompt-format requirements
   for system messages?** AWQ-quantised instruct variant is the
   chat-template flavor; system messages are first-class. Verify with
   a single curl smoke-test before doing the full regen.

## How to pick this up

Open a new Claude session at `~/src/mempalace/` and start with:

> "We have a spec at `~/src/dgx/closet-llm-prefix-cache.md`. Implement
> the prompt-split change in `mempalace/closet_llm.py` per the spec.
> Run the sample bench against vllm-chat on the Spark and verify
> prefix-cache hit rate rises from ~3.5% to ≥80%."

Time estimate: 30 min for code + tests, 5 min for sample bench, 2 hours
if you also want the full-palace re-run for wallclock comparison.

## Phase 2 follow-ups (if you want to push further)

Once the simple prefix-cache fix is in, there are more aggressive
optimizations that bend in the same direction:

* **Drop `source_file`/`wing`/`room` from the prompt entirely.** Qwen
  doesn't need them to do the topic-extraction job. They're useful
  context for human review of the closet but don't change the LLM's
  output shape much. Removing them lets the user message be just
  `"CONTENT:\n<content>"`, which means the prefix
  `"CONTENT:\n"` becomes cacheable across all calls. Minor extra gain.
* **Move the JSON output schema into a few-shot example pair embedded
  in the system prompt.** Qwen's instruction-following improves
  measurably with concrete input/output examples vs. abstract schema
  descriptions. Bigger system prompt (~800 tokens) but better output
  quality, and still 100% cacheable.
* **Use vLLM's `--enable-prefix-caching` (default-on in modern
  versions) more aggressively.** Check the vllm-chat container's
  current flags; if prefix caching is somehow disabled, the entire
  exercise above is moot. Per the setup doc you don't pass
  `--no-enable-prefix-caching`, so it should be on by default.
