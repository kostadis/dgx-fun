# drive-tagger: two-day summary (July 1–2, 2026)

Compiled from the verbatim session transcripts of **driver-tagger-add-llm-endpoints**
(Jul 1) and **drive-tagger-fixes-2026-07-01** (Jul 1–2).

> Note: MemPalace search was unavailable at write time — the chat palace runs on the
> turbovec backend, and turbovecdb was mid-rebuild (`feat/rust-core`, stale
> `_core.abi3.so`). This summary was built directly from the Claude Code session
> JSONLs, the same verbatim source MemPalace mines.

## The arc in one paragraph

Two days took drive-tagger from "agentic loop with a cloud LLM" to "deterministic
pipeline with a local judge." Day one wired in the DGX Sparks (embeddings + chat).
Day two was a running battle with Qwen's tool-calling reliability — five distinct
failure modes, each fixed empirically — until the pattern became undeniable: every
fix moved a control decision from the model into the harness. The session ended by
finishing that move: a pipeline where Python owns the loop and the LLM makes exactly
one JSON decision per file. Quality and speed both improved.

---

## Session 1 — DGX endpoints (Jul 1, PR #67, merged)

- **DGX embedding provider**: `DT_EMBED_PROVIDER=dgx` routes embeddings through
  Ollama `qwen3-embedding:0.6b` on spark2 (`192.168.1.121:11434`, 1024-dim).
  Provider-aware `embed.__name__` so turbovecdb detects embedder mismatches.
- **Status velocity**: `drive-tagger status` gained rate (files/min) and ETA,
  computed from new `processed_at` stamps.
- **`update_metadata` fix**: metadata updates (mark processed, category counts,
  assignments) no longer re-embed documents; `_reupsert_metadata` removed.
- **Diagnosis that shaped day two**: token/sec degrading over a batch was traced to
  linear context growth in the agent conversation (prefill is O(history)). vLLM
  automatic prefix caching was identified as the mitigation — later enabled on the
  Sparks and ultimately exploited by the pipeline's prompt design.

## Session 2 — fixes, then the architecture pivot (Jul 1 evening → Jul 2)

### Operational plumbing
- `DT_PROVIDER` vs `DT_EMBED_PROVIDER` clarified (chat vs embedding are configured
  independently; vLLM :8001 vs Ollama :11434).
- **Worklist sharding** (`--shard 0/2`, MD5-stable) for running instances on both
  Sparks; later generalized to any N (`0/4`…`3/4`).
- `embed_dim` auto-defaults to 1024 when the embed provider is dgx — a reset no
  longer silently creates a 384-dim collection that mismatches DGX vectors.

### Reliability war with Qwen (the agentic loop)
- **Hallucinated tool calls as prose**: fixed by moving workflow instructions into
  the system turn with a short user trigger.
- **Silent hang in `next_file`**: pypdf can spin forever on malformed PDFs — added a
  90s extraction timeout plus step-level logging so hangs are visible.
- **Skip persistence** (issue #78, PR #82): `skipped_files` table in graph.sqlite
  with reason codes (`no-text`, `extract-timeout`, `extract-error`); `gdrive-error`
  is deliberately *not* persisted (transient); `retry-skipped` CLI command.
- **`stats` loops**: the agent polled `stats` 9× and wrote a prose summary instead of
  working — `stats` removed from the MCP surface (CLI keeps its own path).
- **`link_files` batching**: one call with a list of links instead of 8 sequential
  round trips.
- **Negation blindness (the key discovery)**: "calling link_files first is
  FORBIDDEN" made Qwen call link_files first — 0/5 vs 5/5 across A/B probes on both
  boxes, fully deterministic. Any tool *named* in a prompt tends to get called,
  negated or not. Also applies to imperative tool descriptions. Saved to memory as
  a durable rule: only ever mention the tool you want; A/B probe the endpoint to
  settle prompt disputes.
- **Primed `next_file`** (PR #84, open): the harness makes the first tool call
  itself and injects it as a synthetic turn. This exposed the same literalism in
  positive form — "your first response MUST be next_file" caused a redundant
  re-call — fixed by naming no tools at all.

### Worklist quality (issue #81, PR #83, merged)
- **rpg-lib as authoritative source**: `DT_RPG_LIB_URL` filters the worklist to the
  4,881 canonical books rpg-lib curates — 135k Drive records → ~5.2k relevant files.
- **PDF extraction capped at 20 pages** (intro/TOC/overview is what categorization
  needs) — eliminated the common-case extraction timeout.

### The pivot: deterministic pipeline (PR #85, open)
Recognizing the pattern — the per-file workflow is fixed; only categorize/link is
judgment — the harness became the driver:

- **`drive-tagger pipeline`**: extract → embed → find_similar → full category list →
  **one** LLM call returning JSON `{categories, new_categories, links}` → validate →
  apply. No tools, no tool choice: the entire wrong-tool failure class is gone
  structurally. ~7× fewer LLM round trips per file.
- **Validation the agent couldn't guarantee**: links only to actually-retrieved
  neighbor ids; relations coerced to the allowed vocabulary; JSON parse failures
  retried once then left for the next run.
- **Thread workers** (default 4), each with its own Store/Graph/client — the same
  isolation as the proven multi-shard setup.
- **Killable subprocess extraction**: a stuck pypdf is now *terminated*, not
  abandoned — the thread version leaked a GIL-contending zombie per malformed PDF.
  Child processes also give parses their own GIL.
- **Drains by default**: `--max-files` is an optional cap, not an inherited 50-file
  batch budget — batches were a context-window workaround the pipeline doesn't need,
  since every judge call is stateless.
- **Prefix-cache-shaped prompt**: stable content (category list, member counts
  removed) leads; per-file content trails. With vLLM prefix caching on, the growing
  taxonomy rides the cache and per-call prefill is just the file tail. Confirmed
  much better token/sec.
- Verified: 25/25 files, zero judge failures, idempotent; ~2.5 files/min on fresh
  files, ~10+ files/min through previously-embedded ones.

## State of play (end of Jul 2 morning)

| Item | Status |
|---|---|
| PR #67 DGX embed + status rate | merged |
| PR #82 skip persistence | merged |
| PR #83 rpg-lib filter + PDF cap + prompt fixes | merged |
| PR #84 primed next_file (agentic path) | open |
| PR #85 deterministic pipeline | open |
| Issue #79 OAuth token expiry mid-run | open |
| Issue #80 parallel extraction | largely absorbed by pipeline workers |
| turbovecdb `feat/rust-core` | in progress (separate effort); fresh pipeline launches need `maturin develop` once it lands |

## The durable lesson

Local 80B-class models are **strong judges and weak drivers**. Prompt engineering
reduced the agentic failure rate but never to zero; restructuring so the model only
ever answers — never sequences — eliminated the failure class outright and ran
faster. Salience beats logic in prompts for these models: never name a tool you
don't want called, keep tool descriptions declarative, and settle prompt disputes
with A/B probes against the endpoint rather than argument.
