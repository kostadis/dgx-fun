# Mempalace chunker arm for the adaptive-chunking experiment

## Intent
The ToEE and Phandelver pilots showed recursive-600 beating fixed 600-token
windows under tight context budgets. That baseline is Cognee-style, not what the
module library actually uses: mneme hands files to `mempalace mine`, which cuts
800-char / 100-overlap paragraph-snapped drawers with no heading breadcrumb.
Question: does re-chunking the module library buy anything over mempalace's
current chunker, and is any gain from boundaries or from breadcrumbs?

## Goals / non-goals
- Goal: comparable numbers against the existing arms on the same frozen questions.
- Non-goals: BM25 hybrid ranking, reader answer checks, campaign corpus, any
  change to a live palace or production index.
- Anti-goal: altering frozen questions, spans, or old receipts.

## Approaches
1. **Extend `run.py` with new arms (chosen)** — reuses frozen questions, span
   validation, cached vectors; results directly comparable.
2. New standalone harness — re-validation cost, incomparable numbers.
3. Mine into a scratch palace and query via mempalace — tests BM25 too, but
   no span-coverage scoring; deferred.

## Design
- `mempalace_800`: `mempalace.miner.chunk_text` defaults (800/100/50), embedded
  without breadcrumb.
- `mempalace_800_crumb`: same chunks with breadcrumb.
- `recursive_600_nocrumb`: recursive-600 without breadcrumb (completes 2x2).
- Chunks are mapped back to source offsets; an unlocatable chunk fails the
  coverage assert.
- Same embedder (qwen3-embedding:0.6b, spark2), budgets 1,200 / 2,400, top-5.
- Outputs: new `*-mempalace-results.json` + short write-up; old receipts untouched.

## Left to the builder
Offset-mapping details, how mempalace is imported (sys.path to ~/src/Mempalace).
