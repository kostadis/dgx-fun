# Mempalace Chunker Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three arms (`mempalace_800`, `mempalace_800_crumb`, `recursive_600_nocrumb`) to the adaptive-chunking harness and score them on the frozen ToEE and Phandelver questions.

**Architecture:** New arms are added to `METHODS` but NOT to `CANDIDATES`, so the adaptive selector is unchanged. Each chunk records a `crumb: bool`; `retrieval_text` gets the breadcrumb prefix only when true. Mempalace chunks come from `mempalace.miner.chunk_text` and are mapped back to source offsets.

**Tech Stack:** Python 3, unittest, existing `run.py`, `~/src/Mempalace` (branch `kostadis-dev`) imported via `sys.path`, Ollama `qwen3-embedding:0.6b` on spark2.

**Spec:** `docs/superpowers/specs/2026-11-20-mempalace-chunker-arm-design.md`

## Global Constraints

- Mempalace defaults: `chunk_size=800`, `chunk_overlap=100`, `min_chunk_size=50` (chars). Do not override.
- Budgets `1200 2400`, candidate pool 20, same embedder/endpoint defaults.
- Frozen `raw/cases.json`, `raw/pabtso/cases.json`, and existing `pilot-20261120` dirs and `*results.json` receipts are read-only.
- Outputs stay under `experiments/adaptive-chunking/raw/` (enforced by run.py).
- No LLM calls. No live palace touched.

## Review Focus

- A mempalace chunk whose stripped text occurs twice in a section (repeated boxed text / table rows) — must map to the occurrence at or after the previous chunk's start, not the first in the document.
- A section shorter than `min_chunk_size` → `chunk_text` returns `[]`; the section must still be covered (fall back to one whole-section chunk) or coverage asserts fail.
- `chunk_text` strips whitespace, so `text[start:end]` must equal the chunk exactly after mapping — no leading/trailing whitespace drift.
- Overlapping chunks must not break `span_coverage` (union of intervals already handles it — verify by test).
- The `crumb=False` arms must not leak the breadcrumb via any other field used in embedding.

---

### Task 1: Mempalace piece builder with offset mapping

**Files:**
- Modify: `experiments/adaptive-chunking/run.py` (add helper near `token_windows`, line ~48)
- Test: `tests/test_adaptive_chunking_trial.py`

**Interfaces:**
- Produces: `mempalace_pieces(text: str) -> list[tuple[int, int]]` — ordered `(start, end)` source offsets, one per mempalace chunk; `text[start:end]` equals the chunk's `content`. Empty `chunk_text` result → `[(0, len(text))]`.
- Produces: module constant `MEMPALACE_ROOT = Path.home() / "src" / "Mempalace"`, inserted on `sys.path` lazily inside the function.

- [ ] **Step 1: Write failing tests** (unittest, same loading pattern as existing tests in the file)
  - `test_mempalace_pieces_roundtrip`: for a ~3,000-char multi-paragraph text, every `(s,e)` has `text[s:e] == chunk_text(text, "x")[i]["content"]`, starts are non-decreasing, and union of intervals covers every non-whitespace char.
  - `test_mempalace_pieces_repeated_paragraph`: text with the same paragraph repeated 3× far apart; mapped starts are strictly increasing.
  - `test_mempalace_pieces_short_section`: `mempalace_pieces("tiny")` == `[(0, 4)]`.
- [ ] **Step 2: Run** `python3 -m pytest tests/test_adaptive_chunking_trial.py -k mempalace -v` — Expected: FAIL (`mempalace_pieces` undefined).
- [ ] **Step 3: Implement** — map each chunk with `text.find(content, cursor)` where `cursor` is the previous match's start; raise `AssertionError` with section id context if not found.
- [ ] **Step 4: Run** same command — Expected: 3 passed. Then full file — Expected: all prior 63 tests still pass.
- [ ] **Step 5: Commit** `feat(adaptive-chunking): mempalace chunk offsets`

### Task 2: Wire three arms into prepare_chunks

**Files:**
- Modify: `experiments/adaptive-chunking/run.py:26-28` (constants), `:154-180` (`prepare_chunks`), manifest `inputs` in `main`
- Test: `tests/test_adaptive_chunking_trial.py`

**Interfaces:**
- Consumes: `mempalace_pieces` (Task 1).
- Produces: `EXTRA_METHODS = ("mempalace_800", "mempalace_800_crumb", "recursive_600_nocrumb")`; `METHODS = ("fixed_600", *CANDIDATES, *EXTRA_METHODS)`; `NO_CRUMB = {"mempalace_800", "recursive_600_nocrumb"}`. Each chunk dict gains `"crumb": bool`. Manifest `inputs` gains `"extra_methods": EXTRA_METHODS` and `"mempalace_commit": <git rev-parse HEAD of MEMPALACE_ROOT>`.

- [ ] **Step 1: Write failing tests** using a tiny two-section synthetic corpus run through `prepare_chunks` (upstream loaded as existing tests do):
  - `test_extra_arms_present`: all three keys in result.
  - `test_nocrumb_retrieval_text`: for `NO_CRUMB` arms, `c["retrieval_text"] == c["text"]` and `c["crumb"] is False`; for `mempalace_800_crumb`, `retrieval_text` starts with the module title prefix.
  - `test_recursive_nocrumb_same_bounds`: `recursive_600_nocrumb` `(start,end)` list equals `recursive_600`'s.
  - `test_candidates_unchanged`: `CANDIDATES == ("recursive_600", "recursive_1100", "module_headings_600")`.
- [ ] **Step 2: Run** `-k "extra_arms or nocrumb or candidates_unchanged"` — Expected: FAIL.
- [ ] **Step 3: Implement.** `recursive_600_nocrumb` reuses recursive_600 settings. Mempalace arms skip `repair_gaps_between_chunks`/`check_chunk_gaps` (overlap is intentional) and instead assert full non-whitespace coverage. `intrinsic()` must iterate only `CANDIDATES` — check and restrict if it iterates `METHODS`.
- [ ] **Step 4: Run** full test file — Expected: all pass.
- [ ] **Step 5: Commit** `feat(adaptive-chunking): mempalace and no-breadcrumb arms`

### Task 3: Run both modules and write up

**Files:**
- Create: `experiments/adaptive-chunking/mempalace-results.json`, `experiments/adaptive-chunking/pabtso-mempalace-results.json` (via `analyze.py`)
- Create: `mempalace-chunker-arm-results.md` (repo root, beside the two existing write-ups)
- Modify: `experiments/adaptive-chunking/README.md` (append the two run commands)

- [ ] **Step 1: Seed caches** so old arms cost no embeddings: `cp raw/pilot-20261120/embeddings.sqlite3 raw/mempalace-20261120/` and same for `raw/pabtso/pilot-20261120` → `raw/pabtso/mempalace-20261120/`.
- [ ] **Step 2: Run ToEE** with the README's `run.py` command, `--output experiments/adaptive-chunking/raw/mempalace-20261120`. Expected: `{"stage": "complete"...}`; old-arm counts in results.json identical to `results.json` receipt (257 / 904 / 886 / 621 chunks; fixed 22/40 & 34/40 at 1,200/2,400).
- [ ] **Step 3: Run Phandelver** likewise into `raw/pabtso/mempalace-20261120`. Expected: old arms reproduce 296/1,161 chunks; fixed 28/48 & 40/48, recursive-600 42/48 & 46/48 from `pabtso-results.json`. If any old-arm number differs, stop and report — do not write up.
- [ ] **Step 4: `analyze.py`** each run into the two new results JSON files.
- [ ] **Step 5: Write-up** — one table per module (all arms × 1,200 / 2,400 / top-5 / chunk count / mean actual tokens), a 2×2 table (boundaries × breadcrumb) for recursive-600 vs mempalace-800, and a one-paragraph verdict: does re-chunking beat mempalace's current chunker, and how much is breadcrumb.
- [ ] **Step 6: Commit** `docs: mempalace chunker arm results`
