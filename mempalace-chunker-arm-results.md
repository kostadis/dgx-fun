# Does re-chunking beat mempalace's own chunker?

**Short answer: not clearly.** On the two module pilots, mempalace's existing
drawer chunker (800 characters, 100 overlap, paragraph-snapped) already captures
almost all of the gain recursive-600 showed over fixed 600-token windows. It
trails recursive-600 on Temple of Elemental Evil by 3–5 questions and *beats*
it on Phandelver and Below by one. Heading breadcrumbs, which mempalace drawers
lack, help on ToEE and hurt on Phandelver at the tight budget. **This pilot does
not justify re-chunking the module library.**

Same frozen questions (ToEE 40, Phandelver 48), embedder (`qwen3-embedding:0.6b`,
spark2), dense cosine ranking, candidate pool 20 and budget accounting as the
[ToEE](adaptive-chunking-toee-results.md) and
[Phandelver](phandelver-and-below-chunking-results.md) write-ups. All eight
previously published arm rows reproduced exactly before these numbers were used.

## Complete-evidence questions

| Arm | Breadcrumb | ToEE chunks | ToEE @1,200 | ToEE @2,400 | ToEE top-5 | PaBTSO chunks | PaBTSO @1,200 | PaBTSO @2,400 | PaBTSO top-5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed 600-token windows | yes | 257 | 22 | 34 | 36 | 296 | 28 | 40 | 42 |
| **Mempalace 800 (as stored today)** | **no** | 952 | **34** | **35** | 33 | 1,211 | **43** | **47** | 38 |
| Mempalace 800 + breadcrumb | yes | 952 | 36 | 37 | 36 | 1,211 | 39 | 47 | 39 |
| Recursive 600, no breadcrumb | no | 904 | 35 | 38 | 35 | 1,161 | 45 | 46 | 44 |
| Recursive 600 (published winner) | yes | 904 | 37 | 40 | 37 | 1,161 | 42 | 46 | 42 |
| Adaptive, four metrics | yes | 620 | 37 | 39 | 38 | 759 | 40 | 45 | 41 |

Mean retrieved tokens were within ~40 tokens across the mempalace and recursive
arms at each cap (≈1,140–1,160 at 1,200; ≈2,330–2,350 at 2,400), so packing
differences do not explain the gaps.

## Boundaries × breadcrumb (complete @1,200 / @2,400)

| | No breadcrumb | Breadcrumb |
|---|---|---|
| **ToEE — mempalace 800** | 34 / 35 | 36 / 37 |
| **ToEE — recursive 600** | 35 / 38 | 37 / 40 |
| **PaBTSO — mempalace 800** | 43 / 47 | 39 / 47 |
| **PaBTSO — recursive 600** | 45 / 46 | 42 / 46 |

- **Boundaries:** recursive-600 beats mempalace by 1–3 questions on ToEE in
  both breadcrumb conditions; on Phandelver the two are within 2 questions in
  either direction. No consistent winner.
- **Breadcrumb:** +2 on ToEE for both chunkers; −3 to −4 on Phandelver at 1,200
  for both, and neutral at 2,400. The effect depends on the module, not the
  chunker.
- **Top-5 is weaker for mempalace** (33 vs 37; 38 vs 42) but chunks differ in
  size, so top-5 is not a matched comparison. Under matched token caps the gap
  mostly disappears.

## What this means for the module library

The big win in the earlier pilots was moving *off fixed 600-token windows*.
Mempalace never used those: its small, paragraph-snapped drawers already sit
in the same regime as recursive-600. Differences of 1–5 questions out of 40–48
on one representation and one embedder are within what question selection can
swing. Re-chunking the module palace would cost a full re-mine for no
demonstrated gain.

Not tested: mempalace's BM25 hybrid re-rank (this was dense-only on both
sides), graph-backed retrieval, answer quality, campaign files. If a follow-up
is worth doing, the cheapest is adding BM25 to both arms, since that is how
mempalace actually ranks.

## Isolation

No palace was opened. The harness does not import mempalace: importing
`mempalace.miner` resolves a storage backend and reads `~/.mempalace/config.json`,
so `chunk_text` and its three constants are lifted from `miner.py` source with
`ast` and run in an isolated namespace (`MEMPALACE_PALACE_PATH` is also pinned
to a scratch path under `raw/`). A test runs the chunker with an empty fake
`HOME`, asserts it stays empty and that no `mempalace` module is loaded.
`miner.py`'s SHA-256 is in each run manifest.

Runs took 67.6 s (ToEE) and 80.5 s (PaBTSO); old-arm vectors were reused
from the earlier caches. 17 harness tests pass.

Artifacts: [ToEE aggregate](experiments/adaptive-chunking/mempalace-results.json),
[PaBTSO aggregate](experiments/adaptive-chunking/pabtso-mempalace-results.json).
