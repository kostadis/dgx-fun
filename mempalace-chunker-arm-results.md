# Does re-chunking beat mempalace's own chunker?

**Short answer: mempalace's file-mining chunker is competitive with recursive-600,
but that is not how the module library is stored, so this does not yet answer
whether to re-chunk it.**

Applied per module section, mempalace's `chunk_text` (800 characters, 100
overlap, paragraph-snapped — what `mempalace mine` uses on files) captures most
of the gain recursive-600 showed over fixed 600-token windows: it trails by 3–5
questions on Temple of Elemental Evil and leads by one on Phandelver and Below
(no-breadcrumb arm). Heading breadcrumbs help on ToEE and hurt on Phandelver
at the tight budget.

> **Scope correction (found in review).** The module library does *not* go
> through `chunk_text`. CampaignGenerator's `fivetools_ingest` writes one drawer
> per 5etools entry node (structural), and mempalace's `add_drawer` then hard-splits
> any drawer over `chunk_size` at fixed character offsets; the section path is
> stored as metadata, not embedded. That real arm was not measured here. These
> results describe mempalace's file chunker — the one mneme's campaign-corpus
> mining uses — applied to module text.

Same frozen questions (ToEE 40, Phandelver 48), embedder (`qwen3-embedding:0.6b`,
spark2), dense cosine ranking, candidate pool 20 and budget accounting as the
[ToEE](adaptive-chunking-toee-results.md) and
[Phandelver](phandelver-and-below-chunking-results.md) write-ups. All ten
previously published arm rows (five arms × two modules) reproduced exactly before these numbers were used.

## Complete-evidence questions

| Arm | Breadcrumb | ToEE chunks | ToEE @1,200 | ToEE @2,400 | ToEE top-5 | PaBTSO chunks | PaBTSO @1,200 | PaBTSO @2,400 | PaBTSO top-5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed 600-token windows | yes | 257 | 22 | 34 | 36 | 296 | 28 | 40 | 42 |
| **Mempalace `chunk_text` 800, per section** | **no** | 952 | **34** | **35** | 33 | 1,211 | **43** | **47** | 38 |
| Mempalace 800 + breadcrumb | yes | 952 | 36 | 37 | 36 | 1,211 | 39 | 47 | 39 |
| Recursive 600, no breadcrumb | no | 904 | 35 | 38 | 35 | 1,161 | 45 | 46 | 44 |
| Recursive 600 (published winner) | yes | 904 | 37 | 40 | 37 | 1,161 | 42 | 46 | 42 |
| Adaptive, four metrics | yes | 620 | 37 | 39 | 38 | 759 | 40 | 45 | 41 |

Mean retrieved tokens were within ~40 tokens across the mempalace and recursive
arms at each cap (≈1,119–1,159 at 1,200; ≈2,329–2,349 at 2,400), so packing
differences do not explain the gaps.

## Boundaries × breadcrumb (complete @1,200 / @2,400)

| | No breadcrumb | Breadcrumb |
|---|---|---|
| **ToEE — mempalace 800** | 34 / 35 | 36 / 37 |
| **ToEE — recursive 600** | 35 / 38 | 37 / 40 |
| **PaBTSO — mempalace 800** | 43 / 47 | 39 / 47 |
| **PaBTSO — recursive 600** | 45 / 46 | 42 / 46 |

- **Boundaries:** recursive-600 beats mempalace by 1–3 questions on ToEE in
  both breadcrumb conditions; on Phandelver mempalace leads by 1–2 without
  breadcrumbs and trails by 3 with them at 1,200. No consistent winner.
- **Breadcrumb:** +2 on ToEE for both chunkers; −3 to −4 on Phandelver at 1,200
  for both, and neutral at 2,400. The effect depends on the module, not the
  chunker.
- **Top-5 is weaker for mempalace** in breadcrumb-matched pairs (ToEE 33 vs 35
  and 36 vs 37; PaBTSO 38 vs 44 and 39 vs 42), though chunk sizes differ so
  top-5 is not token-matched. Under matched caps the gap is mixed, and the
  largest single gap is at ToEE 2,400 without breadcrumbs (35 vs 38; 35 vs 40
  against breadcrumbed recursive-600).

## What this means

- **For mneme's campaign corpus** (which `mempalace mine` does chunk with
  `chunk_text`): on module text, that chunker is already in the same regime as
  recursive-600. These are module questions, not campaign questions, so this is
  suggestive only.
- **For the module library:** unanswered. Its real drawers are structural
  entries plus fixed-offset splits of long entries, untested here. That is the
  arm to add next.

Harness differences from real mining: a section under 50 characters gets one
whole-section chunk here (the miner would store nothing), and a dropped tail
fragment under 50 characters would abort the run rather than be modelled
(this did not occur on either corpus). No variance estimate was made for the
1–5 question differences.

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
