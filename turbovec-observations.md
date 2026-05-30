# turbovec on the Spark — CPU vector index evaluation

Append-only experiment log. Question: is [turbovec](https://github.com/RyanCodrai/turbovec)
(a CPU vector index using Google's TurboQuant 4-bit quantization, NEON kernels
on ARM) viable as a storage backend for `~/src/mempalace`, whose design bar is
"100% recall, verbatim always"?

turbovec runs on the Spark's otherwise-idle Grace ARM cores while the Blackwell
GPU serves the LLMs. All embeddings are `nomic-embed-text-v1.5` (768-d) from
`vllm-embed` on `:8000`; natural-language queries (the final test) are drafted
by `Qwen3-Next-80B` on `vllm-chat` `:8001`. This was a **client-side probe** —
nothing about the Spark's live serving state changed, so `current-setup.md` is
intentionally untouched.

## TL;DR

**turbovec 4-bit looks 3–4 points worse than ChromaDB on a pure-vector recall
benchmark, but is quality-indistinguishable from ChromaDB end-to-end in a
realistic hybrid (BM25+vector) retrieval system — while ~12–30× faster, ~8×
smaller, and free on the idle ARM CPU.** The index-level recall metric
overstates the cost of quantization for a hybrid retrieval system. Only the
end-to-end test in the *balanced query regime* told the truth.

This is the same shape of lesson as "decode-only benchmarks hide the production
gap": the cheap, obvious metric measures the wrong thing.

## Scripts (the apparatus)

- `turbovec-smoke.py` — install + correctness gate. Proves turbovec imports on
  aarch64 off a wheel (no Rust source build) and returns the obvious neighbor
  from real nomic vectors.
- `build-corpus.py` — turns a markdown tree into a drawer-shaped `corpus.txt`
  (paragraph chunks, ~700 chars, deduped, token-safe cap). Run locally. The
  test corpus was 15,805 chunks from `~/src/campaigns-test`.
- `turbovec-recall-ab.py` — index-fidelity A/B: turbovec (bit-width sweep) vs
  ChromaDB vs exact brute-force kNN, recall@k on identical vectors.
- `turbovec-hybrid.py` — **superseded / kept as a cautionary artifact.** Hybrid
  test with a *circular* gold standard (see Regime 1). Replaced by
  `turbovec-target.py`.
- `turbovec-target.py` — known-target retrieval. Short queries (deterministic
  `template` keywords, or `llm` natural questions), gold = the source drawer,
  metric = hit@k. The test that actually decides adoption.

`corpus.txt` and `queries*.jsonl` are generated artifacts, gitignored.

## Result 1 — index fidelity (recall A/B, 15,805 docs, 500 queries)

`turbovec-recall-ab.py --corpus corpus.txt --chroma --bits 2,3,4`

| index | r@1 | r@10 | r@100 | p50 ms | p95 ms | build s | size |
|---|---|---|---|---|---|---|---|
| exact kNN | 1.000 | 1.000 | 1.000 | 0.66 | — | 0.33 | 48.6 MB |
| turbovec 2-bit | 0.876 | 0.872 | 0.870 | 0.55 | 0.64 | 0.76 | 3.2 MB |
| turbovec 3-bit | 0.924 | 0.937 | 0.941 | 1.14 | 1.26 | 0.87 | 4.7 MB |
| **turbovec 4-bit** | **0.968** | **0.959** | **0.967** | **0.76** | **0.85** | 1.24 | **6.3 MB** |
| chromadb HNSW | 1.000 | 0.999 | 0.971 | 9.54 | 26.46 | 14.03 | — |

- **turbovec's bit menu is 2/3/4 only — 4 is the ceiling** (`bit_width must be
  2, 3, or 4`). There is no 8-bit rung to climb toward 1.0; 0.968 r@1 is the
  best it can do. ~+5 points per bit; they run out where it still hurts.
- **Chroma's "perfect recall" was a small-N illusion.** At 284 docs it was
  1.000 everywhere; at 15k its r@100 fell to 0.971 — tied with turbovec 4-bit
  (0.967). HNSW's approximation shows up in the tail, as predicted.
- turbovec wins latency (~12× p50, ~31× p95 — HNSW has a nasty tail), size
  (~8×), and build (~11×).
- **Failure modes differ:** HNSW miss = search-path error (exact vectors,
  greedy traversal misses); turbovec miss = distance error (approximate vectors,
  ranking flips near ties). Same tail effect.

Naive read of this table: "turbovec loses 3–4 points, use Chroma." Result 3
shows why that read is wrong.

## On "100% recall"

The mempalace bar is a **store** guarantee (durability / verbatim — every
drawer physically retained), not an **index** metric. The quantized index holds
vectors *for search*; the verbatim text lives in a separate document store. So
lossy 4-bit quantization corrupts *which neighbors rank highest*, never *what
text comes back* — it cannot violate "verbatim". And "100% recall" can't be a
property of any approximate index (HNSW included), only of the store. The A/B
above measures *search fidelity*, a different thing.

## Result 2 — end-to-end retrieval, three query regimes

`turbovec-target.py` — gold = source drawer, metric = hit@k, 300 queries,
nomic `search_query:`/`search_document:` prefixes applied (a real bug we caught:
worth +4–6 vector points, changed no verdict).

We bracketed the query space. The regime decides everything:

| regime | how built | vector hit@1 | BM25 hit@1 | does hybrid help? |
|---|---|---|---|---|
| doc-as-query | full drawer text | (circular — see below) | — | — |
| keyword | distinctive terms | 0.230 | 0.910 | no — RRF *hurts* |
| **natural question** | LLM paraphrase | **0.367** | 0.313 | **yes** |

### Regime 1 — doc-as-query (circular, in `turbovec-hybrid.py`)

Gold was exact-*vector* top-k, so any non-vector ranker could only dilute the
score. Adding BM25 dropped turbovec 0.968 → 0.596. **Lesson: you cannot measure
a hybrid against a single-modality gold — the gold decides the outcome.** This
is why the script is kept only as a cautionary artifact.

### Regime 2 — keyword queries (`--gen template`)

| strategy | @1 | @10 | @100 |
|---|---|---|---|
| exact vector | 0.230 | 0.437 | 0.620 |
| turbovec 4-bit | 0.230 | 0.437 | 0.607 |
| BM25 alone | 0.910 | 0.997 | 1.000 |
| turbovec+BM25 | 0.397 | 0.867 | 1.000 |
| chromadb alone | 0.217 | 0.390 | 0.557 |

- Vector search is genuinely weak on keyword lookups (~0.23 even with correct
  prefixes) — this is *why* BM25 exists in a hybrid.
- **exact = turbovec = 0.230, identical: when vector is the weak partner,
  quantization is invisible.**
- Naive 50/50 RRF *hurt* (BM25 0.910 → 0.397): when one modality dominates,
  blind fusion drags the strong ranker toward the weak one.

### Regime 3 — natural questions (`--gen llm`) — the decisive test

| strategy | @1 | @10 | @100 |
|---|---|---|---|
| exact vector | 0.367 | 0.670 | 0.883 |
| **turbovec 4-bit** | **0.363** | **0.673** | **0.883** |
| BM25 alone | 0.313 | 0.673 | 0.863 |
| **turbovec+BM25** | **0.407** | **0.747** | **0.957** |
| chromadb alone | 0.367 | 0.670 | 0.883 |
| chroma+BM25 | 0.410 | 0.747 | 0.957 |

- **Vector and BM25 finally comparable** (0.367 vs 0.313 @1) — the regime where
  both contribute.
- **Hybrid wins here, the only regime it does:** turbovec+BM25 (0.407) beats
  both turbovec-alone (0.363) and BM25-alone (0.313); @100 climbs 0.883 → 0.957.
  Textbook justification for hybrid search — visible only when modalities are
  balanced.
- **The 4-bit deficit vanishes end-to-end:** turbovec 0.363 ≈ exact 0.367 ≈
  chroma 0.367; turbovec+BM25 0.407 ≈ chroma+BM25 0.410. The 3–4 point index-
  level loss does not survive contact with hybrid retrieval — the misses are in
  the tail, and BM25 plus "the right drawer is reachable more than one way"
  absorbs them.

## Verdict

For end-to-end hybrid retrieval on realistic queries, **turbovec 4-bit is
quality-indistinguishable from ChromaDB** while being far faster, smaller, and
CPU-resident. As a *standalone* vector index it fails the search-fidelity bar
(0.968 ceiling, no higher bit-width); inside a hybrid it's a wash on quality and
a large win on cost.

**Transferable lesson:** index-level recall metrics overstate the cost of
quantization for hybrid retrieval systems. Had we stopped at the recall A/B
we'd have wrongly rejected turbovec.

## Caveats

- One corpus (D&D prose), one embedder (nomic 768-d), 4-bit. May differ
  elsewhere.
- Fusion is naive RRF (k0=60). It *helped* in the balanced regime but *hurt* in
  the imbalanced ones — a production hybrid needs query-type routing/weighting.
  This affects both backends equally, so it doesn't change turbovec-vs-Chroma.
- ~0.4-point differences are within the noise of 300 partly-imperfect LLM
  queries; a few generated questions were off-target (fair across strategies).
- "gold = source drawer" is pessimistic (a short query may fit several near-
  duplicate drawers) but identically so for every strategy.
- Not yet tested: turbovec as a real mempalace `BaseBackend` (it's "half a
  backend" — vector ANN only; needs a sidecar doc/metadata + id-mapping store).
