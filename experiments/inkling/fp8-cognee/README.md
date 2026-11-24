# Cognee replay on native 411K FP8 Inkling

**Completed:** all 48 questions and the six-search module agent finished.
Factual p50/p95 were **6.03/8.41 s**, with **44/47 complete and fully grounded**.
The agent took **180.03 s**, recovered the actual handoff, and retained a
belief-attribution error. [Results](RESULTS.md) · [Audited metrics](results-20261122.json).

A later [stock-index comparison](../stock-cognee/RESULTS.md) held this serving
configuration fixed: stock measured 10.87 s median and 46/47 complete, versus
this adaptive run's 6.03 s and 44/47; both had 44/47 fully grounded.

This follow-up repeats the existing 48-question benchmark and the independent
Wave Echo Cave transition agent on the selected **411,520-token native vLLM
Inkling service**. It is a small-prompt workload comparison with the historical
Qwen and native 64K BF16 results. The separate
[near-limit capacity test](../context/RESULTS.md) already exercised 410,496
input tokens; these factual and agent prompts do not fill the larger window.

The model service stays running throughout. Native container identity, image,
start time, command, advertised context, Clef identity/readiness and the active
12 GiB host-headroom guard are checked before and after both phases. The
13 GiB fixed FP8 KV budget and pinned checkpoint are unchanged. The worker
has a 12 GiB VM memory limit and no swap allowance.

Both phases use the existing corrected adaptive index: 759 chunks, 4,335 nodes
and 19,700 edges, with Qwen-ingested graph/summaries and Ollama
`qwen3-embedding:0.6b` embeddings reused. No ingestion is performed. Each phase
clones the saved index and each question/search has a fresh session, with
automatic feedback disabled. Historical/input files are protected with hashes.

Factual readers retain the frozen 48 questions, `top_k=15`, one retrieval and
one answer per question, temperature zero, seed 604 and a 4,096-token cap.
Generated thinking is blocked using native `bad_words`; requested effort stays
`none`. Distinct cache salts isolate each phase from historical cached prefixes.
The same four synthetic arithmetic/JSON/negation/tool warmups precede the
factual run and are excluded from measured query times.

The agent uses the original question, research policy, schema-guided actions,
`top_k=15`, eight-search maximum and 2,800-token action cap. It reads each
observation before selecting its next query, without prior answers or
reviewer-directed searches. Its reader cap remains 4,096 tokens.

Factual source labels are carried forward only for byte-identical answer text
and identical supplied reader messages. New answers are reviewed against the
frozen gold, actual supplied context and existing rubric. q37 stays excluded
as predeclared ambiguous; q38 retains its original trigger-and-round scope.
Correct completeness and grounding are reported separately. Agent final
answers receive a separate source review. Review is by Codex rather than
blinded human adjudicators.

Full requests, answers, source context, embedding captures, reviews, commands,
code snapshots and service receipts stay under ignored
`experiments/adaptive-chunking/raw/pabtso/cognee-20261120/inkling-native-fp8-411k-20261122/`.
`run.py` controls the bounded replay; `analyze.py` verifies request/source
provenance and operational continuity before exporting public metrics.

These runs change KV precision, context limit and cache allocation together.
Historical comparisons occurred at different times on shared devices and do
not isolate FP8 cost or establish repeated-run confidence intervals. Agent
elapsed times also include different, independently selected search work.
AtlasKV and the earlier synthetic decode experiments remain separate.
