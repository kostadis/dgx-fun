# Stock Cognee chunking on native Inkling

**Completed:** stock scored 46/47 complete and 44/47 fully grounded, at
10.87/14.72 s p50/p95. Its six-search agent took 263.71 s and produced a
supported core transition explanation with minor wording caveats.
[Results](RESULTS.md) · [Audited metrics](results-20261122.json).

This experiment replays the frozen 48 factual questions and the independent
Wave Echo Cave transition agent on the saved **stock Cognee index**, using the
selected native vLLM Inkling service: 411,520-token context, FP8 KV, fixed
13 GiB KV budget and generated thinking blocked. Its comparison is the
[completed adaptive-index FP8 run](../fp8-cognee/RESULTS.md).

The stock index has **78 chunks, 1,410 nodes and 5,287 edges**; the corrected
adaptive index has **759 chunks, 4,335 nodes and 19,700 edges**. Both cover the
same full Phandelver module and retain the same original Qwen ingestion
model, source hashes and Ollama `qwen3-embedding:0.6b` embedding model/digest.
The test reuses both existing indexes without reingestion. Their learned
graphs and summaries differ along with chunking, so this is an index-level
comparison rather than an isolated change to reader chunk size.

The model checkpoint, image, launch arguments, thinking controls, temperature
zero, seed 604, reader cap 4,096 and `top_k=15` match the adaptive FP8 run.
Four unmeasured synthetic warmups precede the factual questions. Each phase
uses a fresh index clone, fresh question/search sessions, disabled automatic
feedback and a distinct cache salt. Reader context differs by design; the
audit compares generation controls and question/policy hashes rather than
requiring identical retrieved passages.

The agent receives the same original question, policy and schema-guided action
format, choosing each next query after reading its preceding result. It has
an eight-search budget and 2,800-token action cap. No prior answer or reviewer
search is supplied. Its elapsed time includes the work it independently
chooses, so time must be considered alongside coverage and source errors.

`run.py` pins the running Inkling and Clef container identities, images,
commands and start times, checks endpoint readiness and the 12 GiB host
headroom guard before and after both phases, and hashes historical inputs.
It does not restart either service. The local worker has a 12 GiB memory
limit with swap disabled. `analyze.py` audits actual requests, source-context
provenance, sessions, embeddings and operational continuity before exporting
allowlisted metrics.

Answers are assessed against the frozen gold, actual supplied context and
existing rubric; completeness and grounding are separate measures. q37
retains its predeclared ambiguous exclusion, leaving 47 adjudicable questions.
q38 retains the original trigger-and-round rubric. Agent review checks the
actual Chapter 4 ending, Chapter 5 opening and unsupported claims separately.
Review is by Codex, without blinded human adjudication.

Full requests, source passages, answers, embeddings, reviews, code snapshots
and service receipts stay under ignored
`experiments/adaptive-chunking/raw/pabtso/cognee-20261120/inkling-stock-fp8-411k-20261122/`.
The public exports omit source text and full answers. These are single runs
on shared hardware at different times, without repeated-run uncertainty
estimates. The earlier adaptive FP8 run preceded the temporary BF16 test and
FP8 restart; image, weights and arguments match, but process uptime differs. AtlasKV and synthetic decode/capacity tests remain separate.
