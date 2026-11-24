# Phandelver: LLM Wiki / CROW versus Cognee

The isolated full-module benchmark completed November 23, 2026 (Pacific).
See the [results](RESULTS.md) and [audited metrics](results-20261123.json).
It uses the same frozen
Phandelver corpus, 48 questions (47 adjudicable) and Wave Echo Cave transition
question as the existing Cognee experiments. No production memory or Cognee
index is edited. Module text, notes, prompts, answers and full captures stay in
ignored `experiments/adaptive-chunking/raw/pabtso/cognee-20261120/llm-wiki-clef-20261124/`.

The upstream library is unmodified **llmwiki2 1.0.5**, commit
`3333c71485bf271a89544090647a91abe25c70ca`, licensed AGPL-3.0-or-later. It is
installed in an isolated temporary environment. This is an experimental adapter,
not a deployed wiki server or a replacement for the existing applications.

## Local services

- Writer and answer model: existing native vLLM **Inkling**, Spark 2 port 8004,
  `inkling-2-mini`, 411,520 context, FP8 KV, fixed 13 GiB cache. Thinking is
  blocked with the verified native token ban; temperature 0 and seed 604.
- CROW classifier: existing stock **Clef**, Spark 2 port 8002,
  `/v1/systemone`, model `clef`. The specialized approval head
  `clef-card-tuned` is not selected. Choice and noul synthetic checks passed.
- Embeddings: existing Ollama at `192.168.56.1:11434`, unchanged
  `qwen3-embedding:0.6b` digest, 1,024 dimensions, API truncation disabled.
  The upstream wiki embeds each note's title, summary and first 3,000 body
  characters; its lexical search reads the full note body. This preprocessing
  differs from Cognee even though the embedding model and digest match.
- No module data is sent to cloud providers. The unnecessary Tev1 setup probe
  used synthetic data only. Its unused trial model and alias were unloaded and
  removed, restoring the original embedding-only Ollama inventory.

The writer reads up to 100,000 characters per call, retains upstream
proportional note-length behavior and has a 16,384-token output cap. Batch
preparation is serialized. Clef uses the library's custom-model preset without
question-based calibration, a 24,000-character request bound and 120-second
client timeout. These are deployed local settings, not a reproduction of the
paper's Jev/GLM configuration. The persisted config records every threshold.

## Measurements

1. Ingest the same thirteen source documents, preserving their full rendered
   text and headings. CROW decides its own note boundaries, placement and merges.
   Record preparation/filing time, decisions, model usage, errors and source loss.
2. Build the note-vector cache with one generic, unmeasured query before timing
   factual questions. Each measured arm gets a separate clone of the saved wiki.
3. Replay all 48 facts with CROW hybrid retrieval and classifier reranking,
   then repeat with reranking disabled. Hold the answer policy (120 words),
   4,096-token cap, model and generation controls fixed. The library's supported
   custom answer-step interface supplies this common reader; upstream CROW
   selection, history/conflict handling and postprocessing remain enabled.
   Both arms explicitly set `crow.k=8`: the custom-provider preset otherwise
   chooses five. The offline regression checks the number of notes that actually
   reach the answer step. Reranking off still runs the conflict check.
4. Record BM25-only selection separately without answer generation. A source
   document represented in a selected note is a routing diagnostic, not evidence
   that its exact facts survived summarization or that an answer is correct.
5. Run the independent search/finish agent with an eight-search budget,
   2,800-token action cap and one newly chosen query after each observation.
   It receives derived wiki notes as evidence, not the complete original
   chapters behind them. Final source review uses the original module.

All HTTP model requests and responses are captured; the client rejects network
calls outside the three local service addresses. The experiment checks the
native/Clef process fingerprints and active memory guard before and after
phases, hashes historical inputs, and checks that queries do not alter wiki text.
The worker has a 12 GiB RAM limit with swap disabled. Neither DGX service is
restarted for this trial.

## Comparison limits

The completed [stock Cognee](../inkling/stock-cognee/RESULTS.md) and
[adaptive Cognee](../inkling/fp8-cognee/RESULTS.md) runs used this Inkling serving
configuration and generation controls for answers, but their graphs/summaries
were originally ingested by **Qwen**. The new wiki is ingested by **Inkling**.
Ingestion time and quality therefore do not isolate the indexing architecture.
Runs also occur at different times on shared hardware.

Cognee's agent receives original module chunks; the wiki agent receives its
retrieved generated notes. That distinction is part of the system comparison
and will be explicit in the result. We score answer correctness and grounding
against the original book, separately from whether a relevant note was found.
The upstream answer prompt is replaced only for the controlled reader comparison;
this is not a claim to measure unmodified end-user `/ask` behavior.

The frozen rubric retains q37's ambiguous exclusion and q38's original
trigger-and-round scope. Review is by Codex, not blinded human adjudication.
Recursive-600 remains the strongest prior vector baseline; these measurements
do not establish a general ranking over it or a recursive Cognee graph index.

## Replaying and monitoring

The pinned checkout is `/tmp/dgx-llmwiki-v2-20261124`; its isolated Python is
`/tmp/llmwiki-env-20261124/bin/python`. With a fresh output directory under the
ignored trial root, run `ingest.py --output <directory>`, then
`query.py --root <directory>` using that Python. The frozen corpus is private
and must already exist at the path used by the Cognee trials.

To recreate the temporary dependencies after they have been removed, use fresh
paths matching the constants in `runtime.py`:

```bash
git clone https://github.com/crow-hq/llm-wiki-v2 /tmp/dgx-llmwiki-v2-20261124
git -C /tmp/dgx-llmwiki-v2-20261124 checkout 3333c71485bf271a89544090647a91abe25c70ca
uv venv /tmp/llmwiki-env-20261124
uv pip install --python /tmp/llmwiki-env-20261124/bin/python /tmp/dgx-llmwiki-v2-20261124
/tmp/llmwiki-env-20261124/bin/python -m unittest discover -s experiments/llm-wiki -p 'test_*.py'
```

The bounded workers were `llmwiki-clef-ingest-20261124.service` and
`llmwiki-clef-query-20261124.service` in the local user's systemd instance.
Both exited successfully and are inactive. The second waited for ingestion
to finish before querying. Progress is saved
in `ingest-progress.json` and `query-progress.json`; full model captures are
in `calls/`. A daemon/chat restart does not require restarting these workers.

The common answer policy and caps match the Cognee benchmark; evidence
packing and formatting differ by backend. Requests are not byte-identical.

Clean committed-tree validation passed **100 repository tests** and **two
adapter regressions**, covering the actual eight-note budget and failure of the
final ingestion service check. The original experiment working tree passed
115 tests, including separately pending chunker tests; that historical receipt
is retained separately in the metrics. Use the documented tokenizer cache:

```bash
TIKTOKEN_CACHE_DIR="$PWD/experiments/adaptive-chunking/raw/tiktoken-cache" python3 -m pytest -q tests
```

`analyze.py --root <completed-directory> --output <metrics.json>` audits the
captures and private source-review labels before exporting metrics. It requires
the final service receipt and reviewer flags retained with this completed run.
Measured source files are preserved under the ignored run's `executed-source/`
directory, with their recorded hashes. The audit verifies those exact files
when a later replay fix changes the current harness. The post-run failure-status
fix changes no benchmark measurements and requires no model rerun.
