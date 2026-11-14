# Actual Cognee shadow comparison

**Completed:** all three arms ingested the seven files and answered ten
source-grounded questions with their own reader and again with the same vLLM
reader. The existing Cognee database and configuration remain untouched.
Hibana exact is restored; fast is stopped and retained.

| Measurement | Pinned vLLM | Hibana exact | Hibana fast |
|---|---:|---:|---:|
| Ingest and persist seven dossiers | 252.57 s | 144.83 s | **65.67 s** |
| Observed ingestion speed relative to vLLM | 1.00× | 1.74× | **3.85×** |
| LLM output tokens during ingestion | 15705 | 13572 | 12427 |
| Aggregate output tokens / ingestion wall second | 62.18 | 93.71 | 189.23 |
| Correct requested answers, native reader | 10/10 | 10/10* | 10/10 |
| Correct requested answers, common vLLM reader | 10/10 | 10/10* | 10/10 |
| Semantic Entity nodes / Entity-to-Entity edges | 63 / 101 | 49 / 83 | 42 / 74 |

The token rates aggregate concurrent pipeline calls and include persistence
time; they are not single-stream decode rates. Fast emitted 20.9% fewer output
tokens than vLLM, so the 3.85× task gain includes a change in extracted output.
The aggregate token-wall-throughput gain was 3.04×. All three full runs made
15 ingestion LLM calls and 40 calls for 20 recalls, all HTTP 200 with natural
stop finish reasons. All seven TextDocuments, DocumentChunks and TextSummaries
were persisted in each graph.

*Exact returned the correct Orb owner but reported conflicting evidence. A
targeted graph audit found an incorrect `Senshock --possesses--> Orb of Golden
Death` edge, directly present in the captured model extraction response, as
well as the correct Falrinth ownership edge. The raw erroneous extraction is
call `0765690bbfe14d20836c74c350d659eb`. Both readers noticed the conflict and
selected Falrinth using the source chunks. This was not a clean graph result.

Fast kept the Orb holder correct. Its concrete omission was different:
**the 4,000-gold ring deposit is absent from its semantic Entity nodes and
Entity-to-Entity edges**, while both other graphs retained it. The ring offers
survived in Sequoia's and Zephyr's descriptions, but without the amount.
Both readers still answered the deposit question correctly from the retained
Lucius source chunk. Thus correct recall does not establish equivalent
semantic extraction coverage.

The tested answers preserved the planned Hedrack meeting, the absence of a
recorded Senshock meeting, unknown current location, scarab warnings and
letters, Orb possession/refusals, payment policy, sword prices, delivery time,
and ring deposit. The unsupported Ironhand/blacksmith question returned
unknown. The ten questions and expected facts were frozen before ingestion;
answers were manually reviewed against them. This is a bounded real-workload
result, not exhaustive graph precision/recall or a full-campaign qualification.

There was one full run per arm. The vLLM/exact jobs overlapped on independent
servers and shared the existing embedding provider; fast ran subsequently.
Both vLLM and exact were primed by a one-document smoke test, as was fast.
Model/prefix/OS caches were not flushed, and no primary-service traffic counter
audit was captured for this follow-up. Query latency also included overlapping
common-reader traffic. Treat timings as observed workflow results, not a
controlled estimate of sustained capacity.

Artifacts: [public measurements and answer hashes](results.json),
[manual quality review and targeted graph checks](quality-review.json),
[frozen corpus manifest](corpus-manifest.json), [questions](cases.json).
Full requests, responses, source files and databases remain in ignored `raw/`.
Full answer text and semantic-edge claims also stay local. `analyze.py` exports
redacted receipts by default; `--include-private` requires an output beneath
`--raw`. The consolidated [results report](../../../hibana-spark-results.md)
contains the selected quality findings needed to interpret the measurements.

This follow-up runs the user's actual Cognee ingestion and graph recall,
using seven complete ToEE NPC dossiers (7,039 words). The prior synthetic
80-entity graph fixture is not reused.

Corpus hashes: [manifest](corpus-manifest.json). Source-grounded questions and
expected answers were frozen before ingestion in [cases.json](cases.json).
The cases include negation, planned versus completed meetings, alias-bearing
characters, item possession, financial amounts, timing, and explicit unknowns.

Runtime: existing `/home/kostadis/cognee-local/.venv/bin/python`, with editable
Cognee source at `/home/kostadis/src/cognee`, commit
`b32d8afc59e1064d9291b9828a8a147be9cc8bab`. Installed distribution metadata
reports Cognee 1.5.3 while the editable checkout declares 1.6.2. Installed
dependencies include LiteLLM 1.100.1, Instructor 1.15.1, Ladybug 0.19.0 and
LanceDB 0.38.0. This is the existing runtime, not a freshly resolved environment.
No installed packages or tracked Cognee source files are changed for the trial.

All graph, SQL, vector, log and cache roots are separate directories beneath
this experiment's ignored `raw/` directory. `COGNEE_ENV_FILE` explicitly selects
each arm's private configuration; the runner asserts storage isolation before
ingestion. The user's existing `cognee-toee` dataset/configuration is not used
as a write destination. No prune, forget or delete operation is called.

Configuration retains `CACHING=true`, `AUTO_FEEDBACK=true`, and the existing
single-user mode. Bulk ingestion uses `remember(..., self_improvement=False,
data_per_batch=2)`, as in the user's existing bulk-ingestion workflow. The
capture proxy allows four in-flight LLM calls and records queue and upstream
time separately. It changes only the outbound served model ID. Requests retain
Cognee's actual prompt and structured-output settings. No hand-written graph
extraction prompt substitutes for the pipeline.

All arms use the same existing Ollama embedding endpoint,
`http://192.168.1.121:11434/v1`, model `qwen3-embedding:0.6b`, dimensions 1024.
The runtime warns that its model tokenizer falls back to TikToken, matching the
configured OpenAI-compatible embedding path; no transformer package is added.
No reasoning tokens are requested. Sampling is greedy with seed 604.

`run.py` preserves the actual SDK requests, responses, usage, HTTP errors and
retry attempts in `raw/<arm>/calls/`, plus the persisted graph, `remember()`
result, per-question recall responses and a summary. Graph size is an inventory
metric, not a quality score.

Recall uses Cognee's `GRAPH_COMPLETION` path and its returned references.
It can include source chunks reached through the graph, so a correct answer
does not by itself prove every semantic edge is correct. The graph edges
also need review against the dossier text.

Each arm first uses its own model as reader (end-to-end behavior), then the
same retained vLLM reader (to compare stored memories with a common answer
model). The capture log records the actual route so that intended stage
routing can be verified. The frozen expected answers are used for manual
source review; successful ingestion or valid JSON is not counted as semantic
quality parity.

The primary vLLM service on spark1 is retained. Hibana is reached over a
temporary SSH tunnel to its loopback API; exact and fast are served in turn
on spark2, with the same pinned source/checkpoints as the preceding trial.

One-document smoke tests completed: Hedrack ingestion/persistence took
51.8 s on vLLM, 28.2 s on Hibana exact and 14.4 s on Hibana fast. All correctly answered that the
chapter-40 meeting was planned rather than completed, with Barkinar as the
intermediary. These smoke points are separate from the full-run figures above.
