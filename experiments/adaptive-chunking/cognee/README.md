# Phandelver in real Cognee

The [combined write-up](../../../phandelver-cognee-agent-results.md) connects the
factual benchmark, context repair and Qwen/Haiku/Sonnet sequential-agent results.

This compares Cognee's installed stock `TextChunker` with the frozen adaptive
boundaries from the Phandelver retrieval pilot. It runs actual `remember` graph
extraction, summarization, indexing, and `GRAPH_COMPLETION` recall of all 48
frozen questions. Both arms ingest all 13 sections of the same expanded book.

The adaptive selector is the earlier upstream four-metric implementation (no
coreference metric). Its per-section choices are reused without inspecting the
recall questions. The corrected adapter preserves each chunk's already-frozen
heading breadcrumb in `DocumentChunk.text`, so graph extraction, summarization,
and vector indexing receive the same parent context. The original raw-chunk
behavior remains available with `--adaptive-context raw`. Neither experiment
isolates chunk boundaries from size.

The completed [full-book report](../../../phandelver-cognee-chunking-results.md)
and [metrics](results.json) show 46/47 complete correct answers for current
Cognee versus 43/47 for this raw-chunk adaptive integration. Adaptive reduced
p95 recall latency from 14.04 to 6.65 seconds, while ingestion increased from
64.85 to 230.28 minutes. One item effect lost its parent heading and was
assigned to the next item in generated summaries. The completed
[heading-preserving follow-up](context-results.json) recovered that answer and
two other misses, introduced a new retrieval miss, and reached **45/47** with
**6.16 s p95** recall. Stock remains ahead at 46/47. Six adjacent-item effect
attributions are now correct, but one graph description still gets Quickened's
rarity wrong. The recovered full run used eight calls and was interrupted by
a client VM crash, so its ingestion time is not a continuous benchmark. The
original results and indexes remain unchanged.

An additional [narrative synthesis probe](narrative-probe-results.json) asked how
Wave Echo Cave feeds into the second half. Stock broadly recovered the transition;
the corrected adaptive index missed the handoff and invented a Forge/Obelisk
connection. This question is outside the frozen score and shows a limit of the
short fact-question benchmark. A follow-up at `top_k=60` recovered the main
transition and dropped that invented connection, while still missing some
details. The original failure therefore does not isolate adaptive chunking from
the one-pass retrieval budget. Continued retrieval was then tested.
An explicit gap-driven continuation was subsequently tested: two follow-up
searches expanded the retrieved raw source union from 26 to 40 and recovered
the missing Chapter 5 connection. The final reader used those raw passages.
Source review still caught overstatement after the model declared sufficient
evidence. A reviewer-directed third search expanded the union to 49 and the
revised source-only answer corrected the causal and player-knowledge distinctions.
See the report for the trace and limits. This does not change the frozen
48-question score.

On that narrative question, the same continuation and validation procedure took
118.69 seconds and five model calls for stock, versus 182.00 seconds and nine
calls for adaptive. Model tokens totaled 81,125 versus 101,702. The report records
the different seed budgets and remaining quality differences; these are single
observations from a sequential prototype.

The subsequent [one-question-at-a-time agent test](sequential-results.json)
starts fresh on both indexes at `top_k=15`. Qwen reads each Cognee answer and
the accumulated raw sources before choosing one next question. Stock stalled
after two searches, repeating its follow-up despite protocol corrections;
adaptive finished after five. Client elapsed times were 65.70 seconds until
stock's stall and 143.41 seconds for adaptive, using 146,757 and 61,432 tokens
respectively. Adaptive recovered the Chapter 5 bridge but still invented a
corpse note and an unsupported bandit/mind-flayer hierarchy. Completion did not
establish grounding. Full traces are saved privately; the report distinguishes
these outcomes from a speed comparison at equal quality.

The [Haiku follow-up](haiku-sequential-results.json) reused the adaptive index
and policy, changing both reader and agent to Claude Haiku 5.5. It stopped after
two searches, five calls and 28.72 seconds, using 32,051 tokens at an estimated
$0.0045723 in API charges. Its final answer avoided Qwen's invented links but
left the main transition unanswered, with six searches unused. Including two
preserved compatibility attempts, estimated API charges totaled $0.0132338.
This was a more cautious incomplete answer, not a quality-matched speed win.

The [Sonnet completeness check](sonnet-sequential-results.json) used the same
adaptive index and research policy with Sonnet 5.5 as both reader and agent.
It recovered the actual Chapter 4 ending and Chapter 5 handoff in three searches,
finishing in 52.98 seconds with six calls and 61,632 tokens, at an estimated
$0.161656 API charge. Source review found the main handoff supported without
Qwen's invented links. This is the strongest reviewed answer on this one
question; the report records the remaining source-coverage limits.

The [Inkling 2 Mini follow-up](inkling-sequential-results.json) used the original
mixed-precision model on Spark 2 through a separate OpenAI-compatible endpoint.
Its same-policy agent stopped at the action-schema guard after two searches,
six calls, 30,429 tokens and 137.94 seconds until failure. It proposed a new
query inside an `args` array and omitted required control fields in all three
attempts. Basic text and JSON generation passed; no final narrative answer
was produced. The [installation and client guide](../../inkling/README.md)
records the pinned checkpoint/image and shared-host configuration.

An opt-in [schema-guided Inkling rerun](inkling-schema-sequential-results.json)
resolved that action-format failure. Add `--action-format json_schema` on an
OpenAI-compatible backend; the default remains `json_object`. The server
enforces required field names/types while the existing semantic validator
remains unchanged. This affects agent decisions, not Cognee reader calls.
The fresh run finished after seven searches and fourteen calls, in 499.32
seconds and 104,978 tokens, with zero protocol corrections. It recovered the
main handoff, with remaining geography/GM-wording caveats recorded in the
source review. An exact replay of the failed prompt also passed after changing
only `response_format`. This is a separate decoding-constraint diagnostic;
the original failure and its cost are preserved.

The [original 48-question Inkling replay](inkling-question-results.json) then
completed without an agent. All reader messages and raw source sets matched
the Qwen baseline. Inkling took 12.10 s p50 and 18.60 s p95, versus Qwen's
4.68 s and 6.16 s; the query sums were 636.25 s versus 226.79 s. It scored
44/47 under strict review, or 45/47 if one misleading Ruxithid sentence is
accepted as awkward wording, versus Qwen's 45/47. Both existing retrieval
misses remained. The report records this scoring sensitivity and the unequal
serving configurations; this trial shows no factual accuracy gain.

The [Hibana fast factual replay](hibana-fast-question-results.json) used all 48
identical Qwen reader messages, reducing p50/p95 to **3.41/4.32 s** and the
query sum to **168.99 s**. Complete answers scored **44/47** versus Qwen's
45/47; an added Agatha knowledge contradiction caused the new miss. Strict
review of additional assertions leaves 42/47 fully grounded complete answers.
Its [one-pass module answer](hibana-fast-onepass-results.json) took **14.87 s**
but invented a transition. The [five-search agent](hibana-fast-sequential-results.json)
took **84.51 s**, recovered the broad chapter structure, and still confused
GM shard-scout information with a party-discovered reason to move on. These
reuse the Qwen-ingested index; Hibana fast changes calibrated weights and
decoding policy as well as engine. Full quality notes and timing limits are
in the report.

## Factual questions with a different reader model

`questions.py` replays the original 48 frozen factual questions against an
isolated copy of an existing index. It uses one `GRAPH_COMPLETION` recall per
question, `top_k=15`, the original 120-word instruction, and a fresh session
for every question. Questions run sequentially. It performs no ingestion or
agent decisions and requires the frozen question file to match the index's
recorded hash. The original graph, summaries and embeddings retain their
ingestion identity.

```bash
trial=experiments/adaptive-chunking/raw/pabtso/cognee-20261120
/home/kostadis/cognee-local/.venv/bin/python \
  experiments/adaptive-chunking/cognee/questions.py \
  --source "$trial/adaptive-context-8" \
  --output "$trial/inkling-questions-new/adaptive" \
  --cases "$trial/frozen/cases.json" \
  --backend http://192.168.1.121:8004 --model inkling-2-mini \
  --reasoning-effort none
```

Use a new output directory for each run. The runner saves full answers,
individual timings, actual model requests and embedding captures under ignored
`raw/`, and checks that cloned graph counts remain unchanged. It does not need
the agent action-schema workaround: these calls generate ordinary answers.

## Backends and isolation

- LLM: running Spark 1 vLLM, `qwen3.8-flash-next`, port 8001.
- Embeddings: user-selected `http://192.168.56.1:11434`,
  `qwen3-embedding:0.6b`, 1,024 dimensions.
- Existing Hibana/Cognee `Capture` proxy records real SDK requests. The runner
  sets its concurrency using `--llm-concurrency` (default four, maximum eight).
  The original trial used four; the corrected full follow-up uses eight at the
  user's request. Hibana itself is not the serving engine in these runs.
- An embedding proxy translates the OpenAI request to native Ollama `/api/embed`
  with `truncate:false`; it verifies vector counts, dimensions, and finiteness.
- Each arm has separate relational, graph, vector, cache, log, and source roots.
  The existing Cognee installation and `.env` are reused read-only.
- Automatic feedback is disabled so one answer cannot become evidence for a
  later question through feedback extraction. Each question also uses a fresh
  UUID session: caching alone otherwise adds earlier answers to the default
  session's prompt even when automatic feedback is disabled. Captured requests
  are checked for other benchmark questions. Self-improvement is disabled;
  cache is enabled per arm.
- Chunk batches are four, documents are processed sequentially. Successful
  document and question checkpoints allow an identical run to resume.
- Actual persisted chunk text must match the full planned chunk census.
  HTTP failures and completion truncation fail the run.
- Contextual chunks retain `raw_text`, source offsets, and the exact frozen
  breadcrumb separately in their census. Evidence coverage counts original
  source spans only. Content hashes and token budgets use the contextual text.
  A changed context policy requires a fresh index; it cannot resume a raw index.

The installed native adapter uses `LLM_MAX_COMPLETION_TOKENS` for configuration
and automatic chunk sizing, but its observed SDK requests omit `max_tokens`
and `max_completion_tokens`. Actual output limits therefore come from the
served model/server defaults. The capture's upstream HTTP timeout is 1,800
seconds. Recovered JSON-schema validation retries are counted separately from
HTTP failures; JSON-object mode does not enforce the complete schema at the
server.

## Run

Use the existing Cognee Python environment; do not install the pilot's pandas
dependencies into it. A full run takes hours, especially the 759-chunk adaptive
arm. The current automatic configuration resolves to a 4,096-token cap and
78 chunks on this corpus. Token counting uses the installed Cognee fallback
tokenizer for this Ollama-style model name; native Ollama rejects oversized
inputs instead of silently truncating them.

The sequential controller freezes all four input JSON files before either arm
starts, records its stage in `control.json`, stops on an arm failure, and
exports aggregate metrics plus a private answer-review worksheet:

Use an unused output directory for a new trial. Existing arm directories
require `--resume`; the controller rejects accidental reuse before changing
its status file.

```bash
python3 experiments/adaptive-chunking/cognee/trial.py \
  --output experiments/adaptive-chunking/raw/pabtso/cognee-20261120
```

For individual runs:

```bash
/home/kostadis/cognee-local/.venv/bin/python \
  experiments/adaptive-chunking/cognee/run.py \
  --arm current --output experiments/adaptive-chunking/raw/pabtso/cognee-20261120/current

/home/kostadis/cognee-local/.venv/bin/python \
  experiments/adaptive-chunking/cognee/run.py \
  --arm adaptive --output experiments/adaptive-chunking/raw/pabtso/cognee-20261120/adaptive
```

Use `--phase plan` for chunk/call estimation without ingestion or LLM calls;
`--smoke` ingests only Appendix B and recalls its six item questions. These require
their own output directories. Use `--resume` only for the same arm, code,
inputs, model digest, and output directory; the manifest enforces identity.
An explicitly audited code-only migration additionally requires the exact
previous `--resume-code-sha`; it preserves the original manifest and source
snapshot. This was used to add independent recall sessions after the first
22 default-session responses were archived. The persisted graph was verified
identical across that restart. The archive is excluded from primary recall
metrics; total wall time and embedding totals retain its overhead.

All source text, vectors, questions, model requests, answers, and graph dumps
stay in ignored `raw/`. `progress.json` updates every 15 seconds. Per-document
`remember/` files, per-question `recall/` files, and `summary.json` hold results.
Zero in the dry-run dollar estimate means no pricing entry for the local model;
it does not mean zero compute cost.

## Corrected adaptive follow-up

Reuse the saved stock baseline and frozen inputs, with a fresh adaptive output
root. Do not run the two-arm controller against the completed original trial.
After the item regression passed, the user indicated the server's eight-request
capacity. The initial four-call full rebuild was stopped and preserved as a
partial pilot; the complete follow-up starts from a fresh index at eight calls.
Its ingestion timing therefore combines the context fix and increased
concurrency and cannot isolate either one's speed effect.

```bash
trial=experiments/adaptive-chunking/raw/pabtso/cognee-20261120
/home/kostadis/cognee-local/.venv/bin/python \
  experiments/adaptive-chunking/cognee/run.py \
  --arm adaptive --adaptive-context breadcrumb --llm-concurrency 8 \
  --output "$trial/adaptive-context-8" \
  --corpus "$trial/frozen/corpus.json" --cases "$trial/frozen/cases.json" \
  --chunks "$trial/frozen/chunks.json" --selection "$trial/frozen/selection.json"

python3 experiments/adaptive-chunking/cognee/audit_context.py \
  --root "$trial/adaptive-context-8" --output "$trial/context8-input-audit.json"

python3 experiments/adaptive-chunking/cognee/analyze.py \
  --run "$trial" --adaptive-root "$trial/adaptive-context-8" \
  --baseline-runner-sha 843299f1586f2c01ebecc89ca922ce6c269cff3f41b63a25117a30fb31bde1f4 \
  --allow-concurrency-change \
  --cases "$trial/frozen/cases.json" --output "$trial/context8-metrics.json"
```

The analyzer requires an explicit audited baseline runner hash for this
comparison: the preserved baseline ran the earlier code. All logical input,
model, embedding, capture, session, dependency, and chunk-budget identities must
still match. It exports both runner hashes, context policies and concurrency
settings; differing concurrency requires explicit acknowledgment. The input
audit verifies source spans, persisted contextual chunks, and captured inputs
to both extraction and summarization. It also matches retrieved chunk evidence
against the actual reader requests; provenance IDs alone do not pass this check.
It does not grade model interpretation.
Review all frozen answers before reporting corrected recall quality.

The completed corrected run retained six successful section checkpoints after
the VM crash and reprocessed the interrupted seventh section. All 119 chunk IDs
and texts in that section remained stable, and the final exact 759-chunk graph
census passed. For recovery, repeat the same individual runner command with
`--resume`, preserving context policy, concurrency, code and input identities.
Back up the isolated storage first. The run used a transient user service to
continue independently of the chat session; checkpoints provide VM recovery.
Keep failed-attempt work and downtime separate from successful-document timing.

All 48 frozen answers were reviewed against book evidence and supplied model
context. q37 remains excluded for its predeclared ambiguity. As in the original
assessment, q38 is scored on its requested trigger and combat round; its battle
label should be clarified in a future fixture. All 759 contextual embedding
inputs and both ingestion stages, plus 329 retrieved chunk appearances in actual
reader requests, passed the input audit. Seventeen local checks passed.

## Sequential research agent

`sequential.py` chooses one new query only after observing the previous Cognee
answer. Each search uses a fresh session and the same saved index; automatic
feedback is disabled. The agent sees unverified tool answers separately from
the union of raw passages actually supplied to readers. It may finish on its
own, or reach the search budget. A repeated query is rejected with up to three
protocol attempts. An unused `answer` field on a search, or `query` field on a
finish, may be omitted. Protocol guards do not grade semantic correctness.
Plain JSON and a single complete Markdown JSON fence are accepted. Malformed
JSON and invalid actions share the same bounded correction mechanism; prose
around an action is rejected with feedback rather than silently discarded.

Use unused output directories. The following runs both arms sequentially from
the same saved narrative question; no earlier answers are loaded:

```bash
trial=experiments/adaptive-chunking/raw/pabtso/cognee-20261120
probe="$trial/sequential-new"
/home/kostadis/cognee-local/.venv/bin/python \
  experiments/adaptive-chunking/cognee/sequential.py \
  --source "$trial/current" --output "$probe/current" \
  --question-file "$trial/synthesis-wave-echo-20261121/question.json" \
  --top-k 15 --max-searches 8

/home/kostadis/cognee-local/.venv/bin/python \
  experiments/adaptive-chunking/cognee/sequential.py \
  --source "$trial/adaptive-context-8" --output "$probe/adaptive-context-8" \
  --question-file "$trial/synthesis-wave-echo-20261121/question.json" \
  --top-k 15 --max-searches 8
```

`protocol.json` records code and question hashes and the complete agent policy.
`steps/` records observations, valid decisions and rejected actions; `calls/`
captures actual requests and responses. Success writes `result.json`,
`evidence.json` and `answer.txt`. Failure leaves `progress.json` with the error
and the complete captured trace; it must not be counted as a completed answer.
Review the final answer against source passages after the agent stops. The
completed trial and its separately preserved first attempts are under
`sequential-agent-20261121/` and `sequential-agent-20261121-attempt1/` in the
private trial root. Their audited costs, including failed-attempt overhead,
are in the public receipt.

For a Haiku generation comparison on the same adaptive index, add
`--provider anthropic --model claude-haiku-5-5 --api-key-file ~/.anthropic-key`
and choose a fresh output directory. This uses Haiku for both research-agent
decisions and Cognee reader answers. Graph extraction and stored summaries stay
from the original Qwen ingestion; embeddings still use the recorded Ollama model.

The Anthropic adapter sends native Messages requests and captures the actual
native body and response alongside the local compatibility representation.
Authentication headers and key contents are never written to captures or `.env`.
Haiku thinking is disabled. For a Sonnet comparison, use
`--provider anthropic --model claude-sonnet-5-5` with the same key-file option
and a fresh output directory. Sonnet uses `between_tools`, its supported minimum
thinking setting, which suppresses up-front thinking. Actual thinking-token
usage is retained. Unsupported Qwen seed, temperature, chat-template,
and JSON-object options are omitted. Decisions request JSON through the same
research policy, without adding a schema-enforced stopping advantage. Reader
outputs have a 4,096-token ceiling; decisions keep the 2,800-token ceiling.
Provider sampling and tokenizer differences limit direct token comparisons.
Native usage counters are retained for pricing the actual calls, including
protocol retries. This sends retrieved source passages to the external provider.
