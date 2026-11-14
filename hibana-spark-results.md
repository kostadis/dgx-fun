# Hibana on DGX Spark Results

**Measured 14 November 2026, using Qwen3.8 Flash Next on one Spark per engine.**

Hibana fast completed a real Cognee ingestion in **65.67 seconds**, compared
with **252.57 seconds** on the retained vLLM service: an observed **3.85× gain**.
All ten requested facts were answered correctly, both through fast itself and
through a common vLLM reader querying the resulting memory. The semantic
graphs differed, however. Fast omitted a deposit amount that Cognee recovered
from the original document; exact introduced a conflicting item-ownership edge.

The earlier replay tests also showed gains: fast roughly doubled completion
throughput for 200K-token campaign calls and four persistent sessions. A
controlled synthetic JSON task reached **196 output tokens per wall second**,
four times vLLM's rate. These results support continued shadow deployment,
with answer quality and graph coverage checked separately from speed.

## Configuration and scope

The [Hibana preprint](https://arxiv.org/abs/2611.10469) motivated the test. It
combines changes to model representation, persistent session state and
speculative decoding for unified-memory devices. This experiment compares
usable local configurations against an already tuned, pinned vLLM deployment.
It covers Qwen3.8 Flash Next; Nemotron and cross-node tensor parallelism are
outside its scope.

Two GB10 DGX Sparks ran independently. Spark1 retained the primary vLLM
container throughout. Spark2 ran Hibana as a loopback-only shadow service;
its previous Clef container was stopped and retained. Application clients and
the existing Cognee database were preserved.

| Setting | Pinned vLLM | Hibana exact | Hibana fast |
|---|---|---|---|
| Base checkpoint | Qwen3.8 Flash Next NVFP4 | Same revision | Same revision plus calibrated three-bit copies |
| Side layers | Existing FP8 hybrid | Hibana FP8 trunk | Calibrated decode copies |
| Routed experts | NVFP4 | NVFP4 | Three-bit weights and expert sharing |
| KV format | BF16 | FP8 | NVFP4 |
| Speculation | MTP with two draft tokens | MTP with four draft tokens | Adaptive MTP up to eight |
| Acceptance | Existing vLLM behavior | Exact verification | Relaxed acceptance enabled |
| Maximum active requests | 8 | 4 | 4 |
| KV capacity | About 600K tokens | 300032 tokens | 300032 tokens |
| Request context limit | 262144 tokens | 262144 tokens | 262144 tokens |

Hibana exact disables its optional three-bit copies, expert sharing and
relaxed acceptance. Its name does not promise identical floating-point
results to vLLM or identical answers across cache paths. The different KV
formats, pool sizes, kernels and draft policies mean these are configuration
comparisons rather than an isolated engine ablation.

The publisher checkpoint revision was
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`. Hibana was pinned to
[`d474b74`](https://github.com/sharar-project/hibana/tree/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb).
Fast used the calibrated
[`5c4c62a62853` checkpoint](https://huggingface.co/yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3-qad/tree/5c4c62a62853caebea8cedda7fbed952efa1effe),
verified against its manifest. The vLLM image identity and full launch
configuration are retained in the [baseline receipt](experiments/hibana/evidence/before-spark1.txt).

## Real Cognee ingestion

The existing Cognee runtime ingested seven complete ToEE NPC dossiers,
**7039 words**, through its actual `remember()` pipeline. Each arm used fresh
graph, vector, SQL and cache storage. The embedding provider was the same
existing Ollama Qwen embedding service. Memory caching and automatic feedback
remained enabled; bulk ingestion used the existing workflow's
`self_improvement=False`, with two documents per batch and at most four
in-flight LLM calls.

| Measurement | vLLM | Exact | Fast |
|---|---:|---:|---:|
| Ingest and persist seven documents | 252.57 s | 144.83 s | **65.67 s** |
| Observed speed relative to vLLM | 1.00× | 1.74× | **3.85×** |
| Output tokens during ingestion | 15705 | 13572 | 12427 |
| Aggregate output tokens per ingestion wall second | 62.18 | 93.71 | **189.23** |
| Semantic entities | 63 | 49 | 42 |
| Relationships between semantic entities | 101 | 83 | 74 |

The last token rate combines concurrent calls and includes persistence time;
it is not a single-stream decode rate. Fast generated 20.9% fewer output tokens
than vLLM, so the **3.85× task gain includes less extracted output**. The
aggregate token-throughput gain was **3.04×**. All seven documents, source
chunks and text summaries were persisted in every arm.

Each run made 15 ingestion LLM calls. Ten questions then exercised planned
versus completed meetings, negation, item possession, scarab warnings,
prices, delivery time and unknown facts. They were answered once with the
arm's own model and again with the same vLLM reader. Captured request routes
verified that the common reader actually reached vLLM. All **60 requested
answers** matched the expected facts on manual review. The 165 HTTP calls
across the full runs returned 200 and ended naturally.

The graph audit exposed two distinct problems:

- **Exact introduced a conflicting owner.** It stored both the correct
  Falrinth ownership of the Orb of Golden Death and an incorrect claim that
  Senshock possessed it. The error was already present in the model's
  extraction response. Both readers selected Falrinth using the source
  chunks, while acknowledging conflicting evidence.
- **Fast omitted a financial detail.** The ring offers survived in Sequoia's
  and Zephyr's descriptions, but their **4000-gold deposit amount was absent
  from the semantic graph**. Both readers still returned the correct amount
  from Lucius Graeme's retained document chunk. The other two semantic
  graphs preserved the amount.

This distinction matters for using Cognee. Document-backed recall can give a
correct answer despite an omission or contradiction in the semantic graph.
A downstream consumer using only extracted relationships would receive
different information. Ten correct answers therefore establish success on
these retrieval cases, rather than equivalent graph completeness.

There was one full run per arm. The vLLM and exact ingestions overlapped on
independent servers and shared the embedding service; fast ran subsequently.
Each arm had a one-document smoke test first. Caches were retained, and
query timing included overlapping traffic to the common reader. These are
observed workflow timings, not a sustained-capacity estimate. The exact
runtime, corpus hashes, questions and targeted quality checks are in the
[Cognee comparison](experiments/hibana/cognee/README.md).

## Long context and persistent session replays

The initial suite used three identical campaign requests with **200038
rendered input tokens**, four independent sessions starting at about 24K
tokens and taking three turns each, and two synthetic graph extractions.
The sessions carried NPC/coding roles and a remembered code; they exercised
state reuse rather than autonomous agents or actual code execution.

| Completed work per hour | vLLM | Exact | Fast | Fast relative to vLLM |
|---|---:|---:|---:|---:|
| 200K campaign requests | 69.3 | 107.4 | 137.0 | 1.98× |
| Complete three-turn session replays | 179.7 | 298.7 | 384.5 | 2.14× |
| Synthetic graph extractions | 41.4 | 65.3 | 213.8 | 5.16× |

These are short-run extrapolations that include the first request's prefill
and exclude engine startup. Before/after vLLM counters matched exactly the
24 replay requests, 894391 input tokens and 11330 output tokens; no additional
generation was visible in that interval. This counter audit applies to the
initial replay, not the later Cognee run.

### Cold and warm campaign cost

| 200K request | vLLM TTFT and total | Exact TTFT and total | Fast TTFT and total |
|---|---:|---:|---:|
| First | 112.09 / 126.78 s | 72.92 / 81.56 s | 64.47 / 69.17 s |
| Repeat 1 | 1.86 / 14.74 s | 1.29 / 10.00 s | 1.04 / 5.17 s |
| Repeat 2 | 1.64 / 14.32 s | 0.45 / 9.03 s | 0.43 / 4.47 s |

Hibana reused **200000 cached tokens** on each repeat. One warm fast call
produced **307 output tokens in 5.17 seconds**, or **59.3 tokens per second**
including the wait for first output. The next repeat delivered 68.6 tokens
per wall second. Average output rate after the first chunk was about
74–76 tokens per second for the warm fast calls. These rates describe short
campaign-prep completions, not a fixed-length generation benchmark.

### Structured generation and quality

The synthetic graph task explicitly named 40 people and 40 places with 40
relationships. All engines returned the correct graph twice. It tested
controlled extraction, separately from the real Cognee pipeline above.

| Synthetic graph measurement | vLLM | Exact | Fast |
|---|---:|---:|---:|
| Output tokens per response | 4262 | 4262 | 3302 |
| First response total time | 88.19 s | 56.14 s | 17.31 s |
| Repeated response total time | 85.67 s | 54.13 s | 16.37 s |
| Output tokens per wall second across both | 49.02 | 77.29 | **196.08** |

Fast used fewer formatting tokens for the same graph content. Its **5.16×
task-throughput gain** therefore includes that saving; the measured
token-throughput gain was **4.00×**.

Each engine passed seven arithmetic/copy/schema/tool checks, three campaign
control-fact checks, twelve session-memory checks and two graph checks.
Those **24 checks did not grade campaign prose quality**. All fifteen marked
control-fact values were recovered, but fast's prep omitted Harshnag and his
100-gold bounty share in all three responses. Its cold answer also introduced
an Ironhand clan and a legendary forge/blacksmith absent from the supplied
archive. The other engines retained the Harshnag obligation.

Both Hibana profiles produced different cold and warm prep prose; their two
warm responses matched. vLLM's three responses matched. The fixture ended
partway into a chapter titled *Downtime in Mirabar*, so location differences
were excluded from the quality judgment. The verified unsupported additions
and omitted commitment are sufficient to show that successful retrieval of
marked facts did not ensure faithful long-context synthesis.

## Latency measurements

Time to first output excludes role-only SSE events and includes reasoning or
tool output. Tool parsing may buffer an entire call. The table gives empirical
p50 / p95 across requests; the sample counts are small and these tails are
descriptive rather than stable estimates.

| TTFT in seconds | vLLM | Exact | Fast |
|---|---:|---:|---:|
| Campaign, n=3 | 1.86 / 112.09 | 1.29 / 72.92 | 1.04 / 64.47 |
| All session turns, n=12 | 1.28 / 57.51 | 0.30 / 33.38 | 0.22 / 31.33 |
| Warm session turns, n=8 | 1.28 / 1.33 | 0.29 / 1.21 | 0.22 / 0.23 |
| Synthetic graphs, n=2 | 0.63 / 0.66 | 0.40 / 1.29 | 0.10 / 0.79 |

True per-token inter-token-latency percentiles were unavailable from the chat
streams because a speculative SSE event can deliver multiple tokens.
Captured event gaps are labeled chunk latency. Average time per output token
(TPOT) uses first/last output-chunk times and server token usage:

| Per-request average TPOT p50 / p95 in milliseconds | vLLM | Exact | Fast |
|---|---:|---:|---:|
| Campaign | 26.12 / 29.79 | 23.41 / 27.39 | 13.22 / 13.50 |
| All session turns | 66.91 / 455.22 | 28.60 / 229.07 | 16.51 / 206.32 |
| Warm session turns | 62.10 / 78.49 | 26.97 / 35.79 | 14.14 / 16.64 |
| Synthetic graphs | 19.94 / 20.53 | 12.60 / 12.86 | 4.93 / 5.00 |

These are percentiles of per-request averages, not of individual token
intervals. First-turn session generation can overlap other requests' prefill.

## Memory and recovery

| Initial replay memory measurement | vLLM | Exact | Fast |
|---|---:|---:|---:|
| Minimum sampled global MemAvailable | 11.16 GiB | 24.89 GiB | 34.51 GiB |
| Native CUDA allocation | Not recorded | 84.93 GiB | 73.93 GiB |
| Hibana resident weights | — | 68.89 GiB | 53.03 GiB |
| Hibana recurrent-state pool | — | 6.03 GiB / 56 slots | 10.34 GiB / 96 slots |

Memory was sampled every two seconds. Swap usage did not increase during the
measured intervals, although old swapped pages remained. The unequal KV
precision/capacity and state pools matter when interpreting the headroom.

Exact's native allocation stayed at 84.93 GiB as cached tokens increased from
201211 after the campaign calls to 299148 after the sessions. SSD use grew
from 4.158 to 7.667 GiB and ended at 7.989 GiB. Fixed pools explain the stable
allocation as session occupancy increased. This run did not establish a
latency curve against resident KV state or stress persistent sessions at
twice GPU cache capacity.

An idle exact process group was deliberately killed with SIGKILL. It was
healthy **69.7 seconds after restart**, reopening twenty stored prefixes.
Total observed outage was 109.3 seconds, including a launcher correction for
the Spark's missing ripgrep. The first post-crash campaign request restored
**200000 tokens**, passed its checks, and finished with **1.40 seconds TTFT
and 9.98 seconds total**. Native counters reported **529.6 ms** for the SSD
restore and zero restore/read/write errors. This validates one completed
session's recovery after a process crash; power-loss durability and prolonged
failure recovery need separate testing.

Both Hibana profiles rejected three malformed requests with HTTP 400 and
cancelled a disconnected stream while staying healthy. Automatic and required
tool calls, their JSON arguments and subsequent tool-result turns passed.
Initial exact startup took about 202 seconds while kernels compiled; later
exact restarts took about 70 seconds. The calibrated fast download took about
nine minutes plus verification. Profile startup receipts are retained with
the [operational evidence](experiments/hibana/evidence/).

## Deployment and evidence

The primary vLLM image, container start time and client configuration were
preserved. Hibana exact was restored after the Cognee comparison; fast's
validated configuration, weights and separate cache are retained. The shadow
API is loopback-only, with a transient user unit and no boot startup. Native
memory/cache/speculation counters are available through `/get_server_info`;
its metrics differ from vLLM's Prometheus histograms.

The measured gains justify further use in shadow deployments, especially for
Cognee retrieval that includes original documents. Promotion for canonical
campaign prep or graph-only reasoning requires broader tests of unsupported
claims and missing facts. The current evidence supports substantial speed
gains alongside workload-dependent changes in information quality.

Supporting material:

- [Initial replay measurements](experiments/hibana/results.json) and
  [runtime and rollback instructions](experiments/hibana/README.md).
- [Real Cognee measurements](experiments/hibana/cognee/results.json),
  [manual quality checks](experiments/hibana/cognee/quality-review.json), and
  [corpus hashes](experiments/hibana/cognee/corpus-manifest.json).
- [Experiment log](hibana-observations.md) and [current deployment](current-setup.md).
- Replay and Cognee runners under `experiments/hibana/`, with tests for timing,
  completed-task counting and publication redaction.

Published receipts contain metrics, hashes and targeted quality findings.
Full campaign prompts, answers, extracted graphs and test databases remain
in ignored local storage. No production dataset was a test write destination.
