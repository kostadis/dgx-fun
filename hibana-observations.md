# Hibana on one Spark beside pinned vLLM

## 2026-11-14: initial shadow replay

Hibana `exact` completed the campaign, persistent-session and structured-output
replays faster than the retained Qwen vLLM configuration, with all objective
checks passing. Its persisted 200K-token session also survived a deliberate
process crash. This supports keeping a shadow instance available for further
quality testing. It does not establish general answer parity or justify
replacing the primary service.

The `fast` profile also passed the objective checks and was faster still;
its results and the output-length confound are recorded below. Source, fixtures, commands, rollback and compact
results are in [the trial directory](experiments/hibana/README.md) and
[results.json](experiments/hibana/results.json).

### What actually ran

Two independent GB10 Sparks, one server per device. Spark1's retained
`qwen38-flash` container, image and flags were left unchanged. Spark2's
completed Clef training service was stopped and retained. Hibana was installed
in a private directory at commit `d474b74d5a892b2cd1cc8fc50cb17946e2a539bb`.
Both engines use the previously characterized Qwen3.8 Flash Next base revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`.

vLLM retains its FP8 hybrid trunk, NVFP4 experts, BF16 KV, MTP-2, APC, eight
sequence limit and roughly 600K-token KV pool. Hibana `exact` uses its FP8
trunk and KV, NVFP4 experts, MTP-4 with exact acceptance, four sequence limit,
300032-token KV pool, FP32 recurrent state and SSD session persistence.
These are deployable-configuration comparisons, not isolated engine ablations.

The client ran on spark2, reaching vLLM over LAN and Hibana over loopback.
Before/after vLLM counters increased by exactly **24 completed requests,
894391 prompt tokens and 11330 completion tokens**, matching the replay
receipts. No additional vLLM generation was visible in that interval.
No production cache was flushed; disk and OS caches were uncontrolled.

### Completed work per hour

These are short-run extrapolations, including cold starts within each workload
but excluding engine startup and the smoke suite. Output length is allowed
to differ for open-ended prep and session plans. The graph output has the
same 4262 completion tokens on both measured engines.

| Workload | Pinned vLLM | Hibana exact | Exact / vLLM |
|---|---:|---:|---:|
| Campaign requests/hour, 3 × 200038 input tokens | 69.3 | 107.4 | 1.55× |
| Completed three-turn sessions/hour, 4 concurrent sessions | 179.7 | 298.7 | 1.66× |
| Structured graph extractions/hour, 2 requests | 41.4 | 65.3 | 1.58× |
| Structured output tokens/second, including prefill | 49.0 | 77.3 | 1.58× |

Each engine passed **7/7 smoke checks, 3/3 campaign retrieval/prep-shape checks,
12/12 session-memory checks, and 2/2 exact graph-content checks**. Sessions
use campaign histories with NPC/coding roles and a remembered code; they are
state-reuse proxies, not complete autonomous agents. Graph extraction is
Cognee-shaped, not an end-to-end Cognee pipeline.

### Cold and warm costs

| 200K campaign request | vLLM TTFT / total seconds | Hibana exact TTFT / total seconds |
|---|---:|---:|
| First | 112.09 / 126.78 | 72.92 / 81.56 |
| Identical repeat 1 | 1.86 / 14.74 | 1.29 / 10.00 |
| Identical repeat 2 | 1.64 / 14.32 | 0.45 / 9.03 |

Hibana reports 200000 cached input tokens on both repeats. vLLM's aggregate
cache-hit counter increased by 576000 tokens across the full suite; this
vLLM build did not expose per-request cached tokens in the streaming usage.

| Requests | vLLM TTFT p50 / p95, seconds | Exact TTFT p50 / p95, seconds |
|---|---:|---:|
| Campaign, n=3 | 1.86 / 112.09 | 1.29 / 72.92 |
| Four sessions, all 12 turns | 1.28 / 57.51 | 0.30 / 33.38 |
| Session warm turns only, n=8 | 1.28 / 1.33 | 0.29 / 1.21 |
| Structured graphs, n=2 | 0.63 / 0.66 | 0.40 / 1.29 |

The empirical p95 is the maximum when n is 2, 3 or 12; this is too little
data for a stable tail-latency estimate. The first strict-schema compilation
and first-use kernels can also appear in smoke TTFT.

True per-token ITL p50/p95 is **not available** from these OpenAI chat streams:
a speculative SSE chunk can contain several tokens, and tool parsers can
buffer calls. Raw chunk-gap percentiles are retained, with that label.
Average TPOT, calculated per request from the first/last output chunks and
server token usage, is available:

| Requests | vLLM TPOT p50 / p95, milliseconds | Exact TPOT p50 / p95, milliseconds |
|---|---:|---:|
| Campaign | 26.12 / 29.79 | 23.41 / 27.39 |
| All session turns | 66.91 / 455.22 | 28.60 / 229.07 |
| Warm session turns | 62.10 / 78.49 | 26.97 / 35.79 |
| Structured graphs | 19.94 / 20.53 | 12.60 / 12.86 |

These are percentiles across per-request averages, not percentiles of
individual token intervals. First-turn session TPOT includes decoding that
overlaps other sessions' prefill.

### Memory and state persistence

Minimum sampled global MemAvailable during the benchmark was **11.16 GiB
on spark1** and **24.89 GiB on spark2**. Samples were taken every two seconds.
Neither run increased swap usage during its measured replay interval;
preexisting swapped pages remained. CUDA allocation is not fully represented
in CPU RSS on GB10, so the native memory ledger matters.

Hibana's GPU allocation stayed at **84.93 GiB** throughout the replay:
68.89 GiB weights, 5.15 GiB KV pool, 6.03 GiB recurrent-state pool,
3.62 GiB proposer/verification and smaller workspaces/graphs. As sessions
accumulated, cached tokens increased from **201211 after campaign** to
**299148 after sessions**; SSD use increased from **4.158 to 7.667 GiB**.
Fixed pools explain why resident allocation did not grow with cache occupancy.
The suite ended with 7.989 GiB persisted. It did not exercise prolonged
multi-session pressure at twice KV capacity or establish a causal latency
curve against resident KV occupancy.

### Recovery and operational behavior

The shadow rejected empty messages, negative temperature and zero output
budget with HTTP 400. Closing an active client stream incremented cancelled
requests from zero to one, and health remained good. Automatic and required
function calls, parsed JSON arguments and subsequent tool-result turns passed
on both engines. Generated tool calls were inert local fixtures.

The idle shadow process group was then killed with SIGKILL. Restart read **20
stored prefixes totaling about 8 GiB** and was healthy **69.7 seconds after
unit start**. Observed total outage was **109.3 seconds**, including a launcher
failure because ripgrep was absent on the Spark; the launcher was corrected
to use the installed grep before retrying. Initial installation/startup also
compiled native kernels; first readiness took about **202 seconds**.

The first post-crash campaign request reused **200000 tokens** and passed:
**1.40 s TTFT, 9.98 s total**. Native counters report one SSD restore,
**529.6 ms restore time**, 3.922 GiB read, zero restore failures and zero SSD
read/write errors. This is one successful crash trial, not a durability soak.

No system packages, boot services, client endpoints or primary-container
settings were changed. The private environment uses the upstream uv lock.
The model, CUDA kernels, transient systemd unit, guard thread, disk cache and
separate OpenAI model ID are additional operational components to maintain.
`/get_server_info` provides cache/SSD/speculation counters and a memory ledger;
the small `/metrics` surface differs from vLLM's Prometheus histograms.

### Quality and promotion limit

The graph fixtures matched all 80 entities and 40 relationships exactly.
Arithmetic, copy, schema and tool-call contracts passed. This supports API
smoke parity for these cases, not broad OpenAI compatibility.

All campaign control facts matched, but generated prep prose differed between
engines. Hibana `exact` also generated a different cold response from its
two identical warm responses; vLLM's three responses were identical. One cold
Hibana response placed the party in Mirabar while the warm responses described
the aftermath at Xantharl's Keep. These details have not been adjudicated
against the source. `exact` refers to speculative acceptance and weight
options; it does not guarantee bitwise stability across prefill/cache paths.
Semantic prep quality and real coding-agent outcomes therefore remain open.

The paper's headline figures describe different concurrency and benchmark
conditions. Its own accuracy table also reports changes under lossy options.
These local measurements should be assessed on their own. See
[the paper](https://arxiv.org/html/2611.10469v1) and
[pinned serving-profile documentation](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/operations.md).

### Fast-profile follow-up

The same frozen fixture and replay ran with calibrated checkpoint
`yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3-qad` at
`5c4c62a62853caebea8cedda7fbed952efa1effe`. All 328 downloaded files
completed, and the upstream fetch checked tensor SHA256s before publishing
the checkpoint directory. Download took about nine minutes, plus verification.
The launcher requires this pinned directory; the measured run had
`w3_decode=true`, no W3 warning and a non-null checkpoint identity.

The profile uses three-bit experts/trunk copies, expert sharing (four experts
per token, 12 per request / 64 per batch budget), NVFP4 KV, adaptive MTP up to
eight drafts, and relaxed greedy acceptance (top three within 2.0 logits).
Recurrent state remains FP32 in this `fast` configuration; INT8 recurrent
state was not tested. It is a lossy serving profile. The run recorded 146
relaxed accepts and about 8.12 emitted tokens per server speculative round,
versus 4.97 for `exact` across the mixed-concurrency suite.

| Workload | vLLM | Hibana exact | Hibana fast | Fast / vLLM |
|---|---:|---:|---:|---:|
| Campaign requests/hour | 69.3 | 107.4 | 137.0 | 1.98× |
| Completed three-turn sessions/hour | 179.7 | 298.7 | 384.5 | 2.14× |
| Graph extractions/hour | 41.4 | 65.3 | 213.8 | 5.16× |
| Graph output tokens/second, including prefill | 49.0 | 77.3 | 196.1 | 4.00× |

Fast passed **24/24** checks: 7 smoke, 3 campaign, 12 session turns and 2
graphs. Both graph responses contained all 80 entities and 40 relationships
exactly. However, fast emitted **3302 tokens per graph**, compared with
4262 for vLLM and exact: equivalent content with fewer formatting tokens.
The **5.16× task throughput must not be presented as a 5.16× decode speedup**;
the measured token-wall-throughput gain was 4.00×. No fixed-length decode
microbenchmark was run. Campaign and session prose lengths also differ.

| Fast 200K campaign request | TTFT / total seconds | Cached tokens |
|---|---:|---:|
| First | 64.47 / 69.17 | 0 |
| Repeat 1 | 1.04 / 5.17 | 200000 |
| Repeat 2 | 0.43 / 4.47 | 200000 |

| Fast requests | TTFT p50 / p95, seconds | Average TPOT p50 / p95, milliseconds |
|---|---:|---:|
| Campaign, n=3 | 1.04 / 64.47 | 13.22 / 13.50 |
| All session turns, n=12 | 0.22 / 31.33 | 16.51 / 206.32 |
| Warm session turns, n=8 | 0.22 / 0.23 | 14.14 / 16.64 |
| Structured graphs, n=2 | 0.10 / 0.79 | 4.93 / 5.00 |

The same small-sample and chunk-timing qualifications apply. Fast's GPU
allocation was **73.93 GiB**, with 53.03 GiB weights, 3.65 GiB KV and
10.34 GiB recurrent-state pool. Its planner allocated more recurrent slots
than exact (96 versus 56), so the configurations differ beyond precision.
Minimum sampled global MemAvailable was **34.51 GiB**, with no increase in
swap during the replay. Fast also rejected all three malformed requests,
cancelled a disconnected stream, and stayed healthy. Its separate SSD cache
ended at 6.442 GiB and was retained. Initial fast readiness took 130.1 seconds
after unit start. Crash recovery was tested on exact, not fast.

Fast also produced two different campaign responses across its three
identical calls; its two warm responses matched. All control facts passed,
but semantic prose quality remains unadjudicated. The final running shadow
is therefore **exact**, with fast's validated launch configuration and results
retained for explicit further comparison. Spark1 stays on pinned vLLM.

### Local validation

The replay's three focused unit tests passed: role-only SSE events do not
set TTFT, truncated/missing-DONE responses cannot count as completed tasks,
and concurrent-session throughput uses wall time rather than summed request
times. Python compilation and launcher shell syntax checks passed. Live
replay evidence supplies the runtime validation; the full repository and
upstream GPU test suites were not run.

### Follow-up: manual review of fast-profile quality

Reviewing the saved answers against the actual supplied archive gives a mixed
assessment. Fast preserved exact graph content, control facts and the tested
tool contracts, but its campaign prep was less anchored to the source than
the vLLM and exact responses. The 24/24 objective score must not be read as
24 human-reviewed, high-quality answers: the prep scorer checked the five
control facts and three nonempty strings, not campaign consistency.

In the cold prep response, fast introduced a powerful dwarven clan called
"Ironhand" and a legendary forge/blacksmith. Searches of the exact supplied
200K-token user message found no Ironhand, blacksmith or legendary-forge
reference. Its contingencies also floated a curse on the Weevil and his
possible innocence without grounding those ideas in the supplied events.
These are speculative additions, not retrieved campaign facts.

All three fast prep responses omitted Harshnag by name and his explicit
100-gold bounty share, despite that material appearing near the archive's
end. The vLLM and exact responses retained this obligation and made his
proposed oracle journey central. Fast's warm prep instead emphasized generic
Mirabar downtime, shopping and faction escape contingencies. This is a
material difference in usefulness for source-grounded prep.

The party's exact location is not a clean error test here: the fixture ends
partway into a document titled "Downtime in Mirabar", immediately after the
Xantharl's Keep account. The differing location claims therefore need a
better-bounded chronology fixture before being scored as errors.

The strongest quality result remains structured extraction: both fast runs
matched all 80 entities and 40 relationships exactly. All five control facts
were correct on all three long calls (15/15 values), all twelve memory-code
turn checks passed, and automatic/required tool calls plus result round trips
passed. The session fixtures did not test actual code generation, repair,
execution or complex NPC voice. Fast is promising for constrained extraction;
this run does not qualify it for campaign prep or establish quality parity.

### Actual Cognee follow-up

The user's existing Cognee runtime subsequently ingested seven complete ToEE
NPC dossiers (7,039 words) into separate graph/vector/SQL/cache roots. The real
`remember()` pipeline completed in **252.57 s on pinned vLLM, 144.83 s on
Hibana exact, and 65.67 s on Hibana fast**. Fast's observed ingestion gain was
**3.85×**; output tokens fell from 15705 to 12427, so its aggregate
token-wall-throughput gain was **3.04×**, including concurrent calls and
persistence rather than measuring one decode stream.

Ten source-grounded questions were answered with each arm's own model and
again with the same vLLM reader. All requested answers matched the expected
facts, including negation and unknowns. Captured routes confirmed the shared
reader actually used vLLM. No HTTP errors occurred in the full runs.

The persisted semantic graphs were not equivalent. Exact introduced a false
claim that Senshock possessed the Orb of Golden Death; its correct Falrinth
edge and source chunks allowed both readers to resolve the conflict. Fast
kept the Orb holder correct but omitted the **4,000-gold deposit amount** from
its semantic graph, even though its source chunk still enabled a correct
answer. The ring offers remained in the character descriptions.

This is stronger evidence for fast's usefulness in actual Cognee document-backed
recall than the earlier synthetic graph fixture. It also demonstrates that
successful answers can conceal extraction omissions or contradictions. The
test did not measure exhaustive graph precision/recall or qualify full-campaign
ingestion. One run per arm, different output sizes and overlapping use of the
embedding/common-reader services limit timing conclusions.

The existing Cognee data/configuration and primary vLLM service were preserved.
Exact was restored after testing. See [the real Cognee comparison](experiments/hibana/cognee/README.md)
for timing, quality review, configuration and complete evidence links.
