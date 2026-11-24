# Stock versus adaptive Cognee chunking with Inkling

Completed November 22, 2026 (Pacific). **Adaptive was faster; stock recovered
two more complete factual answers and gave the better-supported module
transition answer in this run.** The tradeoff remains after fixing the reader
and agent to native Inkling with thinking blocked.
[Audited metrics](results-20261122.json) · [Protocol](README.md) ·
[Adaptive FP8 baseline](../fp8-cognee/RESULTS.md) ·
[Combined report](../../../phandelver-cognee-agent-results.md).

Both runs used the same pinned Inkling checkpoint, image and launch arguments:
411,520-token context, FP8 KV, a fixed 13 GiB cache, one active request and no
speculation. The two saved indexes cover the same full Phandelver module,
with the same Qwen ingestion model and Ollama embeddings. Stock has 78 chunks;
corrected adaptive has 759. Neither index was reingested for this test.

## Original 48 factual questions

| Measure | Stock Cognee | Corrected adaptive |
|---|---:|---:|
| Chunks | 78 | 759 |
| Query p50 | **10.87 s** | **6.03 s** |
| Query p95 | 14.72 s | 8.41 s |
| Sum of 48 queries | 514.61 s (8 min 35 s) | 300.32 s (5 min) |
| Complete correct answers | **46/47** | 44/47 |
| Fully grounded complete answers | **44/47** | **44/47** |
| Input tokens | 844,840 | 204,946 |
| Output tokens | 3,355 | 3,069 |
| Captured upstream model-request time | 466.69 s | 237.53 s |

Adaptive was **1.80× faster at the median**, with **41.6% less summed query
time**. Stock supplied **4.12× as many input tokens**. The number of verified
contextual chunk appearances was similar (321 stock, 329 adaptive), but stock
passages were much larger. That is consistent with a prompt-volume cost;
these captures do not isolate GPU prefill or decode time. The residual query
time outside captured model requests was 47.93 s stock versus 62.79 s adaptive.
Cached-token counts were not reported, so no cache-hit rate is inferred.

Stock correctly answered q20 (Ruxithid's mistaken belief) and q22 (his
clairvoyance), both missed by the adaptive FP8 run. The flood timing evidence
was still missing in q04; stock correctly identified the origin and outflow
and explicitly declined to supply an exact timing, leaving a partial answer.
q37 remains the predeclared ambiguous exclusion. q38 retains the original
trigger-and-round rubric. All six item questions passed.

Stock also added unsupported details to two complete answers. In q29 it
assigned Voalsh's hovering above others to her wings; the supplied passage
assigns that hovering to a magical floating disk and describes unfurling her
wings for intimidation. In q30 it made treat demands another means of ensuring
Chishinix's loyalty; the source assigns that role to thought reading and lists
treat demands separately. Thus, the two additional complete answers did **not**
increase the count of fully grounded complete answers. All 47 adjudicable
answers were reviewed against the frozen gold and actual supplied context by
Codex, without blinded human adjudication.

All 48 responses stopped naturally, with zero reasoning tokens and no
truncations. Reader controls matched the adaptive FP8 run, including 4,096-token
output caps, seed, temperature and thinking-token ban; retrieved context was
allowed to differ. All 48 actual stock reader messages also matched the original
stock Qwen benchmark's messages exactly.

## Agentic module question

The agent independently answered how the first part, culminating in Wave Echo
Cave, feeds into the second part. It read each result before choosing another
query, under the same original question, policy, `top_k=15` and eight-search
budget. It received no prior answer or reviewer-directed query.

| Measure | Stock Cognee | Corrected adaptive |
|---|---:|---:|
| Client elapsed | **263.71 s (4 min 24 s)** | **180.03 s (3 min)** |
| Research loop | 259.24 s | 175.72 s |
| Searches / model calls | 6 / 13 | 6 / 12 |
| Distinct raw passages | 18 | 25 |
| Protocol corrections | 1 | 0 |
| Input / output tokens | 290,123 / 4,324 | 78,419 / 4,263 |
| Total model tokens | 294,447 | 82,682 |
| Actual Chapter 5 opening first retrieved | Search 2 | Search 4 |
| Actual Chapter 4 ending first retrieved | Search 3 | Search 6 |
| Core transition complete | Yes | Yes |
| Substantive answer supported by raw-source union | Yes, with wording caveat below | Remaining belief-attribution error |

Stock took **46.5% longer** and consumed **3.56× as many total model tokens**,
although both agents chose six searches. It recovered the main evidence earlier,
then continued checking the goblins, Obelisk shards and the later Chapter 5-to-6
transition. One attempted search repeated an earlier query and was rejected;
the agent chose another query after protocol feedback. Both runs left two
searches unused. These are different search trajectories, not an equal-work
decoder benchmark.

The stock final answer covered the optional Chapter 4 stopping point, level
progression, the 10% mine-profit award, escorting Nundro home, the return to
Phandalin, its new goblin crime spree, and the investigation leading to
Ruxithid and the Obelisk plot. It correctly retained first-half psionic-goblin
foreshadowing and avoided a Forge/Obelisk causal invention, a physical note on
the corpse, and the adaptive answer's mistaken belief attribution.

Its scouting sentence was imprecise: it said the group left no Obelisk shards;
the source says they found none and left their companion's corpse. This supports
the absence of shards, not evidence of shard removal. The count of three fanatics
is supported by retrieved S001, cited later in the same paragraph, rather than
the immediately adjacent S008/S011 citations. Under the stated review scope,
substantive claims were supported by the accumulated raw-source union; this
was not a word-for-word or citation-placement perfection test. The measured
answer was retained unchanged.

This single stock run produced a more consistently supported transition
explanation at greater cost. It does not establish a general quality advantage
for stock, or a quality-matched speed comparison between the two agents.

## Provenance and operations

Stock reused **1,410 nodes, 5,287 edges and 78 document chunks**; adaptive reused
4,335 nodes, 19,700 edges and 759 chunks. Graphs and summaries were originally
created by Qwen. Both manifests have matching frozen source/input hashes and
embedding model/digest: Ollama `qwen3-embedding:0.6b`, 1,024 dimensions, truncation
disabled. Changing the saved index changes its learned graph and summaries as
well as passage granularity; this is an index-level comparison.

Both new phases used fresh storage clones and sessions, automatic feedback
disabled and distinct cache salts. Four synthetic warmups were excluded from
factual timing. The audit verified all **321 factual and 33 agent contextual
chunk appearances**, seven chronological agent observation/source-union checks,
actual generation controls and all **13,489 protected historical/input files**.
The agent's thirteenth call includes the one protocol correction. All 61
measured responses stopped naturally with zero reasoning tokens.

The largest factual and agent inputs were **29,146 and 46,516 tokens**. These
workloads do not exercise the 411K capacity limit. The separate large-context
and AtlasKV experiments retain their own results.

Inkling and Clef kept their container identities, images, commands and start
times throughout this replay. Their endpoints remained ready, the 12 GiB host
headroom guard stayed active without tripping, and final headroom was
**14.81 GiB**. At **14:17 Pacific**, Inkling still advertised 411,520 tokens and
Qwen model discovery also passed. The selected native FP8 service remains
running; Clef was not restarted.

The adaptive baseline ran earlier, before the temporary BF16 experiment and
FP8 restart. It matches the serving configuration, but not process uptime or
all shared-device activity. These are single runs without repeated-run
uncertainty estimates. Raw requests, source text, answers, reviews and receipts
remain in the ignored experiment directory documented in the protocol. The
public export contains metrics and review flags, not full passages or answers.
