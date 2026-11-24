# Cognee on the selected 411K FP8 Inkling configuration

Completed November 22, 2026 (Pacific). The current native Inkling deployment
was **slower on the factual benchmark** than the historical Qwen and native
64K BF16 deployments. It retained **44/47 complete correct answers**, with
all 44 fully grounded. Its six-search agent recovered the actual Chapter
4-to-5 handoff that the previous thinking-blocked native run had missed, but
retained a separate belief-attribution error.
[Audited metrics](results-20261122.json) · [Protocol](README.md) ·
[Combined comparison](../../../phandelver-cognee-agent-results.md).

The later [stock Cognee replay](../stock-cognee/RESULTS.md) reused this serving
configuration. Stock's 10.87 s median and 46/47 complete compare with this
adaptive run's 6.03 s and 44/47; fully grounded complete remained 44/47 in both.
Its better-supported module handoff took 263.71 s versus 180.03 s.

## Factual questions

| Configuration | Query p50 | Query p95 | Sum of 48 queries | Complete correct |
|---|---:|---:|---:|---:|
| Qwen, pinned vLLM, historical | 4.68 s | 6.16 s | 226.79 s (3 min 47 s) | **45/47** |
| Native Inkling, 64K BF16 KV, thinking blocked, historical | **4.51 s** | **6.42 s** | **222.25 s (3 min 42 s)** | 44/47 |
| **Native Inkling, 411K FP8 KV, thinking blocked** | **6.03 s** | **8.41 s** | **300.32 s (5 min)** | **44/47** |

The FP8 configuration's median latency was **28.9% higher than Qwen's** and
**33.9% higher than the prior native thinking-blocked run**. These are
observations from separate single runs on shared hardware, not repeated-run
confidence intervals or a measurement of FP8 alone: context limit and cache
allocation changed along with KV precision.

The subsequent [182K BF16 fixed-cache test](../bf16-cognee/RESULTS.md) measured
6.10 s median and 301.10 s summed query time, essentially the same as this
FP8 result. Its 48 answers and supplied contexts exactly matched the old 64K
BF16 run. The higher latency also appears in the larger BF16 configuration,
so these observations do not identify KV precision alone as the cause.

All **48 actual reader request bodies matched the 64K BF16 control except for
the cache salt**, including retrieval context, temperature, seed, output cap
and thinking controls. Input was **204,946 tokens** in each native run.
Output was **3,069 tokens**, versus BF16's **3,071**; both generated zero
reasoning tokens. Every FP8 response stopped naturally, with no truncations.

Captured upstream model-request time increased from **157.44 to 237.53 s**,
while the remainder of summed query time was **64.81 versus 62.79 s**.
Thus, the extra elapsed time was inside the model requests, with almost
unchanged token counts. Those timings include server prompt processing,
generation and request overhead; they do not isolate GPU decode time or
identify which serving change caused the difference.

Source review retained the known flood retrieval gap (q04), reversed Ruxithid
belief wording (q20), and abstention on the missing clairvoyance evidence
(q22). q37 remains the predeclared ambiguous exclusion; q38 keeps the original
trigger-and-round rubric. All six item questions passed.

Fully grounded complete answers improved **43/47 → 44/47** relative to the
prior native off run: the new q30 answer did not add the unsupported causal
link between thought reading and the severed head demanding treats. This
did not add another complete answer. Review reused 23 prior labels only for
exact answer and reader-message matches, reviewed 24 novel answers against
the frozen gold and supplied context, and retained q37's predeclared exclusion.
Review was by Codex without blinded human adjudication.

## Agentic module question

| Configuration | Client time | Searches / calls | Raw passages | Model tokens |
|---|---:|---:|---:|---:|
| Qwen, historical | 143.41 s | 5 / 10 | See original receipt | 61,432 |
| Native 64K BF16 KV, thinking blocked, historical | 164.69 s | 5 / 11 | 17 | 57,729 |
| **Native 411K FP8 KV, thinking blocked** | **180.03 s** | **6 / 12** | **25** | **82,682** |

The new client time was **3 min**, including worker startup; the research loop
took **175.72 s**. It used **78,419 input and 4,263 output tokens**, zero
reasoning, zero protocol corrections and two unused searches. The agent chose
each next query after reading the preceding result. Search 4 recovered
Chapter 5's first paragraph, and search 6 recovered Chapter 4's actual ending.

The final answer correctly covered the mine-profit award, escorting Nundro
home, returning to Phandalin, the new goblin crime spree and the investigation
that begins Chapter 5. It also described the level progression and later
Obelisk-fragment investigation. It did not invent a Forge/Obelisk causal link
or a physical note on a corpse. This closes the main retrieval gap left by
the previous native thinking-blocked answer.

It was still **not a fully grounded complete answer**. The final wording
assigned the belief that the mind flayers are gods to the fanatics; source
S014 assigns that belief to Ruxithid. Its claim that the transition is
explicitly narrative rather than mechanical is an interpretation, while the
sources also give levels and rewards. The measured answer was preserved
unchanged. This is more specific handoff coverage from a different search
trajectory, rather than evidence that FP8 inherently improves reasoning.

Client time was 25.5% higher than historical Qwen and 9.3% higher than the old
native off run, with more searches, passages and input tokens. These are not
quality-matched decoder speed comparisons. Qwen's and native Inkling's
historical source errors remain in the combined report.

## Provenance and operations

The experiment reused the corrected adaptive index: **759 chunks, 4,335 nodes,
19,700 edges**, with the original Qwen ingestion and Ollama
`qwen3-embedding:0.6b` embeddings, 1,024 dimensions and truncation disabled.
No ingestion or index updates were performed. Both phases used fresh clones,
fresh sessions and disabled automatic feedback. All **21,151 protected
historical/input files** passed their hash check.

The actual-request audit verified **329 factual and 40 agent contextual chunk
appearances**, all six agent observation/source unions, and the matching
initial agent reader request. Reader/action caps stayed at 4,096/2,800 tokens.
Warmup and storage cloning are excluded from the factual query sum; agent
client time includes worker startup. The standard prompts do not fill 411K:
the [410,496-token capacity check](../context/RESULTS.md) remains separate.

The same native container, image, command and start time persisted throughout.
Clef identity, image, command, start time and readiness were unchanged. The
12 GiB host-headroom guard stayed active and did not trip; final sampled host
headroom was **13.49 GiB**. At **13:11 Pacific**, native Inkling still advertised
411,520 tokens and Qwen model discovery passed. The large FP8 configuration
remains selected. This run did not switch services or restore DSpark.

Full requests, answers, source passages, reviews and code snapshots remain
under ignored `experiments/adaptive-chunking/raw/pabtso/cognee-20261120/inkling-native-fp8-411k-20261122/`.
The public metrics omit that text. AtlasKV, the original Cognee comparisons,
and the earlier synthetic decoder benchmarks retain separate receipts.
