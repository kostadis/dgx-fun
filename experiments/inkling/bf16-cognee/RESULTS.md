# Native 182K BF16 Inkling: Cognee and capacity results

Completed November 22, 2026 (Pacific). **The larger BF16 window works alongside
Clef, but it did not recover the earlier 64K BF16 query latency.** Its factual
timings were essentially the same as the selected 411K FP8 configuration.
All 48 answer texts and supplied reader messages matched the old 64K BF16 off
run exactly, preserving its **44/47 complete and 43/47 fully grounded** scores.
[Audited metrics](results-20261122.json) · [Protocol](README.md) ·
[Combined comparison](../../../phandelver-cognee-agent-results.md).

## Factual questions

| Configuration | Query p50 | Query p95 | Sum of 48 queries | Complete correct |
|---|---:|---:|---:|---:|
| Qwen, pinned vLLM, historical | 4.68 s | 6.16 s | 226.79 s | **45/47** |
| Native 64K BF16 KV, thinking blocked, historical | **4.51 s** | **6.42 s** | **222.25 s** | 44/47 |
| Native 411K FP8 KV, thinking blocked | 6.03 s | 8.41 s | 300.32 s | 44/47 |
| **Native 182K BF16 KV, thinking blocked** | **6.10 s** | **8.71 s** | **301.10 s** | **44/47** |

The new BF16 median was **1.1% higher than FP8's**, with just **0.3% more
summed query time**: effectively similar performance in these single runs.
It was 30.3% higher than historical Qwen and 35.4% higher than the old 64K
BF16 median. Configurations ran at different times on shared hardware, so
these observations do not provide repeated-run confidence intervals.

All 48 new BF16 request bodies matched the old 64K BF16 control except for
the cache salt; the actual answer text also matched exactly. Both native BF16
runs counted **204,946 input and 3,071 output tokens**. New upstream request
time was **239.21 s**, versus **157.44 s** in the old BF16 run and **237.53 s**
in FP8. Every new factual response finished naturally with zero reasoning.

The known flood retrieval gap, reversed Ruxithid belief wording and missing
clairvoyance evidence remain. The q30 unsupported causal qualification also
remains, giving BF16 **43/47 fully grounded complete answers**, versus FP8's
44/47. q37 stays excluded as predeclared, and q38 retains its original scope.
Review labels are reused only after exact text/context matches, preserving
the previous strict rubric rather than changing scores after seeing timings.

This run shows that the higher latency also occurs with BF16 in the larger
fixed-cache setup. It does not support attributing the earlier FP8 timing
difference to KV precision alone. The old 64K service used utilization-derived
and extensible cache allocation; both larger configurations use a fixed 13 GiB
budget and the ordinary allocator. The effect of allocator, budget, nominal
context and host activity has not been isolated.

## Agentic module question

| Configuration | Client time | Searches / model calls | Raw passages | Model tokens |
|---|---:|---:|---:|---:|
| Qwen, historical | 143.41 s | 5 / 10 | See original receipt | 61,432 |
| Native 64K BF16 off, historical | 164.69 s | 5 / 11 | 17 | 57,729 |
| Native 411K FP8 off | 180.03 s | 6 / 12 | 25 | 82,682 |
| **Native 182K BF16 off** | **170.57 s** | **5 / 11** | **17** | **57,729** |

The new BF16 agent took **2 min 51 s including worker startup**, or **166.23 s**
for research. All 11 request bodies, model answer/decision texts, final answer
and source-label union matched the 64K BF16 off run exactly, apart from cache
salts. Its same-work client time was 3.6% higher, a small difference in one
observation. It used one protocol correction and left three searches unused.

The answer supports the broad level-five return to Phandalin and goblin-crime
bridge, but misses the actual mine-profit/Nundro-homecoming ending and Chapter
5's first paragraph. It incorrectly denies first-half Obelisk setup despite
retrieving the cave goblin scouting for shards, and concentrates its specific
transition detail on Chapter 5-to-6. After feedback rejected a finish with a
declared direct Wave Echo Cave-to-Chapter-6 gap, it cleared the gap and rewrote
the answer without searching again. The output remains unchanged.

The FP8 run recovered the exact Chapter 4 ending and Chapter 5 opening after
one more search, with its own belief-attribution error. Its modestly longer
elapsed time is not a quality-matched speed comparison with the BF16 answer.

## Near-limit BF16 validation

After both measured Cognee phases, the same BF16 process accepted **181,116
input tokens** against its **182,144-token context limit**. It correctly
returned all three exact access codes placed near the beginning, middle and
end of a synthetic archive, with **38 output tokens and zero reasoning**.
The tokenizer and actual prompt-usage counts matched. Total request time was
**103.76 s (1 min 44 s)**. This is one capacity/retrieval smoke check, not a
long-context accuracy benchmark or an equal-length speed comparison with the
410,496-token FP8 test.

The limit is the automatic fit into the **chosen 13 GiB BF16 KV budget** while
keeping Clef loaded and a 12 GiB host-memory floor. It is not Inkling's model
maximum or an absolute BF16 maximum alongside Clef. More available memory
could support a different budget; BF16 without Clef was not measured.

## Provenance and restoration

The saved corrected index stayed at **759 chunks, 4,335 nodes and 19,700 edges**.
Qwen ingestion, Ollama `qwen3-embedding:0.6b`, the frozen 48 questions and agent
policy were unchanged. Embeddings retained 1,024 dimensions and disabled
truncation. Fresh index clones/sessions and disabled feedback prevent prior
conversation carryover. All **21,469 protected historical/input files** passed
their hash check. Request audits verified 329 factual and 20 agent contextual
chunk appearances, plus all six agent decision/source unions including retry.

The same BF16 process stayed up through warmups, factual queries, agent and
near-limit validation. Clef's container identity, image, command, start time
and readiness stayed unchanged throughout the switch and tests. The memory
guard did not trip; minimum sampled headroom since BF16 startup was **14.55 GiB**.
Cold starts, warmups, capacity validation and restoration are outside the
factual query and agent client timings.

**The selected 411K FP8 service was restored afterward**, using its original
container, image and command. Its restarted process passed model discovery
and arithmetic with zero reasoning. At **13:45 Pacific**, FP8 and Clef were
healthy, Qwen model discovery passed, and host headroom was **15.65 GiB**.
The BF16 container is stopped and retained. The full benchmark and restoration
receipts are in ignored
`experiments/adaptive-chunking/raw/pabtso/cognee-20261120/inkling-native-bf16-182k-20261122/`;
remote switch/guard receipts are at `spark2:~/inkling-bf16-cognee-20261122/`.
Public metrics omit source passages, prompts and full answers. AtlasKV remains
separate.
