# Larger-context BF16 Inkling benchmark

**Completed:** factual p50/p95 **6.10/8.71 s**, **44/47 complete and 43/47
grounded**, and **170.57 s** for the five-search agent. A **181,116-token
extraction passed** in 103.76 s. The selected 411K FP8 service was restored
and Clef stayed unchanged. [Results](RESULTS.md) · [Audited metrics](results-20261122.json).

This follow-up tests the retained **182,144-token native BF16 KV configuration**
alongside Clef. It repeats the same 48 factual questions and independent
Wave Echo Cave transition agent used in the 64K BF16 and 411K FP8 comparisons,
then checks near-limit BF16 extraction. The selected 411K FP8 container is
restored afterward. AtlasKV remains a separate experiment.

## Configuration and controls

The BF16 and FP8 containers use the same checkpoint, image, 13 GiB fixed KV
budget, ordinary cache allocator, one active request, 4,096-token prefill
batch, eager execution, prefix caching and disabled speculation. Their launch
commands differ only in KV dtype (`auto`, resolving to BF16, versus `fp8`).
Automatic context fitting consequently gives different limits: 182,144 and
411,520 tokens. The larger BF16 comparison with old 64K BF16 also changes
context and cache planning/allocation. These are deployment measurements,
not a claim that only one low-level runtime variable differs in every pair.

The shared-Spark budget reserves a 12 GiB host-memory floor for Clef and
transient activity. The 182,144-token limit is an automatic fit into the chosen
13 GiB KV budget, not the model's native maximum or an absolute hardware limit.
Clef stays running with its identity, image, command and start time checked.

Both Cognee phases reuse fresh clones of the corrected adaptive index:
759 chunks, 4,335 nodes and 19,700 edges, with Qwen ingestion and Ollama
`qwen3-embedding:0.6b` embeddings unchanged. There is no reingestion. Each
question/search uses a new session and automatic feedback is disabled.
All historical/input files are protected by hashes.

Questions retain their original system prompt, `top_k=15`, 120-word instruction,
temperature zero, seed 604 and a 4,096-token reader cap. Native
`bad_words:["<|content_thinking|>"]` blocks generated thinking, with requested
effort `none`. Separate cache salts isolate phases and past runs. The same
four unmeasured arithmetic/JSON/negation/tool warmups precede factual queries.

The agent retains the original question, policy, schema-guided action format,
`top_k=15`, eight-search maximum, 2,800-token action cap and 4,096-token reader
cap. It observes each response before choosing a new query, without previous
answers or reviewer-directed searches. Independent search work and answer
quality must accompany its elapsed time.

## Validation and restoration

`operate.py` temporarily stops the selected FP8 container and starts the
retained BF16 candidate, guarded by `inkling-native-context-memory`. Its
receipts and guard are in `spark2:~/inkling-bf16-cognee-20261122/`.
The original context controller and monitor are copied as immutable helpers.
After the measured Cognee runs, the same three-code archive check fills the
BF16 window to approximately 1,024 tokens below its limit. This capacity check
is separate from the factual and agent timings.

`run.py` executes on the Cognee VM with a 12 GiB memory limit and no swap.
It preserves the unchanged BF16 process through both measured phases and
capacity validation, then restores the selected FP8 container and guard in
its finalization step. FP8 model discovery and arithmetic are checked after
restoration. Cold starts, warmups, storage cloning, capacity validation and
restoration are excluded from factual query times and agent client time.

Full prompts, answers, passages, captures and source reviews remain under
ignored `experiments/adaptive-chunking/raw/pabtso/cognee-20261120/inkling-native-bf16-182k-20261122/`.
Labels are reused only for byte-identical answer text and identical supplied
reader messages. Novel answers require Codex review under the existing strict
rubric, with q37 excluded as predeclared and q38 retaining its trigger-and-round
scope. `analyze.py` audits those reviews, actual requests, source provenance,
near-limit use, service continuity and FP8 restoration before public export.

If the local worker is lost, the retained remote restore action returns port
8004 to the selected FP8 service without restarting Clef:

```bash
python3 ~/inkling-bf16-cognee-20261122/operate.py \
  --root ~/inkling-bf16-cognee-20261122 restore
```

The [411K FP8 replay](../fp8-cognee/RESULTS.md) and
[64K thinking comparison](../native-thinking/RESULTS.md) retain their original
timings and quality labels. One observation per factual question and one agent
run per configuration do not establish repeated-run confidence intervals.
