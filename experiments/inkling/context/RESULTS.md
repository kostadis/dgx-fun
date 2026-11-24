# Inkling context capacity alongside Clef

Completed November 22, 2026 (Pacific). Native vLLM Inkling now serves
**411,520 context tokens**, about **6.28× the original 65,536-token limit**,
at the same `http://192.168.1.121:8004/v1` endpoint and `inkling-2-mini` alias.
It uses **FP8 KV storage and a fixed 13 GiB KV budget**. Clef stayed running,
with unchanged container identity, image, start time and command.
[Audited metrics](results-20261122.json) · [Protocol](README.md) ·
[Current native operations](../NATIVE.md).

| Configuration | Context limit | Validation |
|---|---:|---|
| Original BF16, utilization-derived/extensible cache | 65,536 | Earlier Cognee benchmark; retained for rollback |
| BF16, fixed 13 GiB cache, automatic context fit | 182,144 | Short checks; later [181K extraction and Cognee benchmark](../bf16-cognee/RESULTS.md) completed |
| **FP8, fixed 13 GiB cache, automatic context fit** | **411,520** | **Short checks and near-limit extraction passed; selected** |

The BF16 candidate started in 172.44 seconds; FP8 started in 151.25 seconds.
The largest passing candidate was promoted under the normal `inkling2mini`
container name. The original container remains stopped as
`inkling2mini-64k-20261122`. Neither weights nor checkpoint revision changed.

## Full-window check

The selected server accepted **410,496 input tokens**, leaving 1,024 tokens
below its advertised limit. A synthetic archive contained three exact access
codes near its beginning, middle and end. Inkling returned all three correctly
in a JSON object, finishing naturally after **38 output tokens**, with **zero
reasoning tokens**. The actual API input count exactly matched `/tokenize`.
Total request time was **355.12 seconds (5 min 55 s)**, including long-prompt
processing and generation; it is not a decode-throughput measurement.

The memory guard did not trip. Minimum sampled host memory available was
**14.56 GiB** during the selected FP8 run, and **14.45 GiB** across both
candidates, above the **12 GiB floor**. Final headroom was **14.71 GiB**.
Observed preemptions during the long request were zero. A post-promotion
arithmetic check also passed with zero reasoning tokens. Final checks at
**09:40 Pacific** confirmed Inkling, Clef and Spark 1 Qwen were healthy.

## What changed and what this establishes

The launch settings now include `--max-model-len auto`,
`--kv-cache-memory-bytes 13958643712` and `--kv-cache-dtype fp8`.
The explicit KV budget overrides the utilization-derived budget and disables
the extensible allocator by default in this pinned image. More efficient
cache planning/allocation contributes to the capacity gain; it is not just
a change to the request limit or KV precision.

One active request, a 4,096-token prefill batch, eager execution, prefix caching
and no speculation remain in use. Image
`sha256:f95362fc89f98d28a429d7047caa9090ccab881e1c4ffd0385f643627211c6de`
and checkpoint `b85c3cfa615fc413cc9af3379f491b963f3b586a` remain pinned.
The native per-request thinking controls and OpenAI-compatible client URL
are unchanged. The new guard follows the selected container ID, including
after promotion's rename.

This is the **largest validated context within the chosen conservative 13 GiB
KV budget**, with headroom for Clef and transient allocations. It is not an
absolute hardware maximum. The near-limit archive is one basic capacity and
retrieval check, not a long-context accuracy benchmark. FP8 changes attention
state precision and can affect answers. The earlier Cognee scores and 4.51 s
median were measured on the original 64K BF16 configuration. The subsequent
[FP8 Cognee replay](../fp8-cognee/RESULTS.md) now measures **6.03 s factual p50
and 44/47 complete/grounded**. Its 180.03 s agent recovered the actual
Chapter 4-to-5 handoff with a remaining belief-attribution error. Those
small-prompt benchmarks remain separate from this capacity check and AtlasKV.
The later [182K BF16 follow-up](../bf16-cognee/RESULTS.md) also validated
181,116 input tokens and measured 6.10 s factual p50 with 44/47 complete and
43/47 grounded answers. Its standard-query timings were similar to 411K
FP8's; the older 64K timing was not recovered by retaining BF16 precision
with the larger fixed-cache configuration.

Operational scripts, source hashes, candidate logs, memory samples and the
synthetic probe receipts are retained at `spark2:~/inkling-context-20261122/`.
Its `export-bundle.json` can be copied locally and validated/exported with:

```bash
python3 experiments/inkling/context/analyze.py \
  --input /tmp/inkling-context-20261122-bundle.json \
  --output experiments/inkling/context/results-20261122.json
```
