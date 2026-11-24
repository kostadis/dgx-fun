# Native Inkling context fitting alongside Clef

This is an operational context-capacity check, separate from the Cognee
quality/speed comparisons and AtlasKV. The user requested the largest native
Inkling window that fits alongside Clef. **Completed:** the selected FP8 limit
is **411,520 tokens**, and a **410,496-token input** returned all three access
codes correctly while Clef stayed unchanged. [Results](RESULTS.md) ·
[Audited metrics](results-20261122.json).
The later [standard Cognee replay](../fp8-cognee/RESULTS.md) separately measures
factual and agent speed/quality on the selected FP8 service.
The [182K BF16 follow-up](../bf16-cognee/RESULTS.md) subsequently validated its
near-limit window and measured the same factual/agent workloads, then restored FP8.

## Budget and candidate selection

Both candidates use the same pinned checkpoint and image as the original
native deployment, with one active request, a 4,096-token prefill batch,
eager execution, prefix caching and no speculation. Model weights are unchanged.
Only context fitting and KV storage change:

```text
--max-model-len auto
--kv-cache-memory-bytes 13958643712
--kv-cache-dtype auto     # BF16 control, or fp8 for the other candidate
```

The explicit **13 GiB KV budget** replaces vLLM's utilization-derived KV
allocation. It is a conservative ceiling based on observed headroom on this
shared Spark, with a **12 GiB host-memory floor** and room for transient
allocations and Clef activity. Automatic fitting finds the largest advertised
context within that budget. This is not a claim about an absolute hardware
maximum under every possible cache precision or workload.

In this pinned vLLM runtime, supplying an explicit KV budget also disables
the default extensible KV allocator. The original 64K service used that
allocator. The capacity change therefore includes cache planning and allocation,
not just a higher request-length setting or a precision change.

The pinned Inkling attention implementation explicitly supports FP8 E4M3 KV
on capability-family 120 devices, which includes GB10. FP8 changes the stored
attention state, not model weights; it can change answers. The existing Cognee
benchmark remains a measurement of its original BF16 deployment. The larger
window is not assumed to preserve that benchmark's speed or quality.

## Validation and promotion

For each cache type, `run.py` checks startup, advertised context, arithmetic,
exact JSON and a source-negation question with the native thinking-channel ban.
It then chooses the largest passing candidate and submits a synthetic archive
prompt within approximately 1,024 tokens of that limit. Three exact access
codes appear near the beginning, middle and end of the archive. Token counts
are checked through the native tokenizer API and actual completion usage;
the output must finish naturally with the three correct codes and no reasoning.

Each candidate and phase has a separate prefix-cache salt. This near-limit
extraction is a capacity and basic retrieval smoke check, not a long-context
accuracy benchmark or a realistic campaign-preparation throughput measurement.
Short prompts and source text are synthetic; no D&D passages are sent anywhere.

The original `inkling2mini` container stays available. Promotion renames it to
`inkling2mini-64k-20261122`, then gives the validated candidate the normal
`inkling2mini` name and the same port 8004 endpoint. Its guard monitors the
container ID, so it survives that rename. A failed candidate cannot stop Clef.
If no candidate passes, the controller restores the original native service
and its guard. The guard stops only the candidate below the memory floor.
Clef identity, image, start time and readiness are checked throughout.

## Saved receipts and operations

The controller runs on Spark 2 as transient user unit `inkling-context-20261122`.
Scripts, source hashes, startup/validation receipts, synthetic probes, logs
and memory samples are saved at `spark2:~/inkling-context-20261122/`.
The live candidate guard is `inkling-native-context-memory`.
The final `result.json` identifies the selected context, dtype, exact launch
command, retained rollback container and validation results.

Inspect progress on Spark 2:

```bash
cat ~/inkling-context-20261122/progress.json
systemctl --user status inkling-native-context-memory
journalctl --user -u inkling-context-20261122 -n 30
```

To deliberately return to the saved 64K service after this operation finishes,
run the retained controller's restore action on Spark 2:

```bash
python3 ~/inkling-context-20261122/run.py \
  --root ~/inkling-context-20261122 restore
```

The restore action stops only candidate Inkling containers, frees the normal
container name, and restarts the saved 64K native container and its original
guard. It does not restore DSpark or restart Clef. See the
[native operations guide](../NATIVE.md) for the currently selected service.
