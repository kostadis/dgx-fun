# Hibana shadow trial

The trial compares the retained Qwen3.8 Flash Next vLLM service on spark1
with Hibana on spark2. Application traffic continues to use vLLM. Hibana is
loopback-only, under a transient user service, with no boot startup.

**Final state, 2026-11-14 01:54 UTC:** `exact` is healthy on spark2 and passed
its final seven-check smoke suite. Both `exact` and `fast` passed all 24
workload checks; `fast` is stopped, with its checkpoint/cache retained.
[Measured results and limits](../../hibana-observations.md),
[compact results](results.json), [final host evidence](evidence/final-spark2.json).
Spark1's vLLM container image and start time are unchanged. Temporary memory
samplers are stopped. No application clients were repointed.

Source: [paper](https://arxiv.org/abs/2611.10469),
[engine](https://github.com/sharar-project/hibana/tree/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb),
[operations](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/operations.md),
[API](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/api.md).

## Pinned setup

- Runtime commit: `d474b74d5a892b2cd1cc8fc50cb17946e2a539bb`.
- Target: `RadixArk/Qwen3.8-Flash-Next-NVFP4`, revision
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`; already cached on both Sparks.
- vLLM image: `sha256:f4612b72c529a2e899c008b4ecc970445352cc004f38bbe2afe92507ca43167d`,
  retained container `qwen38-flash`, spark1 `192.168.1.147:8001`.
  Existing FP8 hybrid side layers, NVFP4 experts, BF16 KV, MTP-2, APC,
  262144 context, eight maximum sequences, utilization 0.80. No flag changes.
- Hibana: spark2, `~/hibana-trial/source`, native `uv sync --locked`,
  Python 3.12, torch 2.13.0+cu130, FlashInfer 0.6.17, CUDA 13.0,
  driver 580.173.02. Bootstrap uv 0.13.8 is private to the trial.
- `exact`: FP8 trunk and KV, NVFP4 experts, FP32 recurrent state,
  exact speculative acceptance, MTP-4. Its name does not promise
  identical floating-point results to vLLM or the original BF16 checkpoint.
- `fast` additionally uses the upstream calibrated copies
  `yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3-qad` at
  `5c4c62a62853caebea8cedda7fbed952efa1effe`. The upstream fetch verified
  each tensor file against its manifest before atomically publishing the
  checkpoint directory. The launcher requires this directory and explicitly
  selects it, so an absent download cannot silently become an FP8 fallback.
- Shared launch bounds: 262144 context, batch four, 300032-token KV pool,
  40 GiB SSD cache, 16 GiB host floor; built-in guard acts below 8 GiB.
  `MAX_JOBS=2` bounds first-use compilation. Port `127.0.0.1:30004`.
- spark2's previous `clef-card-train` container is stopped and retained;
  `qwen38-flash` there remains stopped. No model files, containers or
  previous experiment artifacts were deleted. Ollama configuration is unchanged.

This compares usable configurations on two devices. KV precision, pool size,
speculation, kernels and concurrency limits differ. It does not isolate an
engine-only speedup, and no cross-node tensor parallelism is involved.

## Workloads and interpretation

The frozen fixture has SHA256
`e4f149bd425e6cb28ef9c9e5ff75967302651aa9b33f74a61fcea930f407f4c1`.
[Manifest](fixture-manifest.json) records all 86 Storm Giants source summary hashes
and the cached tokenizer hash. `prepare.py` selects distinct source tokens,
without repeated filler. Raw prompts and responses quote campaign material;
they stay in the ignored `raw/` directory and on the Sparks.

1. Three identical campaign preparation requests, each **200038 rendered input
   tokens** (200001 user-text tokens). Five synthetic control facts are
   interspersed through the archive and explicitly marked as benchmark data.
   A pass requires exact retrieval of all five plus three nonempty prep sections.
   The prose still needs human assessment; retrieval is not a quality benchmark.
2. Four independent persistent sessions, two NPC roles and two coding roles,
   initially about 24K tokens each, three turns per session. Histories retain
   actual assistant responses. A pass requires remembering the session's unique
   code and supplying three proposed actions. These are agent-shaped state
   replays, not full autonomous agent executions. Report both validated turns
   and four-session completion throughput.
3. Two Cognee-style structured decodes: 80 entities and 40 relationships,
   exact graph-content checks, JSON-object mode, 8192-token output allowance.
   They are controlled graph-extraction fixtures, not Cognee end-to-end runs.

Seven smoke checks gate each arm: arithmetic, exact copy, strict JSON schema,
automatic and required tool calls, and both tool-result round trips. Tools are
local inert fixtures; no generated tool call executes on the host.

`replay.py` preserves the complete requests, SSE events and timings, usage,
finish reason, objective pass/fail, and snapshots of server counters. A valid
JSON fragment cut off by the token limit does not count as a completed task.
The same fixture and greedy seed are used in all arms. An early run-specific
prefix seeks a cold application cache; no cache is flushed on production vLLM.
OS caches are warm and uncontrolled. Per-request Hibana cache-hit counts and
vLLM's aggregate prefix-cache counters are retained. Other vLLM users can
contaminate timings; inspect request-counter deltas before interpreting results.

Time to first output includes reasoning or tool output, but excludes role-only
SSE events. Tool parsers may buffer a whole call. SSE gaps measure **chunk
latency**; speculative chunks may contain multiple tokens. These streams do
not support true per-token ITL percentiles. Average TPOT is measured from first
to last output chunk divided by completion tokens minus one. Task/hour and
token/second figures are short-run extrapolations, not sustained capacity.
Per-workload p50/p95 uses empirical nearest ranks, with small sample counts.

`monitor.py` samples global unified-memory availability and swap every two
seconds. CPU RSS alone omits CUDA allocations on GB10. Fixed KV pools allocate
memory before sessions fill; use cache counters as well as resident memory.

## Reproduce and operate

Prepare once on the workstation with the cached target's `tokenizer.json`:

```bash
python3 experiments/hibana/prepare.py --tokenizer /tmp/hibana-tokenizer.json \
  --summaries ~/src/campaigns/stormgiants/docs/summaries \
  --output experiments/hibana/fixtures.json --run-id NEW_UNIQUE_RUN
```

Copy the fixture, `replay.py`, `probe.py`, `start-shadow.sh` and `restore-clef.sh` to
`spark2:~/hibana-trial/`. The launcher refuses conflicting model containers,
an occupied port, or a different source commit. Stop the current shadow service
before changing profile. Start on spark2 with
`bash ~/hibana-trial/start-shadow.sh exact` (or `fast`). Wait for `/health`.
Use a fresh output directory; existing output directories are refused.

```bash
python3 ~/hibana-trial/replay.py --base http://127.0.0.1:30004 \
  --fixture ~/hibana-trial/hibana-fixtures.json --output ~/hibana-trial/evidence/NEW_ARM
python3 ~/hibana-trial/replay.py --base http://192.168.1.147:8001 \
  --fixture ~/hibana-trial/hibana-fixtures.json --output ~/hibana-trial/evidence/NEW_BASELINE
```

Observe with `journalctl --user -u hibana-shadow-exact -f` (or `-fast`)
and `/get_server_info`. `probe.py` checks malformed-request rejection,
stream-disconnect cancellation, and continued health; run it only on an idle
shadow instance. Crash/restart recovery must also record engine readiness
time, the retained SSD cache, cache-restored tokens and answer validation.

For workstation access, open an SSH tunnel:

```bash
ssh -F ~/.ssh/config -L 30004:127.0.0.1:30004 spark2
```

Discover the served model ID at `http://127.0.0.1:30004/v1/models`.
No changes to the application model registry are needed for this isolated trial.

Restore the previous Clef service after stopping whichever shadow profile runs
and confirming `/health` is down and the service cgroup is empty:

```bash
ssh -F ~/.ssh/config spark2 'bash ~/hibana-trial/restore-clef.sh'
```

Then verify `http://192.168.1.121:8002/health`. Spark1 vLLM is independent of
this rollback. Update `current-setup.md` to match any deployed change.
