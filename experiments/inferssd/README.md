# InferSSD on spark

[Experiment narrative and results](../../inferssd-observations.md)

Status: **Qwen is restored and verified on spark; InferSSD is stopped**, following the user's request after the context tests. Restoration passed at **2026-11-06 01:18:10 UTC**. The validated 64K InferSSD launcher, weights, and results are retained. spark2 is untouched. [Final restoration evidence](evidence/qwen-return-after-context.json).

The first benchmark run below completed successfully, briefly restored Qwen at 00:13:18 UTC, and was then resumed because the requested outcome was to leave InferSSD running. That resumed instance was used for the later context tests, after which the user requested the return to Qwen documented above. No InferSSD boot service has been installed.

Last 64K InferSSD launcher (now stopped): PID `2888656`; native run: `~/inferssd-trial/runs/20261106T004136Z/`; launcher log: `~/inferssd-trial/context-tests/65536-launch.log`. Former API: `127.0.0.1:8002` on spark; currently stopped.

## Result

One GB10 Spark ran the tested DeepSeek-V4.1-Flash-1017 Q8_0 GGUF at 8K context. Both arithmetic checks and all three exact hexadecimal-copy checks passed. Readiness took 45.0 seconds after starting the owned launcher (profiling excluded).

| Measurement | First request | Repeated request |
|---|---:|---:|
| Input tokens | 3,929 | 3,929 |
| Output tokens | 256 | 256 |
| Server prefill rate | 183.86 tok/s | 189.37 tok/s |
| Server decode rate | 5.40 tok/s | 5.58 tok/s |
| Client time to first streamed token | 23.16 s | 22.59 s |
| Client total time | 68.77 s | 66.65 s |
| Cached prompt tokens | 0 | 0 |

The repeated request used the same weather-station prompt, with `cache_prompt=false`, prompt disk caching disabled, temperature zero, thinking off, and EOS suppressed to obtain 256 output tokens. Five short checks preceded the measurements. No OS page-cache purge was performed. These are two spot measurements on one prompt, not a controlled cold-storage benchmark or a long-context / concurrency evaluation.

Minimum sampled MemAvailable during the checks and benchmarks was **12.09 GiB**, sampled every two seconds. The earlier cutover window included Qwen shutdown and reached 10.63 GiB; that is not an InferSSD inference minimum. Cgroup OOM counters remained zero. Shutdown receipts confirm the worker process group and systemd scope are gone.

The SSD probe measured **12.13 GB/s** for queued 4 MiB reads. This is a storage-probe result, not observed expert-delivery bandwidth. The planner identified unified memory and omitted a discrete CPU–GPU link cost.

Evidence: [summary](evidence/summary.json), [full responses and streaming events](evidence/results.json), [machine profile](evidence/machine.json), [profile estimates](evidence/profile.json), [inference memory samples](evidence/memory-inference.jsonl), [resource counters](evidence/resources-after.json), [clean shutdown](evidence/guardian-cleanup.json).

## Pinned configuration

- Host: `spark`, gx10-46ea, NVIDIA GB10, ARM64, sm_121, 121.7 GiB reported physical memory.
- Runtime: [wici-ai/inferssd v0.1.2](https://github.com/wici-ai/inferssd/tree/v0.1.2), commit `8f060b54dfc8205bf08ebb4ff410b5e2cbe2756f`.
- Checkpoint: `mrfakename/DeepSeek-V4.1-Flash-1017-GGUF`, Q8_0, revision `1df54e7d1b1e442051ad393ce871870a929f243e`.
- All ten shards, **507,954,227,008 bytes / 473.07 GiB**, verified against Hugging Face LFS SHA256 metadata. [Manifest](manifest.json), [completion receipt](evidence/download-complete.json).
- Remote root: `~/inferssd-trial/`; checkpoint under `model/`; source under `source/`; Python environment under `venv/`.
- Native build: CUDA 13.0, `linux-cuda` preset, `CMAKE_CUDA_ARCHITECTURES=121`, two build jobs.
- liburing packages were downloaded and unpacked privately under `deps/`; no system packages were changed.
- Original GGUF expert ranges indexed directly; no additional expert pack created.
- Initial benchmark: context/batch/microbatch 8192 each. Retained larger-context configuration: context 65536, batch/microbatch 8192, one sequence; API `127.0.0.1:8002`; no prompt disk cache.
- Memory policy/native ledger: 100 GiB; server swap disabled; 12 GiB host headroom; resource monitor requires 8 GiB MemAvailable. CUDA allocations on GB10 are not charged to the Linux cgroup, so the native shared-memory ledger and MemAvailable checks matter alongside the CPU cgroup cap.
- Planner: 10 decode threads pinned to CPUs 5–9,15–19; 19 prefill threads; automatic CUDA expert cache; Engram prefetch and fused HC enabled.
- Exact served ID: `DeepSeek-V4.1-Flash-1017-Q8_0-00001-of-00010.gguf`.

The loader reported 268.95 GiB of deferred expert payload and an 8,530 MiB CUDA model buffer. Its large CPU mapped-buffer sizes describe file mappings, not resident physical memory. The runtime reclaimed expert-cache blocks when the 4K prefill required working memory.

This text-model GGUF omits vision and MTP/DSpark. It is a different serving configuration and quantization from the retained two-Spark EXL3 deployment.

## Reproduce using the retained installation

The original `qwen38-flash` container must be stopped before starting InferSSD. The launcher refuses to coexist with it.

```bash
ssh spark 'docker stop qwen38-flash'
ssh -t spark '~/inferssd-trial/start.sh --machine ~/inferssd-trial/machine.json'
```

In another terminal, forward the loopback API if needed:

```bash
ssh -L 8002:127.0.0.1:8002 spark
```

Use `http://127.0.0.1:8002/v1`. Stop the foreground launcher with Ctrl+C, wait for its cleanup, then restore Qwen:

```bash
ssh spark 'docker start qwen38-flash'
```

Wait for `/health` and `/v1/models` on port 8001 before using Qwen again. This trial installs no boot service or automatic InferSSD startup.

Configuration copies: [launcher](start.sh), [owned-launch shim](launch.py), [runtime policy](runtime-policy.json), [build commands](build.sh), [verified downloader](download.py), [validation requests](check.py), [effective settings](evidence/settings.ssd). These scripts assume the recorded remote directory and installed dependencies. The trial's native telemetry and compressed server log remain on spark under `runs/20261105T235558Z/`.

## Preparation and recovery notes

The native build completed. Its schema-only descriptor step initializes CUDA and can fail with cudaSetDevice out-of-memory while Qwen occupies the GPU. Running only descriptor generation with `CUDA_VISIBLE_DEVICES=` succeeded; inference retained normal GPU visibility after Qwen stopped. No runtime source patch was needed.

Changing download concurrency briefly left four old curl processes overlapping their replacements. A file-size check rejected an oversized partial. Those four partials were reset to known-clean 20 GiB prefixes; their final SHA256 hashes all passed. The replacement downloader uses an exclusive lock, explicit child termination, immediate error reporting, and size plus SHA256 checks before accepting each shard. The inference runner required the all-shards-verified marker.

The pre-trial Qwen arithmetic check returned 323. Qwen was stopped at **2026-11-05 23:54:45 UTC**. InferSSD finished validation at **23:59:24 UTC**, stopped cleanly, and the runner started the retained Qwen container. Qwen was verified at **2026-11-06 00:13:18 UTC**: `/health` HTTP 200, `/v1/models` advertised `qwen3.8-flash-next` with 262,144 context, and arithmetic returned `323`. About 16.0 GiB was available after restoration; roughly 521 GiB remained free on SSD. During that brief Qwen restoration, port 8002 was closed. InferSSD was subsequently resumed as noted above. [Restoration evidence](evidence/restored.json).

## Larger-context follow-up

The 32K test retrieved all five facts from 30,498 tokens; the 64K test retrieved all five from 61,986 tokens. First-token latency was 209.37 s and 553.46 s respectively. Minimum sampled request memory availability was 12.09 GiB and 12.21 GiB. The 64K instance passed validation and was subsequently stopped for the requested return to Qwen. Its retained launcher default is 65536.

See the [full narrative](../../inferssd-observations.md) and [measured summaries](context-tests/summary.json). Prompts, responses, timing events, memory samples, and the test script are saved under `context-tests/`.

## Requested return to Qwen after context tests

InferSSD was stopped and the retained `qwen38-flash` container restarted at 2026-11-06 01:06 UTC. Qwen passed model/arithmetic verification at 01:18:10 UTC and a final health check at 01:19:33 UTC: HTTP 200, `qwen3.8-flash-next`, 262,144 context, answer `323`. About 16.34 GiB was available. Port 8002 is closed; Qwen remains on port 8001 with `unless-stopped`. [Evidence](evidence/qwen-return-after-context.json).
