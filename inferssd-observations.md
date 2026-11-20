# InferSSD on DGX Spark: experiments and results

## Outcome

**One DGX Spark successfully ran DeepSeek-V4.1-Flash-1017 through InferSSD at 8K context.** The measured decode rate was **5.40–5.58 tokens/s**, with **183.86–189.37 prompt tokens/s** on two requests containing 3,929 input tokens and producing 256 output tokens. Both arithmetic checks and all three exact-copy checks passed.

Following the successful 32K and 64K retrieval tests, the user requested a return to Qwen at **2026-11-06 01:06 UTC**. InferSSD is stopped, its 64K configuration is retained, and Qwen was restored and verified at **01:18:10 UTC**, serving `qwen3.8-flash-next` on port 8001 with 262,144 context. The larger-context results are appended below. Qwen retains its `unless-stopped` policy; no InferSSD boot service has been installed. `spark2` was untouched.

These results demonstrate that this model can run on one Spark using SSD-backed experts. The follow-up tests establish retrieval and timing at the tested sizes; they do not establish performance beyond 64K, production concurrency, or model quality on real coding workloads.

## Objective and starting state

The experiment followed a review of [SSD-LLaMA](https://arxiv.org/html/2609.18110v1) and its implementation, [InferSSD](https://github.com/wici-ai/inferssd). The question was whether the approach could run a model substantially larger than one Spark's memory without repeating the earlier failed full-offload IQ1_S attempt.

Before the experiment, `spark` served `qwen3.8-flash-next` through the retained `qwen38-flash` Docker container on port 8001. Its arithmetic baseline returned `323` for 17 × 19. About 995 GiB was free on the SSD. Qwen remained running during compilation and checkpoint download, then was stopped for profiling and inference.

This was a single-machine experiment. The second Spark and the interconnect were not used for inference.

## Hardware, software, and checkpoint

| Component | Configuration |
|---|---|
| Host | `spark`, hostname `gx10-46ea` |
| Processor / GPU | NVIDIA GB10, ARM64 CPU, Blackwell GPU, sm_121 |
| Memory | 128 GB unified memory; approximately 121 GiB visible in Linux |
| Storage | Local EXT4 filesystem, Samsung MZALC4T0HBL1-00B07 NVMe, nominal 4 TB |
| Runtime | InferSSD v0.1.2 |
| Runtime commit | `8f060b54dfc8205bf08ebb4ff410b5e2cbe2756f` |
| Build | Native CUDA 13.0, `linux-cuda` preset, CUDA architecture 121, two build jobs |
| Checkpoint | `mrfakename/DeepSeek-V4.1-Flash-1017-GGUF`, Q8_0 |
| Checkpoint revision | `1df54e7d1b1e442051ad393ce871870a929f243e` |
| Download | Ten shards, 507,954,227,008 bytes / 473.07 GiB |
| Remote installation | `~/inferssd-trial/` |

Every shard passed its size and SHA256 checks against Hugging Face LFS metadata. The runtime indexed expert ranges directly in the original GGUF files; no additional expert pack was created.

The Q8_0 label does not mean every tensor uses that format: this checkpoint retains MXFP4 routed experts. This text-model conversion omits vision and MTP/DSpark. It differs from the retained two-Spark EXL3 deployment in format and supported features, so the results are not a direct performance or quality comparison with that deployment.

## Experiment 1: installation and preparation

The native runtime built successfully without modifying its source. Missing liburing development and runtime files were downloaded and unpacked into a private dependency directory. Python, CMake, Ninja, and framework dependencies were also isolated under the trial directory; no system package installation was required.

Two preparation problems were resolved:

1. **CUDA initialization while Qwen occupied memory.** The schema-only descriptor command attempted CUDA initialization and could abort with `cudaSetDevice` out-of-memory. Generating that descriptor with `CUDA_VISIBLE_DEVICES=` avoided GPU initialization for this metadata step. Normal GPU visibility was retained for inference after Qwen stopped.
2. **Overlapping resumed downloads.** Changing download concurrency briefly left four older curl processes writing alongside their replacements. A size check rejected an oversized partial. The four affected files were reset to known-clean 20 GiB prefixes and downloaded again. All final hashes passed. The retained downloader now uses an exclusive lock, child-process termination, immediate error reporting, and size plus SHA256 verification before accepting a shard.

The second issue delayed preparation but did not reach inference: the launch runner required an all-shards-verified completion marker.

## Experiment 2: profiling and bounded inference

Qwen was stopped at **2026-11-05 23:54:45 UTC**. The systemd user scope and cgroup v2 prerequisites passed without sudo.

The storage probe measured **12.13 GB/s** with queued 4 MiB reads. This is a probe measurement, not sustained expert-delivery throughput during generation. The planner recognized GB10's unified memory and omitted the cost of a discrete CPU–GPU transfer link.

The launch configuration was:

| Setting | Value |
|---|---|
| Context / batch / microbatch | 8192 / 8192 / 8192 |
| Concurrent sequences | 1 |
| Memory policy / native ledger | 100 GiB |
| Host headroom | 12 GiB |
| Available-memory stop threshold | 8 GiB |
| Server swap | Disabled |
| CUDA expert cache | Automatic |
| Decode threads | 10, pinned to CPUs 5–9 and 15–19 |
| Prefill threads | 19 |
| Engram prefetch / fused HC | Enabled |
| Prompt disk cache | Disabled |
| API | `127.0.0.1:8002` |

On GB10, CUDA allocations are not charged to the CPU cgroup. The runtime's shared-memory accounting and global MemAvailable checks therefore complement the cgroup limit; the cgroup alone does not describe total GPU residency.

The loader reported **268.95 GiB of deferred expert payload** and an **8,530 MiB CUDA model buffer**. Large CPU mapped-buffer values represent virtual file mappings rather than resident memory. During prefill, the runtime reclaimed expert-cache blocks to make room for working buffers.

Readiness took **45.03 seconds** after the owned launcher started, excluding the preceding profiling step.

### Correctness smoke checks

| Check | Expected result | Outcome |
|---|---|---|
| 17 × 19 | `323` | Passed |
| 144 ÷ 12 | `12` | Passed |
| Exact hexadecimal copy | `e71b04c9a28f63d5` | Passed |
| Exact hexadecimal copy | `9d60a2f7bc814e35` | Passed |
| Exact hexadecimal copy | `43f1c8e09ab672d5` | Passed |

These checks verify basic coherent responses and copying behavior. They are not a model-quality benchmark or a numerical-equivalence test against another runtime.

### Streaming performance

The benchmark repeated a weather-station prompt containing 3,929 tokens. Requests used temperature zero, thinking off, a 256-token output limit, and EOS suppression to obtain the full measurement window. Prompt caching was disabled in the request, and the server reported zero cached prompt tokens.

| Request | Input tokens | Output tokens | Prefill tok/s | Decode tok/s | First token, s | Total, s | Cached tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| First | 3,929 | 256 | 183.86 | 5.40 | 23.16 | 68.77 | 0 |
| Repeat | 3,929 | 256 | 189.37 | 5.58 | 22.59 | 66.65 | 0 |

Prefill and decode rates above come from the server's timing fields. First-token and total latency were measured by the streaming client. They cover different timing boundaries and should not be substituted for one another.

Five short smoke checks preceded these requests. No OS page-cache purge was performed. The first request is therefore not a controlled cold-storage result, and the repeated request may benefit from retained experts or filesystem cache despite having no prompt-cache hits. The small difference between the two runs does not establish a repeatable speedup.

### Memory and cleanup

During the checks and streaming requests, the minimum two-second sample of MemAvailable was **12.09 GiB**. The broader cutover window reached 10.63 GiB while Qwen was still shutting down; that lower value is not an InferSSD inference measurement.

The CPU cgroup recorded **zero OOM events**. After the first run, cleanup receipts confirmed that the worker process group and systemd scope were gone. This supports a successful short bounded run; larger contexts and sustained concurrent traffic were not tested.

## Experiment 3: restoration and resumed deployment

The first run finished validation at **2026-11-05 23:59:24 UTC**. Its automation stopped InferSSD and restored Qwen, which passed health, model-ID, and arithmetic verification at **2026-11-06 00:13:18 UTC**. Treating the requested deployment as a temporary benchmark was an incorrect assumption.

InferSSD was restarted at **00:18 UTC** and left running. Its new instance passed health and an arithmetic request returning `323` in **4.86 seconds**. The performance benchmark was not repeated because the runtime, weights, and settings were unchanged.

The following snapshot describes the 00:18 UTC resumption; the larger-context deployment below supersedes its process IDs and context.

| Item | Deployment at 00:18 UTC |
|---|---|
| Native run | `~/inferssd-trial/runs/20261106T001817Z/` |
| Launcher PID at startup | `2837829` |
| Native server PID at 00:22 UTC | `2837857` |
| Launcher log | `~/inferssd-trial/active-server.log` |
| Served ID | `DeepSeek-V4.1-Flash-1017-Q8_0-00001-of-00010.gguf` |
| Qwen | Stopped, container retained unchanged |
| Automatic restoration | None scheduled |
| Boot startup | Not installed |

For access from another machine:

```bash
ssh -L 8002:127.0.0.1:8002 spark
```

Then use `http://127.0.0.1:8002/v1`. The service is bound to loopback on Spark; it is not exposed directly on the LAN.

## Interpretation

The practical requirement is an appropriate GGUF checkpoint, enough local SSD capacity, and a native InferSSD build. No custom ARM64 or unified-memory port was needed for this tested configuration.

The result is usable single-Spark inference for a checkpoint much larger than memory, at roughly 5.5 generated tokens/s in this workload. It also requires dedicating the machine's memory budget to InferSSD: the existing Qwen service was stopped rather than co-hosted.

The experiment does not establish performance at the model's maximum context, tool-calling reliability, vision support, comparative answer quality, or long-running service stability. It also does not justify applying published speedup ratios to this installation: no matched stock llama.cpp performance baseline was run locally.

## Evidence and reproduction

- [Deployment runbook and retained configuration](experiments/inferssd/README.md)
- [Measured result summary](experiments/inferssd/evidence/summary.json)
- [Full requests, responses, and streaming timing events](experiments/inferssd/evidence/results.json)
- [Machine profile](experiments/inferssd/evidence/machine.json)
- [Checkpoint manifest and hashes](experiments/inferssd/manifest.json)
- [Inference memory samples](experiments/inferssd/evidence/memory-inference.jsonl)
- [Cgroup resource counters](experiments/inferssd/evidence/resources-after.json)
- [Initial run cleanup receipt](experiments/inferssd/evidence/guardian-cleanup.json)
- [Brief Qwen restoration evidence](experiments/inferssd/evidence/restored.json)
- [Resumed live-instance health and arithmetic evidence](experiments/inferssd/evidence/active-smoke.json)
- [Current service inventory](current-setup.md)

Native telemetry and the compressed server log for the original benchmark remain on Spark under `~/inferssd-trial/runs/20261105T235558Z/`. All timestamps in this report are UTC.

## Larger context validation — 2026-11-06

Both **32K and 64K context tests passed**. At the end of testing, InferSSD was left running at **65,536 context** on spark, with batch and microbatch capped at 8192, the same 100 GiB memory policy, and 12 GiB host headroom. Qwen was stopped during these tests; spark2 was untouched. The later requested Qwen restoration is recorded below.

Each test used a new server process and a deterministic archive of numbered filler records. Five catalog facts were inserted at approximately 1%, 25%, 50%, 75%, and 99% of the archive. The request asked for an exact JSON object containing all five hexadecimal values. Token counts were checked using the server's template/tokenizer endpoints and confirmed by generation usage. Each response contained 76 output tokens; all five values matched exactly.

| Configured context | Actual input | Facts correct | Prefill tok/s | Decode tok/s | First token, s | Total, s | Minimum available GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 30,498 | 5/5 | 149.38 | 4.46 | 209.37 | 221.29 | 12.09 |
| 65,536 | 61,986 | 5/5 | 113.50 | 3.64 | 553.46 | 567.21 | 12.21 |

Both requests reported zero cached prompt tokens. Thinking was disabled and temperature was zero. Memory was sampled every two seconds; the minimum above covers the actual request. The active 64K cgroup reported zero OOM events. These are single retrieval tests at each size, not a broad long-context quality evaluation.

The 64K test reached its first token after **9 minutes 13 seconds**. Larger context is feasible within the existing memory budget, but long-prompt prefill is costly in this configuration. The shorter 32K test reached its first token in about 3 minutes 29 seconds. These requests use a different prompt and output length from the original 4K/256-token benchmark, so their rates are not a controlled comparison with that benchmark.

The model metadata declares a 1,048,576-token training context. This experiment validates only the tested 32K and 64K configurations; it does not establish that the full declared context is usable on this machine.

The retained launcher now defaults to **65,536 context**, **8192 batch**, and **8192 microbatch**. The model registry idle timeout was increased to 1200 seconds to cover the observed long prefill. Context overrides use the typed `--context` option; the runtime rejects the retired `CONTEXT` environment variable.

64K deployment snapshot at validation: launcher PID `2888656`; native server PID: `2888680`; run: `~/inferssd-trial/runs/20261106T004136Z/`. Health is OK. No automatic Qwen restoration or boot startup is scheduled.

Evidence: [context summaries](experiments/inferssd/context-tests/summary.json), [32K request/results](experiments/inferssd/context-tests/32768-result.json), [64K request/results](experiments/inferssd/context-tests/65536-result.json), [32K prompt](experiments/inferssd/context-tests/32768-prompt.txt), [64K prompt](experiments/inferssd/context-tests/65536-prompt.txt), [test script](experiments/inferssd/context-tests/test-context.py), [active 64K state](experiments/inferssd/context-tests/active.json).

## Requested return to Qwen — 2026-11-06

After the larger-context experiments, the user explicitly requested Qwen again. InferSSD was stopped, and the retained `qwen38-flash` container was restarted at 01:06 UTC without changing its configuration. Model and arithmetic verification passed at **01:18:10 UTC**; final health verification at 01:19:33 UTC returned HTTP 200. The API advertises **`qwen3.8-flash-next`**, **262,144 context**, on **192.168.1.147:8001**. Arithmetic returned `323`.

InferSSD has no running server and port 8002 is closed. Its 64K configuration, build, and weights remain available for a future switch. Qwen retains its `unless-stopped` restart policy. About 16.34 GiB was available after restoration. spark2 was untouched. [Final restoration evidence](experiments/inferssd/evidence/qwen-return-after-context.json).
