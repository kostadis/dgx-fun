# Unsloth Flash-1017 UD-IQ1_S trial

Status: trial failed before inference. Head rebooted; original two-node vLLM
service restored and verified at 2026-10-21 03:45 UTC.

User requested a trial of the 1-bit Unsloth quant following the EXL3 600K test.
Initial target is one DGX Spark, 8K context, one sequence, ordinary decoding.
Existing two-node vLLM containers and configuration are preserved for restoration.

- Weights: `unsloth/DeepSeek-V4.1-Flash-1017-GGUF`, revision
  `60ad3651641b5c97679c807e064ab1d7a3ca1455`, `UD-IQ1_S` (manifest alongside).
- Runtime: `danielhanchen/llama.cpp`, `deepseek41`, commit
  `b72050d03ae14bb3cb5138ecefab3f3f65157d31` (upstream PR 30817).
- Separate build: CUDA 13, sm_121, Release, static libraries, llama-server.
- Remote directory: `spark:~/unsloth-flash1017-trial/`.
- Download: `download-http.py` streams with curl and verifies each SHA256.
  The first Hugging Face Xet attempt was stopped after its process grew to
  approximately 8 GiB RSS. Partial Xet files remain under `model/`; the active
  download is under `model-http/`. Streaming restored available memory.
- Logs: `build.log`, `build.exit`, `download-http.log`, `verified.json` on Spark.
- Launch: `run-trial.sh`, port 8002, started approximately 03:33 UTC.

`compare.py` uses ten exact hex-copy cases, two arithmetic questions, prose,
and a coding prompt. `vllm-baseline.json` captures current serving behavior;
all 12 automatically scored checks passed. The baseline was collected during
build/download, so timings are preliminary, not a controlled benchmark.

The upstream PR contains an unconfirmed-on-Spark report of lazy-loading output
corruption on a different quant/GPU combination. Preserve the unmodified branch
for this trial and include copying checks; passing these is not proof that the
reported issue cannot affect other prompts.

Sources: [Unsloth guide](https://unsloth.ai/docs/models/deepseek-v4.1-flash),
[runtime PR](https://github.com/ggml-org/llama.cpp/pull/30817),
[weights](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF).

## Load attempt and recovery status

All five shards passed SHA256 verification; `verified.json` is on the head.
The runtime built successfully. A GPU-device probe while vLLM was loaded failed
with CUDA out-of-memory at cudaSetDevice; vLLM remained healthy. After stopping
both vLLM containers, the same probe succeeded and reported about 118 GiB free.

The model was launched with 8K context, all layers on GPU, Engram lazy loading
on CPU, batch 256 / microbatch 128, and no draft model. `/health` reported loading.
The last retrieved memory sample showed approximately 51 GiB available while
weights were still loading. The next check could not reach the head. SSH failed
with `No route to host`; pings from the worker failed over both 192.168.1.147
and 10.100.16.1. The worker remained healthy. This does not establish the cause:
OOM, a driver/kernel failure, or another host failure must be checked after
recovery. No 1-bit quality or speed result was obtained.

Remote startup logs and two-second memory samples are in `server.log` and
`memory.tsv`; retrieve them after the head returns. The trial has no auto-start
service. Manually stopped vLLM containers need explicit restart after reboot.
Do not automatically repeat the full GPU-offload trial.

Updated vLLM timing baseline, after compilation: `vllm-baseline-final.json`.
All ten direct-copy and two arithmetic cases passed. All ten forced tool-copy
checks passed too (`vllm-tool-copy.json`). These are baseline results only.

Head returned at 03:39 UTC with uptime under one minute. Both original vLLM
containers were explicitly restarted and passed readiness validation.
Recovered memory trace ends at 03:36:17 with 10.08 GiB MemAvailable, falling
about 1 GiB/s; its final bytes are NULs. The true minimum was not captured.
The previous boot kernel log ends at 03:34:54 without a fatal/OOM event during
the load. Earlier 03:14:25 NVIDIA allocation failures match the separate device
probe while vLLM was still resident, not the later trial failure.

## Final outcome

At 03:45:42 UTC, `/health` returned 200, `/v1/models` reported the original
`DeepSeek-v4.1-Flash-EXL3` ID and 600,000-token context, and arithmetic returned
323 in 0.31 seconds. See `recovery.json`. Both original containers are running
with `unless-stopped`. The trial server is not running. The downloaded weights
and build remain available; the original launch script records a failed
configuration and should not be repeated unchanged.

There is no measured 1-bit inference throughput or accuracy. A future attempt
should reduce per-node memory demand and enforce a host-memory limit; splitting
GGUF inference across both Sparks is a candidate that still needs validation.
