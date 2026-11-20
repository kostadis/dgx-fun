# Qwen3.8-Flash-Next with external DFlash 2 on DGX Spark

## 2026-11-07 — controlled 8K comparison

**The external drafter runs on a single GB10 Spark.** On three short prompts,
the repeated DFlash pass was **22.3% faster than MTP-2** and **88.2% faster
than plain decoding**, using the geometric mean of native decode rates.
Prose was effectively tied with MTP; code and the prime list improved.
These are workload-specific measurements from a small experiment.

This experiment uses llama.cpp with an IQ4_NL target. The previous vLLM
NVFP4/FP8 hybrid deployment is a different runtime and quantization; the
comparison to that deployment cannot isolate the drafter's contribution.
The original `qwen38-flash` container is stopped and retained. spark2 was
untouched. The final context validation is recorded separately below.

### Exact configuration

- spark1, `192.168.1.147`, GB10 / ARM64 / sm_121, about 121.7 GiB unified RAM.
- CUDA 13.0 native Release build, llama.cpp commit
  `ca89968a23a13e571a6b802a9891d22763153b61`.
- Official `ggml-org/Qwen3.8-Flash-Next-GGUF`, revision
  `91fb00353e766eb43944ed223665ea4fb87dad47`.
- Same two-shard IQ4_NL target in all three llama.cpp modes.
- MTP: `mtp-Qwen3.8-Flash-Next-Q4_0.gguf`, `draft-mtp`, maximum 2 draft tokens.
- DFlash: `dflash-Qwen3.8-Flash-Next-Q4_0.gguf`, `draft-dflash`, maximum 5 draft
  tokens, minimum draft probability 0.75. The external sidecar is about 335 MiB.
- 8192 context, one sequence, batch/microbatch 1024, Flash Attention on,
  99 target/draft GPU layers, 10 generation / 20 batch CPU threads.
- mmap loading; runtime's lazy-read PLE table avoids a full GPU table allocation.
- Prefix caching and context shifting disabled. All measured cached counts were 0.
- Temperature 0, seed 604, thinking disabled, 384 generated tokens per performance
  request. Same prompts in every mode, two passes per llama.cpp mode.
- 100 GiB CPU cgroup memory limit, no cgroup swap, plus a separate guard that
  terminates the server if global MemAvailable falls below 8 GiB. CUDA unified
  allocations are not fully accounted for by the CPU cgroup.

All four GGUF files passed size and SHA256 verification. Exact filenames,
hashes, scripts and build evidence are in [the experiment directory](experiments/qwen38-dflash2/README.md).

### Measured decode speed

Native llama.cpp generation tokens/sec, **second pass**; excludes prefill.

| Prompt | Plain | MTP-2 | DFlash-5 | DFlash vs MTP |
|---|---:|---:|---:|---:|
| Distributed-cache guide | 26.8 | 34.6 | 34.3 | -0.9% |
| Python LRU cache and tests | 26.8 | 40.9 | 51.6 | +26.2% |
| Prime-number list | 26.8 | 49.6 | 72.6 | +46.3% |

First-pass native rates were plain **21.8 / 22.2 / 24.5**, MTP
**26.7 / 33.6 / 44.0**, and DFlash **25.4 / 35.8 / 60.3** in the same
prompt order. DFlash's first-pass geometric-mean advantage over MTP was
**11.5%**, compared with 22.3% on repeat. Warm-up materially affects these
numbers. Mode order was plain, MTP, then DFlash; there was no randomized
ordering or MTP repeat after DFlash. This is not a statistical performance claim.

DFlash's repeated-pass acceptance was **78.5%, 85.0%, 97.5%** respectively;
MTP's was **89.4%, 95.3%, 99.6%**. Larger accepted blocks can make DFlash
faster even with a lower acceptance fraction.

### Operational reference: original vLLM deployment

Completion tokens divided by total HTTP request seconds, **including prefill**.
These wall rates deliberately use a different timing boundary from the native
decode table above and must not be mixed with it.

| Prompt | Original vLLM MTP-2 | llama.cpp plain repeat | llama.cpp MTP repeat | llama.cpp DFlash repeat |
|---|---:|---:|---:|---:|
| Guide | 29.9 | 26.4 | 33.9 | 33.6 |
| Code | 37.4 | 26.4 | 39.8 | 50.0 |
| Primes | 44.7 | 26.4 | 48.0 | 69.2 |

Original vLLM had one measured pass and a 262144-token configured context;
llama.cpp used 8192 for this controlled comparison. Runtime, quantization,
context reservation and warm-up differ. These numbers show observed service
latency on these requests, not a pure DFlash speedup over vLLM.

### Correctness and operational observations

- Original vLLM, plain, MTP and DFlash each passed both arithmetic checks and
  three exact random-string copy checks.
- Performance outputs were capped at 384 tokens; generated Python tests were
  **not executed** and the prose/code were not scored for quality.
- Plain and DFlash repeated their own three outputs byte-for-byte. MTP's
  prose and code changed on repeat. Across modes, prose/code differed even
  at temperature zero; the prime list matched. The cause was not established.
  These results do **not** prove lossless equivalence to plain decoding.
- Minimum global MemAvailable during the complete 8K runs: plain **47.68 GiB**,
  MTP **44.78 GiB**, DFlash **45.08 GiB**. No memory-guard termination occurred.
- Startup was roughly 4–5 minutes per load. `/health` readiness after the
  readiness poll began was 302.35 s plain, 278.27 s MTP, 244.30 s DFlash;
  polling began slightly after process launch.
- No vision projector was downloaded. Tool calling, vision, concurrency,
  thinking-on behavior, 262K context, and broad model quality were not validated.
- No boot startup was installed and no existing clients were repointed.

### Reproduce and inspect

See [runbook and launch scripts](experiments/qwen38-dflash2/README.md),
[raw request/results](experiments/qwen38-dflash2/results/),
[computed summary](experiments/qwen38-dflash2/results/summary.json), and
[per-run logs and memory evidence](experiments/qwen38-dflash2/evidence/runs/).
`benchmark.py` contains the complete prompts; `summarize.py` regenerates the
summary. Remote artifacts are retained at `~/qwen38-dflash-trial/`.

Primary references: [official target and sidecars](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF),
[original BF16 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2),
and [GGUF drafter usage notes](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF).

## 2026-11-07 02:00 UTC — final 32K validation and live state

**DFlash 2 remains running on spark1** at
**http://192.168.1.147:8002/v1**, served ID **`qwen3.8-flash-next-gguf`**,
**32768 context / one sequence**. LAN health, model discovery, advertised
context and arithmetic were verified independently after testing.
Original vLLM :8001 is stopped and retained; spark2 is untouched.
This native trial has no automatic boot startup; no clients were repointed.

The final deployment reran all five short checks successfully. Its first
short-prompt decode pass was 25.0 / 34.9 / 55.5 tokens/sec; the repeated
pass after context testing was **34.3 / 51.6 / 71.8 tokens/sec**. These
32K measurements are separate from the controlled 8K comparison above.

The retrieval request placed five random hexadecimal values near 1%, 25%,
50%, 75% and 99% of 1396 filler records. The actual prompt was **29987
tokens**, with **0 cached tokens**, leaving room for output within 32K.
All five values were returned exactly:

| Measurement | Result |
|---|---:|
| Exact facts recovered | 5 / 5 |
| Client first output | 57.10 s |
| Client total time | 59.27 s |
| Native prefill | 57.05 s, 525.61 tokens/sec |
| Native decode | 119 tokens, 54.37 tokens/sec |
| Draft acceptance | 92 / 97, 94.85% |
| Minimum MemAvailable during retrieval | 44.19 GiB |
| MemAvailable after validation | 44.57 GiB |

This demonstrates one synthetic long-input retrieval request. It does not
establish general long-context reasoning quality or the former vLLM 262K
capacity. No guard termination, context truncation or native process crash
occurred in these tests.

Evidence: [context result with full response and SSE timings](experiments/qwen38-dflash2/evidence/32768-result.json),
[final 32K repeat](experiments/qwen38-dflash2/results/dflash-32k-repeat.json),
[LAN verification](experiments/qwen38-dflash2/evidence/final-lan-verification.json),
and [final launch/process/container state](experiments/qwen38-dflash2/evidence/final-state.json).
See [current setup](current-setup.md) for the active endpoint and recovery.

## 2026-11-07 08:19 PST — 256K context, 32000 output allowance

At the user's request, the active DFlash server was raised from 32768 to
**262144 total context tokens**, with **32000 output tokens as the server
fallback** (`-n 32000`). The model ID and :8002 endpoint are unchanged.
The original vLLM container remains stopped and retained; spark2 is untouched.
The existing 100 GiB CPU cgroup and global 8 GiB available-memory guard remain.

Input and output share the context window. Reserving 32000 output tokens
leaves **230144 tokens for the prompt, including its chat template**.
The API permits an explicit `max_tokens` to override the server fallback.
`dgxlib/models.yaml` now requests max_tokens 32000 by default, read timeout
3600 seconds and streaming idle timeout 1800 seconds, max concurrency 1.
No consuming application's endpoint was changed.

The 256K server passed all five short arithmetic/copy checks. Then the
large-context test placed the same five exact values near 1%, 25%, 50%,
75%, and 99% of 10536 filler records. **All five facts were recovered**.

| Measurement | Result |
|---|---:|
| Configured total context | 262144 tokens |
| Actual prompt | 224994 tokens, 0 cached |
| Requested output allowance | 32000 tokens |
| Prompt plus requested output budget | 256994 tokens |
| Exact facts recovered | 5 / 5 |
| Client first output | 298.94 s |
| Client total time | 301.98 s |
| Native prefill | 298.74 s, 753.16 tokens/sec |
| Native decode | 124 tokens, 40.38 tokens/sec |
| Draft acceptance | 96 / 97, 98.97% |
| Minimum MemAvailable during retrieval | 36.50 GiB |
| MemAvailable after validation | 36.58 GiB |

The answer ended naturally after 124 tokens. **This validates accepting a
32000-token allowance with a large prompt; it does not test generation of
an entire 32000-token answer.** No context truncation, native process crash,
or memory-guard termination occurred. This remains one synthetic retrieval
test, not a general long-context quality evaluation.

The first short-prompt decode pass after startup measured **26.0 / 38.0 /
55.8 tokens/sec** for guide/code/primes. These are not warm repeats and
should not be compared directly to the earlier warm table.

The launcher command contains `-c 262144 -n 32000`. `/props` reports the
262144 context correctly but lists the per-request prediction sentinel
`-1`; inspected server source selects `params_base.n_predict` when a
request leaves that sentinel unchanged. An explicit request limit takes
priority. LAN health, model discovery and arithmetic with
`max_tokens: 32000` passed after the long test.

**This 256K configuration remains running.** No reboot startup was installed.
Current configuration and recovery are in [current setup](current-setup.md).
Evidence: [long-input results and response](experiments/qwen38-dflash2/evidence/262144-result.json),
[exact prompt](experiments/qwen38-dflash2/evidence/262144-prompt.txt),
[validation log](experiments/qwen38-dflash2/evidence/256k-validation.log),
[launch and memory snapshot](experiments/qwen38-dflash2/evidence/final-state-256k.json),
[LAN verification](experiments/qwen38-dflash2/evidence/final-lan-verification-256k.json),
and [256K short checks](experiments/qwen38-dflash2/results/dflash-256k.json).
