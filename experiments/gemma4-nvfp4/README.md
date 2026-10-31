# Gemma 4 26B-A4B: 4-bit serving on one Spark (vLLM NVFP4 vs llama.cpp QAT Q4_0)

Run 2026-10-24 on spark2 (`192.168.1.121`) while the two-box DeepSeek
deployment was stopped. Same model (Gemma 4 26B-A4B-it, 3.8B active MoE), two
4-bit weight sets on two serving stacks, measured by the same benchmark. The
running narrative is in `../../gemma4-26b-moe-observations.md` (2026-10-24
sections); this file summarizes the experiment and points at the evidence.

## Question

The 2026-05 BF16 run of this model decoded at 23.3 tok/s single stream. A
2026-05-04 XRPL Commons article reported 51 tok/s single stream and 271 tok/s
at 8 streams with a community NVFP4 checkpoint on vLLM. Does that reproduce on
our stock image, and does Google's own quantization-aware-trained Q4_0 GGUF on
llama.cpp do better or worse?

## Configurations

| | Run A: vLLM NVFP4 | Run B: llama.cpp QAT Q4_0 |
|---|---|---|
| Weights | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` (15.87 GiB loaded) | `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, `gemma-4-26B_q4_0-it.gguf` (14.4 GB) |
| Server | stock `vllm/vllm-openai:v0.22.0-aarch64`, no patched `gemma4.py` | llama.cpp master `f3269ef`, built on box, CUDA arch 121 |
| Key flags | `--quantization modelopt --moe-backend marlin --kv-cache-dtype fp8`, APC, 131K ctx, seqs 16, util 0.80, `gemma4` tool + reasoning parsers | `-ngl 999 -fa on -c 131072 -np 8 --kv-unified --jinja`, text only (no mmproj) |
| Served id | `google/gemma-4-26B-A4B-it` | `google/gemma-4-26B-A4B-it-qat` |
| Script | `../../spin-up-vllm-gemma4-26b-nvfp4.sh` | `../../spin-up-llamacpp-gemma4-26b-qat.sh` |
| Bench start | 2026-10-24 05:20:27 UTC | 2026-10-24 18:43:56 UTC |
| Raw results | `results-20261024.json` | `results-llamacpp-qat-20261024.json` |

Google's `-qat-q4_0-unquantized` checkpoint was considered and rejected: its
config carries BF16 weights (51.6 GB, no quantization config), so vLLM would
serve it at the May BF16 speed.

## Method

`bench.py` (stdlib only, run on the box against `localhost:8001`), temperature
0, thinking off except in the thinking probe:

- **decode**: one stream, tiny prompt, 1000-token generation.
- **concurrency**: 1 / 4 / 8 streams of 500 tokens each, *identical prompts*.
- **prefill**: cold TTFT at ~9K / 36K / 113K prompt tokens, a unique nonce per
  prompt so prefix caching cannot fake a cold number, then a warm resend.
- **tools**: one forced-relevant `get_weather` call.
- **think**: the same question with reasoning on and off; records which response
  field carries the trace.

Rerun: `python3 -I bench.py http://localhost:8001/v1 <served-id> > results.json`.
The script falls back to counting stream chunks when the server omits usage.

## Results

| | Run A: vLLM NVFP4 | Run B: llama.cpp QAT Q4_0 | May BF16 (vLLM) |
|---|---:|---:|---:|
| Decode, 1 stream (1000 tok) | 53.5 tok/s | **74.1 tok/s** | 23.3 tok/s |
| Aggregate @ 1 / 4 / 8 streams | 53.9 / 174.4 / **298.5** | 74.6 / 173.8 / 234.5 | — |
| Per stream @ 8 | 37.3 | 29.3 | — |
| Cold TTFT 9.1K (prefill tok/s) | **1.83 s** (4,971) | 4.63 s (1,966)* | — |
| Cold TTFT 36K (prefill tok/s) | 14.52 s (2,500) | 14.12 s (2,570) | — |
| Cold TTFT 113K (prefill tok/s) | 109.95 s (1,031) | **61.25 s** (1,851) | — |
| Warm TTFT 9K / 36K / 113K | 0.08 / 0.14 / 0.38 s | 0.12 / 0.20 / 0.44 s | — |
| Tool call | ok, `{"city": "Lisbon"}`, 0.36 s | ok, `{"city":"Lisbon"}`, 0.57 s | — |
| Thinking on / off | 302 tok, 5.63 s / 6 tok, 0.16 s | 346 tok, 4.73 s / 6 tok, 0.19 s | — |
| Reasoning field | `reasoning` | `reasoning_content` | — |
| Host RAM available under load | ~19 GB | ~93 GB | — |
| Time to ready | ~5 min (109 s weight load) | seconds after load | — |

\* First long prompt after boot; slower than the 36K cell, so warm-up is likely
included. Treat it as unverified until rerun.

Both answers in the thinking probe were correct ("$0.05").

## Findings

1. **The article's patch is no longer needed.** vLLM 0.22.0 maps the NVFP4
   scale keys to FusedMoE upstream; the checkpoint loaded cleanly on the stock
   image.
2. **NVFP4 runs weight-only on GB10.** vLLM selects Marlin and logs that the
   GPU lacks native FP4 compute, so the W4A4 checkpoint runs as W4A16. The
   gain is bandwidth, not FP4 tensor cores.
3. **4-bit roughly triples decode over BF16.** 23.3 → 53.5 (vLLM, 2.3×) →
   74.1 tok/s (llama.cpp, 3.2×). llama.cpp's int4 group-32 GGML kernels beat
   vLLM's Marlin FP4 dequant by 39% single stream.
4. **vLLM wins batching.** Tied at 4 streams; vLLM leads by 27% at 8 and beat
   the article's 271 tok/s. Caveat: identical prompts.
5. **vLLM has a long-context prefill cliff; llama.cpp does not.** vLLM falls
   from 4,971 to 1,031 tok/s between 9K and 113K, while llama.cpp holds ~1,850–
   2,570 tok/s and is 1.8× faster at 113K. The suspected cause is vLLM's forced
   TRITON_ATTN on Gemma's mixed 256/512 head dims. **Not profiled.**
6. **APC makes resends nearly free on both.** 113K warm TTFT is 0.38–0.44 s.
7. **vLLM's KV pool is huge:** 6,194,430 tokens = 47× at 131K (sliding-window
   local layers plus fp8 KV). Context length is not the limit for this model.
8. **Footprint matters on unified memory.** llama.cpp leaves ~93 GB for the
   host and could co-host a second model; vLLM at util 0.80 leaves ~19 GB.
9. **Reasoning field differs by stack.** vLLM emits `reasoning`, which opencode
   and openclaw drop (same issue as `todo_nano_v3_reasoning_leak`). llama.cpp
   emits `reasoning_content`, which they read.

## Limits

- **No quality evaluation.** No comparison of community NVFP4 against Google
  QAT weights, or of either against BF16. That is the main open question behind
  choosing QAT.
- Concurrency used identical prompts; a mixed-prompt probe may differ.
- One run per cell; the llama.cpp 9K prefill cell is suspect (see above).
- Text only; vision was not loaded or tested on llama.cpp.
- The llama.cpp server ran as a nohup host process, with no restart policy.

## Disposition

- `gemma4-nvfp4` (vLLM) container: `docker stop`ped and retained on spark2.
- llama.cpp build stays at `spark2:~/llama.cpp-gemma4`; GGUF stays in the HF
  cache. Log was `spark2:~/llamacpp-gemma4.log`.
- 2026-10-26: spark2 returned to `qwen3.8-flash-next` (`qwen38-flash`).
  Restart Run B with `../../spin-up-llamacpp-gemma4-26b-qat.sh` after stopping
  `qwen38-flash`; the two do not share port 8001.
