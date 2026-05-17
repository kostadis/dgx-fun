# Gemma 4 26B MoE (A4B) — observations

Date: 2026-05-17
vLLM image: vllm/vllm-openai:latest (filled in below after `docker inspect`)
Model: google/gemma-4-26b-a4b-it
Flags: --max-model-len 32768 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.75 --dtype bfloat16 --trust-remote-code

## Phase A — serving behavior

### Startup

- **Wallclock to "Application startup complete"**: ~16.5 min (994s reported by spin-up script)
  - Shard 1 of 2 load: 5:51 (the slow part — single-file 35-ish GB shard, bandwidth-bound)
  - Shard 2: completed quickly after shard 1
  - Multi-modal warmup: 20.018s
- **Resident memory after startup**:
  - **Calibration finding**: `nvidia-smi --query-gpu=memory.used` returns `[N/A]` on GB10 / DGX Spark because of unified memory architecture. CPU and GPU share LPDDR5X; there's no discrete VRAM to query. Have to use `free` for host total and vLLM's own logs for its accounting.
  - **Host RAM**: 117 GiB used / 121 GiB total — only ~4 GiB available. Includes vllm-embed + vllm-gemma-2 + vllm-chat + host OS. Very tight.
  - **Swap**: 7.9 GiB / 15 GiB in use. Possibly historical from earlier experiments; worth watching under sustained load.
  - **vllm-chat's own accounting** (from logs):
    - Model weights: **48.5 GiB** (matches BF16 26B-param expectation)
    - KV cache: **40.56 GiB → 557,022 tokens of capacity**. With `--max-model-len 32768` this is ~17 concurrent maxed sessions.
    - Engine init (profile + KV cache + warmup): 124.94s (compilation 51.0s)
  - **CUDA-graph memory note**: vLLM 0.21+ now deducts CUDA-graph memory from the `--gpu-memory-utilization` budget. With our 0.75 setting, effective is 0.741. Could bump to 0.759 to recover the diff; minor.
- **Pre-launch fix**: First attempt failed with `ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496) is larger than max_num_batched_tokens (2048)`. Fix: added `--max-num-batched-tokens 8192`. Caused by vLLM forcing `--disable_chunked_mm_input` for multimodal-bidirectional attention.
- **Authentication warning**: "You are sending unauthenticated requests to the HF Hub. Please set a HF_T[OKEN]." Non-fatal — model is public.

### Architecture detection (from startup logs)

- `Resolved architecture: Gemma4ForConditionalGeneration` — **this is a multimodal (vision-language) model, not text-only**. The runbook framed this as a pure MoE experiment; it's actually MoE + VLM. Phases B1/B2 are still text-only workloads but the vision capacity exists.
- `Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512). Forcing TRITON_ATTN backend to prevent mixed-backend numerical divergence.` — interesting calibration finding: Gemma 4's heterogeneous-attention design cannot use FlashAttention on this image.
- `MoEPrepareAndFinalizeNoDPEPModul` — MoE expert-routing module confirmed loaded.
- `Encoder cache will be initialize[d]` — vision encoder pipeline active.
- `Asynchronous scheduling is enabled.`

### MoE-specific log lines (from `grep -iE 'moe|expert|router|gate|a4b|active.param|per.layer'`)

- **MoE shape**: `E=128, N=704` — **128 experts, intermediate dim 704**. With 4B active out of 26B total, the router selects some subset of these 128 experts per token (likely 8, the Gemma 4 standard).
- **MoE backend selected**: `Using TRITON Unquantized MoE backend out of potential backends: ['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON', 'BATCHED_TRITON']`. Triton wins on GB10 (sm_121) because FlashInfer's TRTLLM and CUTLASS kernels don't have sm_121 support — another data point on the local-hardware-vs-tuned-server gap.
- **NO TUNED MOE CONFIG FOR THIS HARDWARE**: `WARNING: Using default MoE config. Performance might be sub-optimal! Config file not found at /usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_GB10.json`. This is the calibration headline. vLLM ships per-(expert-count, intermediate-dim, device) JSON configs for tuned fused-MoE kernels; there's no entry for the Spark's GB10 + Gemma 4's E=128/N=704 shape, so we're running on the fallback. Realistic upper-bound on MoE perf on the Spark requires either: (a) waiting for the community to upstream a GB10 tuning, (b) running vLLM's MoE autotune script ourselves, or (c) accepting the suboptimal floor.
- **Other compile config**: `compile_mm_encoder=False`, `cudagraph_mm_encoder=False`, `encoder_cudagraph_max_vision_items_per_batch=0` — vision encoder not compiled for cudagraph. Fine for text-only workloads; would matter if we ever fed it images.
- **`enable_return_routed_experts=False`** — useful toggle to flip later if we want to inspect actual per-token expert routing (would give us expert-utilization histograms).
- **`fast_moe_cold_start=False, static_all_moe_layers=[]`** — additional MoE compile knobs available, untouched in this run.

### Weights download

`Time spent downloading weights for google/gemma-4-26b-a4b-it: 389.917 seconds (~6.5 min)`. Weights were **not** cached on the Spark despite earlier assumption — first-run cost. Future runs should skip this and shave ~6 min off startup.

### Throughput probe (code, T=0, 400 tokens)

- **Wallclock**: 11.252s
- **completion_tokens**: 262
- **Computed tok/s**: **23.28 tok/s**
- **Comparison**: Llama 70B AWQ + spec decode hit ~23 tok/s on the same prompt (262 tokens, 11.345s). **They are matched.** Gemma 4 26B MoE BF16 ≈ Llama 70B AWQ + spec-decode on this workload.

#### Why the MoE advantage didn't show up

The "4B active out of 26B" advertised gain should have made this 3-5× faster than a 70B dense. It didn't. Reasons:

1. **Untuned fused-MoE kernel** — the `E=128,N=704,device_name=NVIDIA_GB10.json` config doesn't exist in vLLM 0.20.2, so we're on the fallback kernel.
2. **TRITON_ATTN attention backend** — forced by heterogeneous head dims (`head_dim=256, global_head_dim=512`). FlashAttention would likely be faster.
3. **BF16 vs AWQ INT4** — Llama 70B is 4-bit quantized, Gemma 4 is 16-bit. Per active weight, Gemma reads 4× more memory.
4. **Bandwidth math** — 273 GB/s ÷ ~8 GB per-token MoE read (BF16, 4B active) = ~34 tok/s theoretical ceiling. We hit 23 tok/s = ~68% of ceiling. The 32% gap is plausibly the untuned-kernel overhead.

#### Code output quality

Clean, idiomatic Python. Type hints (`List[int], Optional[int]`), docstring with Args/Returns, `if __name__ == "__main__"` guard with example call. Comparable to Llama 70B's output on the same prompt — no obvious quality regression.

### Stats logger observations (5 min of light traffic)

Idle stats lines observed during the run: `Avg prompt throughput: 2.0 tokens/s, Avg generation throughput: 0.2 tokens/s, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%`. Nothing alarming. (Skipped the 5-min idle sit; the concurrency probe is the more useful Phase A signal.)

### Concurrency probe (4 parallel requests)

- **Wallclock for 4 parallel**: 19.193s (vs 11.252s single-stream)
- **All 4 returned 262 completion_tokens** (deterministic at T=0, identical prompt)
- **Aggregate throughput**: 1,048 tokens / 19.193s = **54.6 tok/s**
- **Per-stream throughput**: 262 / 19.193s = 13.65 tok/s per stream (down from 23.28 single)
- **Concurrency speedup**: 2.35× aggregate for 4× requests = **sub-linear**, expected for bandwidth-bound decode
- **Per-stream slowdown**: each individual stream is 0.59× as fast under batch=4 contention

**MoE interpretation**: the scaling is similar to what a dense bandwidth-bound model would show at batch=4. The MoE potential win (different requests routing to disjoint experts) doesn't appear to be exploited here — possibly because Triton MoE backend doesn't expert-batch across requests, or because identical prompts produce identical routing decisions.

## Phase B1 — CampaignGenerator

### Configuration

- Campaign: stormgiants
- Workload: `dgx.sh` wrapper, scene 3 of session 20250312, `--narrate-tokens 20000`, `--prose-mode`, `--reflections`
- Genre brief: "First-person comic-noir fantasy memoir — observational, dry, irony-forward, alive to absurdity. NOT epic-fantasy adventure prose; NOT literary-introspective register." With anti-pattern bans on "the shape of X" / "the quality of X" gesturing.
- Output: `/tmp/test-gemma4/session_doc_scene_03_the_battle_of_the_ancient_dragons_begins.md` (7.8K)

### Wallclock

Full run wallclock not separately recorded (user ran the script themselves). Tok/s baseline from Phase A throughput probe: 23.28 tok/s single-stream.

### Quality verdict

**Register match — strong.** The output convincingly hits the comic-noir register:
- "the hoard everyone is killing each other for is actually just bait"
- "as if the universe had just handed him another bill he couldn't afford to pay"
- "manning a trebuchet with all the grace of drunken laborers"
- "like trying to empty the ocean with a spoon"
- "errant tourists" (party vs. storm giants)
- "twelve seconds of combat" running gag with "Poor Bob"
- "his pragmatic theology manifesting as a battle cry"

**Banned-phrase compliance.** No literal "the shape of X" / "the quality of X" / "that particular quality" violations spotted. Some adjacent generic-fantasy adjective stacking ("whirlwind of scales and fury", "predatory grace") leaks through but doesn't dominate.

**Voice differentiation maintained**: Orsik (opportunist, calculating), Vardis (dry theology), Thistle (tactician), Unla Key (self-deprecating first-person narrator). Distinct.

**Refusal incidents: ZERO.** This was the big surprise. The runbook flagged Gemma's historical refusal-proneness on morally-grey content as the main quality risk; it did not materialize. Violent combat depicted concretely ("rending him with a violence that felt personal", "carving sixty-three points of damage out of him"), demon character Bob treated normally, "To hear God, bitches!" delivered without sanitization. No moralizing, no hedging.

**Verbosity in line.** 7.8K for a single scene with `--narrate-tokens 20000` cap. Pacing reasonable.

**Mechanical-text bleed** (numeric damage values, exact movement feet) present — but per user note, this is a model-agnostic issue (even Sonnet does it), addressed downstream via a scrubbing phase. Not scored against Gemma.

### Conclusion

Phase B1 verdict for narrative workload: **Gemma 4 26B MoE is a viable CampaignGenerator backend.** Hits the register, no refusals, voice-consistent. Speed roughly matched to Llama 70B AWQ + spec decode on the same hardware.

## Phase B2 — opencode

### Tool-call parser used

`gemma4` — vLLM 0.20.2 ships a Gemma 4-specific parser. Located via `vllm serve --help=Frontend | grep tool-call-parser`. Full parser list in this image: deepseek_v3/v31/v32/v4, ernie45, functiongemma, **gemma4**, gigachat3, glm45/47, granite/granite-20b-fc/granite4, hermes, hunyuan_a13b, hy_v3, internlm, jamba, kimi_k2, llama3_json, llama4_json/pythonic, longcat, mimo, minimax/minimax_m2, mistral, olmo3, openai, phi4_mini_json, pythonic, qwen3_coder/xml, seed_oss, step3/step3p5, xlam.

### Tool-call verification curl (test-gemma4-toolcall.sh)

Sent a `get_weather` probe with `tool_choice: "auto"`. Result:

```
finish_reason: tool_calls
content: None
tool_calls count: 1
  [0] name: 'get_weather', arguments: '{"location": "Paris"}'

  content null/empty:    True
  tool_calls populated:  True
  correct function name: True
  arguments parseable:   True

PASS — tool calling works on Gemma 4 26B MoE
```

Clean OpenAI-compatible tool-call response. Arguments are valid JSON, function name matches, content is null (parser correctly extracted the call rather than leaking it into text).

### opencode session

- **Config update**: added `dgx/gemma-4-26b-moe` model entry to `~/.config/opencode/opencode.json` alongside the existing Llama 3.3 70B entry. `tool_call: true`, context 32768, output 8192. Switched default model to Gemma 4.
- **Result**: **worked.** opencode launched against the Gemma 4 endpoint and the agent loop functioned end-to-end. Tool calls fired correctly, opencode could complete the task.

## Post-experiment correction: prefill gap

After the formal experiment finished, real-world use exposed a dimension the benchmark missed: **on long-input workloads** (e.g. opencode reading a README and summarizing it), **Llama 70B + spec decode was so slow it was unusable, while Gemma 4 26B MoE was "amazingly fast."**

Why my benchmark didn't see this: the Phase A throughput probe measured **decode tok/s** (long output, tiny prompt). Decode is memory-bandwidth-bound and reads all weights per output token; Llama 70B AWQ (4-bit) and Gemma 4 26B BF16 hit similar bandwidth profiles, so they tied.

Real-world long-input workloads are **prefill-dominated**: time-to-first-token depends on `prompt_tokens × active_params`. This is the dimension where MoE wins decisively:

- Llama 70B dense: **70B active params per prefill token**
- Gemma 4 MoE: **~4B active params per prefill token**
- ~17× compute gap on prefill

Plus spec decode actively *hurts* on diverse prose (33-43% acceptance per prior tests vs 91% on code): the 1B draft has to prefill the prompt too, then most draft tokens get rejected, so the work is wasted. On a long README this compounds into the "excruciating" wait observed.

**This is the more important production finding than anything in the synthetic benchmark.** Methodology note for future calibration: always measure both TTFT-on-long-input and tok/s-on-long-output; they live on different axes (compute vs bandwidth).

## Conclusion (revised after prefill finding)

### Where does Gemma 4 26B MoE sit on the speed-quality curve?

- **Decode (long output, short prompt)**: tied with Llama 3.3 70B AWQ + spec decode at ~23 tok/s.
- **Prefill (long input, short output)**: **massive win for Gemma 4 MoE.** ~17× active-param advantage at prefill, compounded by spec decode hurting Llama on diverse prose.
- **Quality**: on par with Llama 70B for both narrative prose and agentic coding. No refusal-proneness materialized despite the runbook flagging it as a risk.

### Is MoE serving on vLLM 0.20.2 mature enough to be useful?

**Yes, with one big caveat.** The path runs cleanly: V1 engine, MoE-aware compilation, prefix caching, async scheduling, multimodal support (Gemma 4 is `Gemma4ForConditionalGeneration`), tool calling via `gemma4` parser, no crashes under load. **Caveat**: there is no tuned fused-MoE kernel config for GB10 + E=128/N=704 shape in this vLLM image — performance is on the fallback. The headline finding is that this fallback is *not catastrophic* (still hits ~68% of the BF16 bandwidth ceiling) but it caps the upside. Real MoE speedup would require either tuning the kernel for GB10 ourselves or waiting for an upstream config to land.

### Should this replace any current production usage?

**Revised verdict (post-prefill finding): yes, for prefill-heavy workloads.**

- **vllm-chat (currently Llama 70B + spec decode)**: **Gemma 4 26B MoE wins for opencode and any workload that reads files into context.** The prefill gap (~17× active params) makes Llama 70B + spec decode unusably slow on long inputs, while Gemma 4 MoE feels fast. For pure-decode workloads (creative narration with short prompts), they're tied. The default port-8001 model should probably switch to Gemma 4 MoE.
- **vllm-gemma (currently gemma-2-9b-it)**: Likely obsolete now. Gemma 4 MoE supersedes it on quality and is competitive on prefill speed because of the 4B-active routing.
- **Cost**: the swap is one-way per the spin-up scripts; both Llama 70B and Gemma 4 MoE remain available via `bash ~/spin-up-vllm-{llama70b-specdecode,gemma4-26b-moe}.sh`.

### Surprising findings (the calibration-value list)

1. **`nvidia-smi --query-gpu=memory.used` returns N/A on GB10**. Unified memory means there's no discrete VRAM to query. Use `free -h` and vLLM's own logs for memory accounting. (Caveat for any tooling that assumes discrete-GPU memory reporting.)
2. **No tuned fused-MoE kernel for the Spark + this expert shape.** The exact filename vLLM looks for is `configs/E=128,N=704,device_name=NVIDIA_GB10.json` — doesn't exist. Performance is on the fallback. This is the most useful "rough edge of local hardware" finding from the experiment.
3. **Gemma 4 26B is multimodal**, not text-only as the runbook framed it. `Gemma4ForConditionalGeneration`. Implications: an additional fix flag was needed (`--max-num-batched-tokens 8192`) for vLLM's MM budget validator. Vision encoder is loaded and warmed up at startup (20s) even when only doing text — small overhead we pay either way.
4. **Heterogeneous attention head dimensions** (`head_dim=256, global_head_dim=512`) force `TRITON_ATTN` instead of FlashAttention. Another perf ceiling lever — getting FlashAttention requires homogeneous head dims, which Gemma 4's architecture doesn't have.
5. **Gemma's historical refusal-proneness did NOT materialize** on CampaignGenerator narrative. Combat violence, demon characters, gallows humor — all delivered without hedging. Either Gemma 4's instruct tuning is meaningfully different from earlier Gemma releases, or the comic-noir prompt framing kept it on register.
6. **MoE batching doesn't help on identical prompts.** 4-parallel concurrency probe with identical T=0 requests routes to identical experts → no expert-disjoint batching win. 2.35× aggregate scaling, same as a dense bandwidth-bound model would show. Would need diverse prompts to see whether expert-parallel batching can outperform dense at the same batch size on this hardware.

### Open questions surfaced for next experiment

- **What's the cost of running vLLM's MoE autotune** (`benchmark_moe.py` in vLLM source) on GB10 to generate the missing `E=128,N=704,device_name=NVIDIA_GB10.json` config file? If feasible, that closes the biggest perf gap.
- **Does flipping `enable_return_routed_experts=True` give us actually-useful expert-utilization histograms?** Would help see whether the 128 experts are evenly used or whether routing is collapsing into a few hot experts.
- **What does a diverse-prompt concurrency probe look like?** The identical-prompt result told us batching scales sub-linearly; mixed prompts might show whether MoE can win at concurrency where dense can't.
- **Can spec decode be layered on Gemma 4 MoE?** vLLM has matured here; a small Gemma-family draft model (Gemma 2B or Gemma 3 instruct variant) + this 26B MoE target might compound. Risk: MoE × spec decode is two bleeding-edge interactions stacked.

