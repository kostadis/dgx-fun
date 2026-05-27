# NVIDIA Nemotron 3 Nano 30B A3B — observations

Date: 2026-05-18
vLLM image: `vllm/vllm-openai:latest` (version 0.20.2 per startup log)
Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
Flags: `--max-model-len 262144 --max-num-seqs 8 --gpu-memory-utilization 0.80 --kv-cache-dtype auto --dtype bfloat16 --trust-remote-code --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser-plugin /plugins/nano_v3_reasoning_parser.py --reasoning-parser nano_v3`
Spin-up: `spin-up-vllm-nemotron3-nano-30b.sh`
Source recipe: [vLLM Recipes — Nemotron-3-Nano-30B-A3B "DGX Spark / Jetson Thor" section](https://docs.vllm.ai/projects/recipes/en/latest/NVIDIA/Nemotron-3-Nano-30B-A3B.html)

Mirrors `gemma4-26b-moe-observations.md` so the two deployments are
comparable section-for-section. Methodology and decision criteria
are in `nemotron3-nano-30b-test-plan.md`.

## Executive summary (Phase A only)

Three findings that landed before any synthetic benchmark ran:

1. **Tuned kernels on GB10.** Unlike Gemma 4 which falls back to
   TRITON MoE + TRITON_ATTN, Nemotron picks **FlashInfer CUTLASS MoE
   + FlashAttention 2** — the fast paths. This is the first model
   we've deployed where NVIDIA's vLLM recipe specifically covers DGX
   Spark / Jetson Thor (GB10).
2. **Long context is structurally cheap.** Hybrid Mamba-2 + 6
   attention layers (of 52 total) means per-token KV cost is **~6 KB
   vs Gemma 4's ~73 KB — 12× cheaper**. The 256K context the recipe
   recommends fits comfortably in 36% of the KV pool at the recipe's
   8-concurrent-sessions design point.
3. **Tool calling works with reasoning enabled.** The open HF
   discussion #3 about tool-call + reasoning being broken does NOT
   reproduce on the current vLLM image — the probe passed with
   `content: null`, populated `tool_calls[]`, and reasoning preserved
   in its own field.

Phase B real-workflow validation (CampaignGenerator, opencode) is
still pending. Until that runs, **Nemotron remains an experimental
swap-in**, not a promoted default. Client configs in
`current-setup.md` §6 still reference Gemma 4 intentionally.

## Hardware baseline

Identical to `gemma4-26b-moe-observations.md`. DGX Spark GB10, 128 GB
unified, 273 GB/s. `nvidia-smi` still returns `[N/A]` for memory.used
on this hardware — must use `free -h` and vLLM's own logs for
accounting.

---

## Phase A — serving behavior

### Startup

- **Wallclock to "Application startup complete"**: ~18 min on first
  run (started 05:24, ready 05:42 per container logs 2026-05-19 UTC).
  Breakdown: download dominated, **`init engine (profile + KV cache +
  warmup) took 73.11 s, compilation 11.98 s`** — once weights were on
  disk the post-load init was just over a minute. Warm restarts on a
  cached HF directory should be substantially faster than Gemma 4's
  ~10 min warm-restart cost.
- **Cold-start cost driver**: HF download. ~60 GB BF16 weights pulled
  unauthenticated initially. We've since plumbed `HF_TOKEN` through
  the spin-up scripts (extracted from `~/.bashrc`) so future cold
  starts hit authenticated rate limits.
- **Resident memory after startup**: `vllm-chat` reserves ~102 GB
  (`--gpu-memory-utilization 0.80`). vLLM reports 60 GB weights +
  34.32 GiB KV pool. Combined with vllm-embed's ~6 GB cap, total
  vLLM allocator reservation is ~108 GB out of 128 GB physical.
- **Pre-launch fix history**: first attempt of the spin-up script
  bailed prematurely on a false-positive error match — the args-dump
  log line legitimately contained the literal word "plugin" and my
  error regex grep was too aggressive. Patched the regex; container
  itself was healthy and proceeded to load normally. The script also
  needed `stdout` unbuffering (`python3 -u`) for live progress
  reporting through `tee` — also patched.
- **HF authentication warning**: present at startup on the first run
  (`Please set a HF_TOKEN to enable higher rate limits`). Plumbed
  through the spin-up scripts post-deploy; will not appear on next
  cold start.

### Architecture detection (from startup logs)

The calibration headline. Compare line-for-line with Gemma 4's
backend selection in `gemma4-26b-moe-observations.md`:

- **MoE backend**: `Using FlashInfer CUTLASS Unquantized MoE backend
  out of potential backends: ['FlashInfer TRTLLM', 'FlashInfer
  CUTLASS', 'TRITON', 'BATCHED_TRITON']`. **The tuned path was
  chosen.** (Gemma 4 got TRITON fallback because no
  `E=128,N=704,device_name=NVIDIA_GB10.json` exists.)
- **Attention backend**: `Using FLASH_ATTN attention backend out of
  potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN',
  'FLEX_ATTENTION']` + `Using FlashAttention version 2`. **FA2 fast
  path.** (Gemma 4 was forced to `TRITON_ATTN` by mixed head dims.)
- **Trust-remote-code path**: `[transformers] A new version of the
  following files was downloaded ... configuration_nemotron_h.py`.
  The custom modelling module loaded cleanly with
  `--trust-remote-code`.
- **Parser plugin load**: no errors. `--reasoning-parser-plugin
  /plugins/nano_v3_reasoning_parser.py` + `--reasoning-parser
  nano_v3` accepted at startup.

### Mamba + MoE shape (from logs)

- **Layer composition** (per the HF model card):
  - 23 Mamba-2 layers — constant per-sequence SSM state, no
    quadratic context cost
  - 23 MoE layers with 128 routed experts + 1 shared, 6 active per
    token
  - 6 attention layers with GQA (2 groups)
- **MoE shape** (logged by vLLM): consistent with the card.
  FlashInfer CUTLASS picked up the tuned config for this expert
  count + intermediate dim on GB10. No "fallback" warnings.
- **Padding**: `WARNING [kv_cache_utils.py:1152] Add 1 padding
  layers, may waste at most 4.35% KV cache memory`. The 6 attention
  layers get padded to 7 for alignment. Loses ~4% of the pool.
  Logged twice (once during profiling, once after KV creation).
  Minor; not actionable.

### KV cache budget (measured by vLLM)

```
WARNING [kv_cache_utils.py:1152] Add 1 padding layers, may waste at most 4.35% KV cache memory
INFO    [gpu_worker.py:440]      Available KV cache memory: 34.32 GiB
INFO    [gpu_worker.py:455]      CUDA graph memory profiling is enabled (default since v0.21.0).
                                 The current --gpu-memory-utilization=0.8000 is equivalent to
                                 --gpu-memory-utilization=0.7924 without CUDA graph memory profiling.
INFO    [kv_cache_utils.py:1708] GPU KV cache size: 5,769,184 tokens
INFO    [kv_cache_utils.py:1709] Maximum concurrency for 262,144 tokens per request: 22.01x
```

Headline numbers:

| | value |
|---|---:|
| KV pool size | **34.32 GiB / 5,769,184 tokens** |
| Per-token KV cost | **~6 KB/token** |
| Operating point (`--max-num-seqs 8` × 256K) | 2,097,152 tokens used (**36% utilisation**) |
| Theoretical concurrency at 256K if uncapped | **22.01×** |
| Effective `--gpu-memory-utilization` after CUDA-graph deduction | 0.7924 (vs nominal 0.8000) |

Comparison to Gemma 4 on the same hardware:

| | Gemma 4 26B MoE @ 128K | Nemotron 3 Nano 30B @ 256K |
|---|---:|---:|
| Attention layers | ~60 | **6** |
| KV pool | ~40.5 GiB | 34.32 GiB |
| Tokens held | ~557K | **5,769,184** |
| Per-token cost | ~73 KB | **~6 KB** (12× cheaper) |
| Concurrent maxed sessions | ~4 (saturated) | **22 theoretical / 8 used** |

This is the architectural confirmation of "long context is cheap on
a Mamba-hybrid" — turned out to be 12× cheaper per token, not 10×.
Means the 1M-context mode (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`) is
actually feasible to run if needed.

### Smoke test (curl)

`curl -sS http://localhost:8001/v1/chat/completions -d '{"model":"...", "messages":[{"role":"user","content":"Reply only with OK."}], "max_tokens": 2048}'`:

```json
"message": {
  "role": "assistant",
  "content": "\nOK",
  "tool_calls": [],
  "reasoning": "Hmm, the user just asked me to reply only with \"OK\" and nothing else. That's very specific and straightforward.\n\nI wonder if they're testing whether I follow instructions precisely, or maybe they need a quick confirmation for some technical reason. The phrasing \"reply only with OK\" feels like they're setting strict boundaries for my response.\n\nSince they're being so clear about wanting just \"OK\", I shouldn't add any extra words or explanations. ..."
},
"usage": {
  "prompt_tokens": 21,
  "completion_tokens": 234,
  "total_tokens": 255
}
```

Two notable observations:

1. **Field naming quirk**: reasoning is emitted under the `reasoning`
   key, NOT the OpenAI-convention `reasoning_content`. Required
   patching `bench-prefill.sh` and `bench-decode.sh` (and any client
   that wants to inspect reasoning) to check both field names.
2. **Reasoning overhead made concrete**: 21 prompt tokens produced
   234 completion tokens, of which 232 were reasoning and 2 were
   content (`"\nOK"`). **A 116:1 reasoning:content ratio** on the
   most trivial possible prompt. Real workloads will have smaller
   ratios (reasoning is bounded by problem complexity, not output
   length) but this is a token-cost multiplier that doesn't exist on
   Gemma 4.

### Tool calling probe (test-gemma4-toolcall.sh, MODEL= override)

Ran the existing Gemma 4 tool-call probe with the model id swapped:

```bash
ssh kostadis@192.168.1.147 \
  'MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 bash ~/test-gemma4-toolcall.sh'
```

**Result: PASS.**

```
finish_reason: tool_calls
content: None
tool_calls count: 1
  [0] name: 'get_weather', arguments: '{"location": "Paris"}'

  content null/empty:    True
  tool_calls populated:  True
  correct function name: True
  arguments parseable:   True
```

Plus a separate `reasoning` field on the message:

> Okay, the user is asking about the current weather in Paris. Let
> me check the tools available. There's a function called
> get_weather that takes a location parameter... I don't see any
> other tools, so this should be straightforward.

**This contradicts the open HF discussion #3** which reports
tool-call + reasoning being broken in some configurations. Either
that bug has been fixed in the current vLLM image, or the
configuration we're running doesn't trigger it. **Watch for it
re-surfacing under multi-turn loads** in Phase B2 (opencode), where
the bug allegedly bites after the first few turns.

Wallclock: 5.05 s for the full tool-call turn. Decomposing: 285
prompt tokens (small prefill, probably ~0.5 s) + 130 completion
tokens (115 reasoning + tool-call structure) at ~23 tok/s ≈ 4.5 s.
**Each tool-call turn under reasoning costs ~5 s wallclock** — for
agentic loops with N tool calls, expect ~5N seconds of reasoning
latency on top of model decode. Compounds in opencode.

### Decode tok/s (organic, from vLLM stats logger)

Captured opportunistically from the vLLM stats-logger output during
the smoke tests and tool-call probe:

```
Engine 000: Avg prompt throughput: 2.1 tokens/s, Avg generation throughput: 23.4 tokens/s
```

**~23 tok/s single-stream decode** — essentially identical to Gemma
4's measured ~23 tok/s. The "decode is bandwidth-bound and dominated
by active params" prediction holds: 3.5B vs 4B active params produce
near-identical decode rates.

(The synthetic `bench-decode.sh` runs against Nemotron weren't
captured cleanly — the result files contained leftover prefill
output. The organic stats-logger reading suffices for Phase A
recording.)

### Prefill calibration synthetic probe — DEFERRED

Ran `bench-prefill.sh` against both Gemma 4 baseline and Nemotron at
input lengths 1K / 8K / 32K / 64K / 100K. **Data was too noisy to
publish a comparison**:

- Gemma 4 baseline: 1024 was a 19s outlier (cold start), then 8K-65K
  were implausibly fast (sub-second TTFTs implying 30,000-70,000
  tok/s prefill — physically impossible on this hardware), then 100K
  jumped back to 35s. Likely cause: cold CUDA-graph compile for the
  first call's batch shape, then subsequent calls hit warm graphs,
  then 100K hit a new shape.
- Nemotron: 1024 was 0.47s (sensible), 8192 was 27.76s (clear
  outlier), then 32K/64K/100K formed a sensible monotonic curve at
  14.35 / 36.34 / 65.49 s. Same first-call JIT effect on a different
  axis.

**Conclusion**: the synthetic probe needs methodology improvements
(warmup pass + multiple samples per length + median) before it
produces clean comparable data. Phase B real-workflow comparisons
will supersede this as the authoritative measurement — wallclock on
a CampaignGenerator session run is direct and noise-averaged across
the full prompt distribution.

The qualitative answer to "does the tuned-kernel path materially beat
fallback?" is already settled by the architecture-detection logs at
the top of this section.

---

## Phase B — pending

### B1. CampaignGenerator (TODO)

Plan: run `session_doc.py` against the same campaign / session /
scene range used for Gemma 4's Phase B1 baseline in
`gemma4-26b-moe-observations.md`. Compare:
- Wallclock per scene
- Total token cost (reasoning + content), expect total substantially
  higher than Gemma 4
- Subjective voice fidelity on 2-3 voice-critical scenes
- **Reasoning leakage into narrative** — known failure mode for
  reasoning models

### B2. opencode (TODO)

Plan: edit `~/.config/opencode/opencode.json` locally (do NOT
commit) to add a `nemotron-3-nano-30b` entry and flip default.
Exercise a representative session against this repo: "explain the
difference between the three Gemma spin-up scripts." Watch:
- Wallclock vs gut-feel for Gemma 4
- Whether think tokens leak into user-visible output
- Whether tool calls stay reliable across multi-turn sessions (the
  HF #3 bug's alleged failure mode)
- Whether reasoning visibly helps multi-step planning or is pure
  latency overhead

---

## Calibration findings (Phase A)

The "rough edges of local hardware" list, Nemotron-specific:

1. **Tuned vendor recipes exist for GB10.** Until Nemotron we had
   only the fallback path. Now we know the gap can be closed when
   NVIDIA bothers to ship a recipe — and that the resulting
   deployment lands two-for-two on tuned kernels (MoE + attention).
2. **`--reasoning-parser-plugin` requires a host-side file mount.**
   The plugin Python file (`nano_v3_reasoning_parser.py`) lives on
   the host at `~/vllm-plugins/` and gets mounted into the container
   at `/plugins:ro`. This is a different deployment pattern from
   built-in parsers — worth knowing if we encounter other custom
   reasoning models.
3. **vLLM emits reasoning under non-standard field names.** The
   `nano_v3` parser uses `reasoning`, not OpenAI's
   `reasoning_content`. Clients that introspect reasoning need
   defensive coding (check both).
4. **Reasoning models change the smoke-test convention.** A
   `max_tokens=10` smoke test will get an empty `content` because
   the think phase eats the budget. Floor for reasoning-model smoke
   tests is `max_tokens=2048` minimum.
5. **Per-tool-call latency budget grows under reasoning.** ~5 s
   per turn even on a one-shot weather lookup. Agentic loops
   amplify this multiplicatively.
6. **The hybrid Mamba-Transformer KV math is real.** ~6 KB per token
   vs Gemma 4's ~73 KB. Long context is genuinely cheap, not just
   "less bad."
7. **HF unauthenticated rate limits dominate first-cold-start
   time.** Plumbing `HF_TOKEN` from `~/.bashrc` through the spin-up
   scripts (via `eval "$(grep ... .bashrc)"` + `-e HF_TOKEN=...`) is
   a one-time fix that meaningfully reduces future cold-start cost.

---

## Open questions for Phase B

- **Does reasoning help or hurt on real D&D narration workloads?**
  Hypothesis: helps on multi-character scenes where voice
  consistency matters, hurts on simple scene transitions where it's
  pure overhead.
- **Does the long-context advantage translate to opencode session
  feel?** With ~6 KB per token, a fully-loaded 256K opencode session
  costs ~1.5 GiB of KV — trivial. Gemma 4 at 128K cost ~9 GiB. Does
  opencode actually feel faster across a long session, or is decode
  parity the dominant feel?
- **Multi-turn tool-call stability.** Does HF #3's "fails after the
  first few turns" claim reproduce when opencode does 10+ tool calls
  in a session?
- **Reasoning leakage into prose.** If the model breaks the fourth
  wall mid-narration, that's a hard rejection criterion.
- **Promote, keep-as-swap, or revert?** Depends on B1/B2 outcomes
  vs the decision criteria in `nemotron3-nano-30b-test-plan.md`.

---

## Phase B verdict — reject (2026-05-19)

After real-workflow use over 2026-05-18 → 2026-05-19, Nemotron 3 Nano
30B A3B is **rejected** as the `vllm-chat` default. Reverted to Gemma 4
26B MoE longctx on 2026-05-19 via `bash
~/spin-up-vllm-gemma4-26b-moe-longctx.sh`.

**Primary failure mode**: **llm_wiki client incompatibility with
reasoning traces.** llm_wiki (Tauri app on Windows) has no parser for
the `<think>...</think>` blocks Nemotron emits. The vLLM server side
worked fine — the `--reasoning-parser nano_v3` plugin correctly splits
reasoning into the `reasoning` response field on the chat completions
API — but llm_wiki reads `content` and renders it directly, so the chat
box filled with raw thinking output. There is no server-side flag to
disable reasoning on Nemotron 3 Nano; it's how the model is trained.

This is a **client-compatibility failure, not a model-quality
failure**. The hybrid Mamba-2 + MoE architecture, FlashAttention v2
backend, 5.77M-token KV pool, and tool-call parity (HF #3 did not
reproduce) all remained intact. The deployment is technically sound;
the workflow it needed to serve isn't.

### What this means for re-trying Nemotron

Don't re-promote Nemotron without one of:

1. **Adding `<think>` stripping in llm_wiki** (probably a regex pass in
   the Tauri client before render). This is the cleanest fix because it
   makes llm_wiki tolerant of *any* thinking-by-default model
   (DeepSeek-R1, QwQ, etc.), not just Nemotron 3.
2. **A vLLM-side flag or model variant that suppresses
   `<think>` generation entirely.** None known to exist for Nemotron 3
   Nano as of this writing — the reasoning is bound into the model
   weights, not an inference-time option.
3. **Restricting Nemotron to clients that already handle reasoning**
   (opencode does — handles the `reasoning` field; Anthropic / OpenAI
   clients do). Not viable if llm_wiki is part of the workflow.

### What stays from the experiment

- `nemotron3-nano-30b` model entry in `~/.config/opencode/opencode.json`
  is preserved as an alternate (not default). opencode handles the
  reasoning field cleanly.
- `~/spin-up-vllm-nemotron3-nano-30b.sh` stays on the Spark; not
  checked into the repo since the experiment concluded reject. If you
  want to re-try, the script + this observations file + the test plan
  are enough to rebuild.
- `current-setup.md` top-of-doc Nemotron note documents the rejection
  for future readers.
- All Phase A measurements above (kernel selection, KV pool sizing,
  tok/s, tool-call probe) remain valid characterisation work even
  though the model didn't ship.

### What the experiment was good for

Calibration: Nemotron is the first reasoning-by-default model tried on
this Spark, and the failure surfaces a real workflow constraint —
*reasoning verbosity is a client-side compatibility axis, not just a
token-cost concern*. Future model selection on this hardware should
filter for client compatibility before perf characterisation. This
matches the global "Local AI Hardware Exploration" doctrine: the
exercise is calibration, not optimisation, and a rejected experiment
that clarifies a constraint is a successful exercise.

---

## Related files

- `current-setup.md` — live deployment state (Gemma 4 26B MoE longctx
  re-installed 2026-05-19 after Nemotron rejection).
- `nemotron3-nano-30b-test-plan.md` — methodology + decision
  criteria for this experiment.
- `~/spin-up-vllm-nemotron3-nano-30b.sh` — deployment script (lives
  on the Spark only after experiment concluded; not in repo. Auto
  picks up `HF_TOKEN` from `~/.bashrc`).
- `bench-prefill.sh` / `bench-decode.sh` — synthetic probes (used in
  Phase A, deferred for Phase B in favour of real workflow
  wallclocks).
- `gemma4-26b-moe-observations.md` — Phase A/B record for the model
  Nemotron replaces. Source of baseline numbers.
- `model-comparisons.md` — high-level family-vs-family comparison;
  updated with a Nemotron 3 family entry.
