# DGX Spark calibration report — May 2026

A synthesis of the recent calibration sweep on the DGX Spark: speculative
decoding, agentic tool calling, and a Gemma 4 26B MoE evaluation. The
goal is not optimization — Anthropic API beats every result here on raw
quality and speed — the goal is **calibration**: understanding what
local-AI serving actually feels like on this specific hardware, where
the rough edges are, and which workloads make local models genuinely
competitive.

## Executive summary

Three experiments, in order:

1. **Speculative decoding** on Llama 3.3 70B Instruct AWQ + Llama 3.2 1B
   draft. Worked. ~2× decode speedup on code (91.6% draft acceptance,
   23 tok/s). Hurts on diverse prose (33-43% acceptance, 7-9 tok/s).
2. **Tool calling for opencode** on Llama 3.3 70B with `--tool-call-parser
   llama3_json`. Worked. Verified via `get_weather` probe; agent loop
   functions end-to-end.
3. **Gemma 4 26B MoE (A4B)** at BF16 on vLLM 0.20.2. Worked. ~23 tok/s
   decode (tied with Llama+spec-decode), narrative quality on par with
   Llama 70B, tool calling clean via `--tool-call-parser gemma4`. The
   really interesting finding came *after* the formal experiment: on
   the user's actual workloads (long-input reads), Gemma 4 MoE feels
   "amazingly fast" while Llama 70B + spec decode feels "excruciatingly
   slow" — because prefill is dominated by active params (4B vs 70B),
   not by decode tok/s.

**Headline calibration finding**: synthetic decode-tok/s benchmarks
hide the production gap. For read-heavy workloads (opencode, RAG,
session-prep), the active-params-per-prefill-token axis matters more
than steady-state tok/s. MoE wins decisively where dense+spec-decode
ties. Spec decode is the wrong optimization for prefill-heavy work.

**Production recommendation**: keep `vllm-chat` on Gemma 4 26B MoE as
the default. Toggle to Llama 70B + spec decode only for short-input,
long-code-output workloads where its 91% draft acceptance pays off.

## Hardware baseline

DGX Spark (NVIDIA GB10):
- Unified memory: 128 GB LPDDR5X, ~121 GiB usable to userspace
- Bandwidth: 273 GB/s (shared CPU+GPU)
- Compute capability: sm_121 (not officially supported by mainline vLLM
  builds; `vllm/vllm-openai:latest` Docker image works empirically)
- No discrete VRAM — `nvidia-smi --query-gpu=memory.used` returns `[N/A]`
- KV cache, model weights, and host RAM all draw from the same 121 GiB
  pool. Memory accounting requires `free -h` plus vLLM's internal logs.

Existing serving topology (before this sweep):
- `vllm-embed` (port 8000, ~6 GB) — nomic-embed-text-v1.5
- `vllm-chat` (port 8001, ~52 GB at start of sweep) — Qwen 2.5-14B AWQ
- `vllm-gemma` (port 8002, ~19 GB) — gemma-2-9b-it
- Ollama, accessible separately

All vLLM containers bind-mount `~/.cache/huggingface:/root/.cache/huggingface`.

## Experiment 1: Speculative decoding (Llama 3.3 70B AWQ + Llama 3.2 1B draft)

### Motivation

Autoregressive decoding is memory-bandwidth-bound, so a forward pass that
verifies K draft tokens costs roughly the same as one that produces 1.
A small draft model proposes K tokens, the big model scores them in a
single pass, accepted prefixes are kept, first disagreement is the big
model's pick. Output distribution exactly matches the target — no quality
loss. The win is real when draft acceptance is high.

### Setup

- Target: `casperhansen/llama-3.3-70b-instruct-awq` (4-bit AWQ, ~40 GB)
- Draft: `unsloth/Llama-3.2-1B-Instruct` (~2 GB)
- `--speculative-config '{"model": "unsloth/Llama-3.2-1B-Instruct",
  "num_speculative_tokens": 5}'`
- `--max-model-len 65536`, `--gpu-memory-utilization 0.65`
- Container: `vllm-chat` on port 8001 (replaced the Qwen 14B baseline)

Script: `/home/kroussos/src/dgx/spin-up-vllm-llama70b-specdecode.sh`.

### Results

Binary search prompt, T=0.0, 400 max tokens, identical to the Gemma 4
benchmark used later for apples-to-apples comparison:

- **23 tok/s** sustained generation
- **262 completion tokens in 11.345s**
- **Draft acceptance rate: 91.6%**
- **Mean acceptance length: 5.58 / 5 speculative tokens** — the draft was
  effectively producing accepted bursts of ~5-6 tokens per verifier pass

Prose-style workload (creative writing, T>0):

- **7-9 tok/s** sustained generation
- **Draft acceptance rate: 33-43%**

The acceptance gap between code and prose is the key calibration data
point. Code at T=0 is deterministic and structured — the 1B draft gets
the same easy decisions right as the 70B verifier most of the time. Prose
at higher temperature is high-entropy — the draft and target diverge
frequently, and rejected draft tokens are wasted work, eating the
speculative-decoding margin.

### Calibration findings

1. **Spec decode is workload-dependent, not a uniform win.** On
   structured/code output it's a real ~2× speedup. On creative prose
   it can actually be slower than the target alone, because draft
   prefill and rejected-draft compute pile up.
2. **The draft must share a tokenizer family with the target.** Mixing
   tokenizers gives wrong results or no speedup. Llama 3.2 1B works as
   a draft for Llama 3.3 70B; Qwen 1.5B would not.
3. **K=5 is a reasonable starting point.** Higher K wastes more compute
   on rejections; lower K under-amortizes the verifier pass.

### Practical use

Llama 70B + spec decode is a viable backend for code-generation workloads
on the Spark. It's not the right default for general use — see Experiment
3 for the prefill-side finding that changes the recommendation.

## Experiment 2: Tool calling for opencode

### Motivation

Local agentic coding loops need function calling to be production-grade.
opencode (and similar agents) emit `tool_choice: "auto"` requests; vLLM
refuses by default — it requires explicit flags to enable structured
function-call parsing.

### Setup

Added to the Llama 70B spin-up:
- `--enable-auto-tool-choice`
- `--tool-call-parser llama3_json`

Parser families in vLLM 0.20.2 (full list, useful for picking by model):
deepseek_v3/v31/v32/v4, ernie45, functiongemma, gemma4, gigachat3,
glm45/47, granite/granite-20b-fc/granite4, hermes (Qwen),
hunyuan_a13b, hy_v3, internlm, jamba, kimi_k2, **llama3_json** (Llama
3.1/3.3 Instruct), llama4_json/pythonic, longcat, mimo, minimax/minimax_m2,
mistral, olmo3, openai, phi4_mini_json, pythonic (Llama 3.2 1B/3B), qwen3_coder/xml,
seed_oss, step3/step3p5, xlam.

### Results

Verification curl with a `get_weather` function:

```
finish_reason: tool_calls
content: None
tool_calls count: 1
  [0] name: 'get_weather', arguments: '{"location": "Paris"}'
```

Clean OpenAI-API-compatible tool-call output. `content` is null (parser
correctly extracted the call), function name matches, arguments are
valid JSON.

opencode was then configured with a `dgx` provider in
`~/.config/opencode/opencode.json` pointing at `http://192.168.1.147:8001/v1`,
model id `casperhansen/llama-3.3-70b-instruct-awq`, `tool_call: true`,
context 65536. Agent loop ran end-to-end on real coding tasks.

### Calibration findings

1. **vLLM tool calling requires both flags.** `--enable-auto-tool-choice`
   alone returns `tool_choice not supported`; the parser needs to match
   the model family.
2. **If tool calls come back as plain text in `content`** (instead of in
   the `tool_calls` array), either the parser is wrong for the family
   or the chat template doesn't render the tool format — fallback is
   `--chat-template examples/tool_chat_template_<family>.jinja`.
3. **Verify one tool call manually before pointing an agent at it.** Cheap
   insurance against an agent loop that hallucinates because the parser
   silently misfires.

## Experiment 3: Gemma 4 26B MoE (A4B)

### Motivation

First non-dense model in the sweep, first multimodal-capable model,
first with Per-Layer Embeddings (PLE). Calibration value is in
*serving behavior unique to MoE* — expert routing overhead, decoupling
of memory cost from compute cost, batching behavior, anything that
doesn't exist on dense models.

### Setup

- Model: `google/gemma-4-26b-a4b-it` (BF16, ~52 GB weights)
- `--max-model-len 32768` (256K native is structurally infeasible — KV
  cache for a 26B-class model at full context would be ~130 GB)
- `--gpu-memory-utilization 0.75` (~96 GB of 128 GB cap)
- `--max-num-batched-tokens 8192` (required to satisfy vLLM's
  multimodal-budget validator — Gemma 4 is multimodal, so it forces
  `--disable-chunked-mm-input` and the MM token budget must fit)
- `--dtype bfloat16`, `--trust-remote-code`
- No quantization (avoids AWQ × MoE × PLE compound risk)
- Container: `vllm-chat` on port 8001 (replaced Llama 70B + spec decode)

Scripts:
- `/home/kroussos/src/dgx/spin-up-vllm-gemma4-26b-moe.sh`
- `/home/kroussos/src/dgx/test-gemma4-toolcall.sh`

### Phase A — serving behavior

#### Architecture confirmations from startup logs

- `Resolved architecture: Gemma4ForConditionalGeneration` — **multimodal
  (vision-language) model**, not text-only. The runbook framed this as
  pure MoE; it's actually MoE + VLM. Vision encoder gets loaded and
  warmed up at startup (20s) even when only doing text.
- `Gemma4 model has heterogeneous head dimensions (head_dim=256,
  global_head_dim=512). Forcing TRITON_ATTN backend to prevent
  mixed-backend numerical divergence.` — FlashAttention is unavailable;
  we run on Triton attention.
- `Using TRITON Unquantized MoE backend out of potential backends:
  ['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON', 'BATCHED_TRITON']`
  — FlashInfer's TRTLLM/CUTLASS kernels don't support sm_121, Triton
  wins by elimination.
- `WARNING: Using default MoE config. Performance might be sub-optimal!
  Config file not found at .../configs/E=128,N=704,device_name=NVIDIA_GB10.json`
  — **128 experts, intermediate dim 704, no tuned vLLM kernel config
  for this hardware × this shape.** Running on the fallback.

#### Memory accounting

Host RAM (from `free -h`):
- 117 GiB used / 121 GiB total / 4 GiB available
- Swap: 7.9 GiB / 15 GiB (some of which is historical from earlier
  experiments)

vLLM internal logs:
- Model weights: **48.5 GiB** resident (matches BF16 26B expectation)
- KV cache: **40.56 GiB → 557,022 tokens of capacity** (~17 concurrent
  maxed-out 32K-context sessions if we wanted them)
- Engine init (profile + KV cache + warmup): 124.94s; compilation 51.0s
- CUDA-graph memory note: vLLM 0.21+ deducts CUDA-graph memory from the
  `--gpu-memory-utilization` budget. Effective 0.741 vs nominal 0.750.

#### Startup wallclock

- Total to "Application startup complete": **~16.5 min** on first run
- Weights download: 389.9s (~6.5 min, ~52 GB — weights were not cached
  on first launch despite assumption otherwise)
- Shard 1 of 2 load (single 35-ish GB shard): 5:51, bandwidth-bound
- Multi-modal warmup: 20s
- torch.compile + flashinfer autotune + CUDA graph capture: rest of the
  time

#### Throughput probe (single-stream)

Identical binary search prompt to Experiment 1 (T=0.0, 400 max tokens):

- **Wallclock: 11.252s**
- **completion_tokens: 262**
- **Computed: 23.28 tok/s**

**Tied with Llama 3.3 70B AWQ + spec decode (23 tok/s) on the same
workload.** The MoE's "4B active per token" theoretical advantage did
not materialize in this benchmark — see "Why the MoE advantage didn't
show" below.

#### Concurrency probe (4 parallel, identical T=0 requests)

- Wallclock: 19.193s
- All 4 returned 262 completion_tokens (deterministic at T=0)
- Aggregate: **54.6 tok/s**
- Per-stream: 13.65 tok/s (each individual stream slowed to 0.59×)
- Speedup: 2.35× for 4× requests — **sub-linear**, expected for
  bandwidth-bound decode

MoE-specific interpretation: scaling looks like a dense bandwidth-bound
model at batch=4. The MoE potential win (different requests routing to
disjoint experts in parallel) does not appear to be exploited by the
Triton MoE backend in this configuration — possibly because identical
prompts route to identical experts, or because the Triton kernel doesn't
expert-batch across requests in the way FlashInfer might.

#### Why the MoE advantage didn't show up (on decode)

The advertised "4B active out of 26B" should have made decode 3-5×
faster than Llama 70B. It didn't, for compounding reasons:

1. **Untuned fused-MoE kernel** for GB10 + E=128/N=704.
2. **TRITON_ATTN attention backend**, not FlashAttention (forced by
   heterogeneous head dims).
3. **BF16 vs AWQ INT4**: Llama 70B reads 4× less memory per active weight.
4. **Bandwidth math**: at 273 GB/s and ~8 GB per-token MoE read (BF16,
   4B active), theoretical ceiling is ~34 tok/s; we hit 23 = ~68% of
   ceiling. The 32% gap is plausibly the untuned-kernel overhead.

### Phase B1 — CampaignGenerator narrative

Ran the `stormgiants` campaign's session-3 narration workload (~7.8K
output for one scene, `--narrate-tokens 20000`, `--prose-mode`,
`--reflections`, comic-noir genre brief with banned anti-patterns).

Run completed cleanly. Output at
`/tmp/test-gemma4/session_doc_scene_03_the_battle_of_the_ancient_dragons_begins.md`.

**Register hits (strong):**
- "the hoard everyone is killing each other for is actually just bait"
- "as if the universe had just handed him another bill he couldn't afford to pay"
- "manning a trebuchet with all the grace of drunken laborers"
- "like trying to empty the ocean with a spoon"
- "errant tourists" (party vs. storm giants)
- Running "Poor Bob" gag with "twelve seconds of combat"
- "his pragmatic theology manifesting as a battle cry"

**Banned-phrase compliance**: No literal "the shape of X" / "the
quality of X" / "that particular quality" violations. Some adjacent
generic-fantasy adjective stacking ("whirlwind of scales and fury",
"predatory grace") leaks through but doesn't dominate.

**Voice differentiation maintained**: Orsik (opportunist, calculating),
Vardis (dry theology), Thistle (tactician), Unla Key (self-deprecating
first-person narrator). Distinct.

**Refusal incidents: zero.** Runbook predicted Gemma's historical
refusal-proneness as the main quality risk. Did not materialize. Violent
combat depicted concretely ("rending him with a violence that felt
personal", "carving sixty-three points of damage out of him"), demon
character Bob treated normally, "To hear God, bitches!" delivered
without sanitization.

**Mechanical-text bleed** (damage numbers, exact movement feet) present
— per user note, this is a model-agnostic issue (even Sonnet does it),
addressed downstream via a scrubbing phase. Not scored against Gemma.

**Verdict**: Gemma 4 26B MoE is a viable CampaignGenerator backend for
narration. Register match, no refusals, voice-consistent.

### Phase B2 — opencode (tool calling)

Located the Gemma 4 tool-call parser via
`vllm serve --help=Frontend | grep tool-call-parser`. Available choice:
**`gemma4`** (explicit Gemma 4 parser, plus a `functiongemma` variant).

Restarted the container with:
- `--enable-auto-tool-choice`
- `--tool-call-parser gemma4`

Verification curl (`get_weather` probe via `test-gemma4-toolcall.sh`):

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

Updated `~/.config/opencode/opencode.json` with a `dgx/gemma-4-26b-moe`
model entry (context 32768, `tool_call: true`), switched default to
Gemma 4. Launched opencode against the Gemma endpoint and ran real
coding tasks. **Worked.** Agent loop functioned end-to-end.

## Cross-cutting finding: prefill vs decode

This is the most important result of the sweep and didn't come from a
formal probe — it came from the user actually trying to use the
Llama 70B + spec decode stack on a real workload after the formal
experiment was "done."

### The observation

User reported: Llama 3.3 70B + spec decode reading a `README.md` and
summarizing it was "so slow I gave up." The same task on Gemma 4 26B
MoE was "amazingly fast."

Both stacks measured at ~23 tok/s decode in the formal probe. So why
the gap?

### The mechanism

LLM inference has two distinct phases bounded by different resources:

- **Decode** (steady-state token generation) is **memory-bandwidth-bound**.
  Each output token requires reading all active model weights from
  memory once. Llama 70B AWQ (4-bit, ~35 GB to read) and Gemma 4 BF16
  with 4B active (~8 GB to read at BF16 = ~16 GB equivalent traffic
  with MoE routing overhead) hit similar bandwidth profiles, so they
  tied at ~23 tok/s.

- **Prefill** (processing the input prompt before generation begins) is
  **compute-bound**. It scales with `prompt_tokens × active_params`.
  Llama 70B is dense, so 70B active params per prefill token. Gemma 4
  MoE has ~4B active params per prefill token. **~17× compute gap on
  prefill.**

For a long README read with a short summary output:

| Stack | Prefill (10K input) | Decode (500 token summary) | Total |
|---|---|---|---|
| Llama 70B + spec decode | ~45s+ (estimated from user "excruciating") | ~10s | ~55s+ |
| Gemma 4 26B MoE | ~5s | ~22s | ~27s |

Gemma starts producing output much sooner. The benchmark missed this
because its prompt was ~30 tokens — prefill cost was negligible. Real
workloads load files into context, and prefill dominates.

### Spec decode interaction

Spec decode actively *hurts* on long-input workloads. The 1B draft
model has to prefill the same long prompt too, adding TTFT latency
before any draft tokens can be proposed. Then on diverse prose content,
the draft's acceptance is low (33-43% measured) and most draft work is
wasted. The result on long-input/short-output tasks is worse TTFT than
the target alone would have.

### Methodology lesson

For any LLM serving setup evaluation, measure **both** axes:

1. **Decode probe**: short prompt (~30 tokens), long output (400+
   tokens). Measure tok/s. Bandwidth-bound. Sensitive to model size,
   quantization, KV cache pressure.
2. **Prefill probe**: long prompt (5K-10K+ tokens), short output (~100
   tokens). Measure time-to-first-token. Compute-bound. Sensitive to
   total active params, MoE routing, prefill chunking.

A decode-only benchmark hides the production-relevant dimension when
real workloads are read-heavy. MoE vs dense comparison is meaningless
without the prefill probe — that's the whole point of MoE.

## Recommendations

### Default backend: Gemma 4 26B MoE

For prefill-heavy workloads (the user's typical shape — opencode reads
files, CampaignGenerator loads context, RAG-style flows):

- Gemma 4 MoE wins on TTFT decisively (~17× active-params advantage)
- Decode tok/s ties Llama 70B + spec decode (~23 tok/s)
- Quality on par for both narrative prose and agentic coding
- Tool calling works cleanly via `gemma4` parser
- Memory headroom is tight but adequate (~4 GiB available after
  weights + KV cache)

### Toggle to Llama 70B + spec decode for: short-input, long-code-output

The sweet spot for the spec-decode stack is "agent asks for a specific
code change with brief context" — small file or no context in, lots of
code out. The 91% draft acceptance on code at T=0 delivers a real ~2×
decode speedup. If the workload has minimal prefill, that win is
visible.

Toggle command (one ssh call):
```
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-llama70b-specdecode.sh'
```

…and flip `"model"` in `~/.config/opencode/opencode.json` back to
`"dgx/llama-3.3-70b"`.

### Don't bother with: Llama 70B + spec decode for prose

Spec decode's acceptance rate on diverse prose (33-43%) eats the
margin. Llama 70B alone (no draft) would be faster than Llama 70B +
1B draft for narrative workloads. Or just use Gemma 4 MoE.

### Decision rule of thumb

| Workload shape | Best stack |
|---|---|
| Long input, short output (reading code/docs, opencode summaries) | **Gemma 4 26B MoE** |
| Short input, long *code/structured* output | **Llama 3.3 70B + spec decode** |
| Short input, long *creative prose* output | Toss-up — Gemma 4 MoE is the safer default |
| Plan-driven code generation (Opus plan → code) | **Gemma 4 26B MoE** (plan is prefill-dominant) |

## Calibration findings — the "rough edges of local hardware" list

1. **No tuned fused-MoE kernel for the Spark + Gemma 4's expert shape.**
   The exact filename vLLM looks for is
   `configs/E=128,N=704,device_name=NVIDIA_GB10.json`. Doesn't exist.
   Performance is on the fallback — not catastrophic (~68% of the BF16
   bandwidth ceiling) but it caps the upside. Real MoE speedup on the
   Spark requires either running vLLM's `benchmark_moe.py` autotune
   script to generate the config ourselves, or waiting for the
   community to upstream a GB10 tuning.

2. **FlashAttention unavailable on Gemma 4** due to heterogeneous head
   dimensions (`head_dim=256, global_head_dim=512`). Forces TRITON_ATTN.
   Another perf-ceiling lever that's structural to Gemma 4's design,
   not specific to the Spark.

3. **FlashInfer TRTLLM/CUTLASS MoE kernels don't support sm_121** on
   GB10. Triton MoE backend wins by elimination — another "Spark
   gets the fallback, not the optimized path" finding.

4. **`nvidia-smi --query-gpu=memory.used` returns N/A on GB10.** Unified
   memory means there's no discrete VRAM to query. Have to use `free
   -h` (host total) and vLLM's own logs (its own model + KV cache
   accounting). Any tooling that assumes discrete-GPU memory reporting
   is broken on the Spark.

5. **Gemma 4 is multimodal**, not text-only, despite the colloquial
   framing. `Gemma4ForConditionalGeneration` architecture. Required
   adding `--max-num-batched-tokens 8192` to satisfy vLLM's MM-budget
   validator. Vision encoder loaded and warmed up at startup (20s)
   even for pure text workloads — small overhead paid regardless.

6. **`vllm serve --help` reorganized into config groups in vLLM 0.20.2.**
   Flags moved under `Frontend`, `ModelConfig`, etc. To find any flag,
   use `vllm serve --help=Frontend` (or `--help=all`). Confused initial
   parser-name discovery.

7. **Default `DGX_MODEL` in CampaignGenerator is `Qwen/Qwen2.5-14B-Instruct-AWQ`**
   regardless of what's actually serving at port 8001. Requests will
   return 400 unless `DGX_MODEL` env var is set to match the real
   served model. Easy footgun.

8. **MoE batching at small batch sizes doesn't show a win on identical
   prompts.** 4 parallel identical T=0 requests route to identical
   experts → no expert-disjoint batching speedup. Scaling looks dense.
   Whether MoE wins on diverse-prompt concurrency on the Spark is an
   open question.

9. **Gemma's historical refusal-proneness on morally-grey content did
   not materialize** on the CampaignGenerator narrative workload.
   Combat violence, demon characters, gallows humor — all delivered
   without hedging. Either Gemma 4's instruct tuning is meaningfully
   different from earlier Gemma releases, or the comic-noir prompt
   framing kept it on register.

10. **vLLM 0.21+ deducts CUDA-graph memory** from the
    `--gpu-memory-utilization` budget. Effective utilization is
    slightly lower than nominal. Bump by ~0.01 if you want to recover
    the diff exactly.

## Open questions for next experiments

- **Run vLLM's `benchmark_moe.py` autotune** on GB10 to generate the
  missing `E=128,N=704,device_name=NVIDIA_GB10.json` config file. If
  feasible, that closes the biggest perf gap surfaced by this sweep.

- **Flip `enable_return_routed_experts=True`** and inspect actual
  per-token expert routing. Would tell us whether the 128 experts are
  evenly used or whether routing collapses into a few hot experts —
  affects whether expert-parallel batching is even theoretically
  exploitable on this model.

- **Diverse-prompt concurrency probe.** Identical-prompt batching
  scaled like dense; mixed prompts might show whether MoE wins at
  concurrency where dense can't.

- **Spec decode on Gemma 4 MoE.** A small Gemma-family draft (Gemma
  2B or Gemma 3 instruct) + this 26B MoE target might compound the
  prefill win with a decode win. Risk: MoE × spec decode is two
  bleeding-edge interactions stacked.

- **Actual TTFT probes for both stacks.** The prefill-vs-decode finding
  came from user anecdote, not a measurement. A formal TTFT probe at
  several input lengths (1K, 5K, 10K, 30K tokens) on both stacks would
  quantify the gap and make the recommendation table above more rigorous.

- **Long-context behavior.** Gemma 4 advertises 256K native but we cap
  at 32K. Bump and characterize: at what length does TTFT become
  unusable? Does prefix caching cover most of the practical cost?

## Related files

- `/home/kroussos/src/dgx/spin-up-vllm-llama70b.sh` — base Llama 70B AWQ
- `/home/kroussos/src/dgx/spin-up-vllm-llama70b-specdecode.sh` — Llama 70B + Llama 3.2 1B draft
- `/home/kroussos/src/dgx/spin-up-vllm-gemma4-26b-moe.sh` — Gemma 4 26B MoE
- `/home/kroussos/src/dgx/test-gemma4-toolcall.sh` — tool-call verification probe
- `/home/kroussos/src/dgx/gemma4-26b-moe-runbook.md` — Phase A/B runbook (input to the experiment)
- `/home/kroussos/src/dgx/gemma4-26b-moe-observations.md` — Phase A/B raw observations + verdict
- `/home/kroussos/src/dgx/current-setup.md` — baseline serving topology before the sweep
- `/home/kroussos/src/dgx/model-comparisons.md` — pre-existing model-vs-model comparison notes
- `~/.config/opencode/opencode.json` — opencode provider config (DGX endpoint + model entries)
