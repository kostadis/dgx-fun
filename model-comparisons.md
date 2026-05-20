# Model Comparisons for the DGX Spark

Working notes on candidate models to run on the Spark vs the current
baseline. Focused on calibration tradeoffs (what does this teach me,
what does this cost me) rather than chasing raw benchmarks.

Baseline as of 2026-05-10 (see `current-setup.md`):

- **vllm-chat**: `Qwen/Qwen2.5-14B-Instruct-AWQ`, 32K context, ~9 GB AWQ weights
- **vllm-embed**: `nomic-embed-text-v1.5`, ~600 MB
- Spark hardware: GB10, 128 GB unified, 273 GB/s bandwidth
- Single-sequence decode on vllm-chat: ~15 tok/s (bandwidth-bound)

---

## Gemma 4 family (released 2026-04)

Four variants, all Apache 2.0, all built from the same research as
Gemini 3. Native vision + audio. Up to 256K context. 140+ languages.

| variant | params | context | notes |
|---|---:|---:|---|
| E2B | "effective 2B" via PLE | 128K | Phone/Pi class. Audio. |
| E4B | "effective 4B" via PLE | 128K | Phone/Pi class. Audio. |
| 26B MoE (A4B) | 26B total / ~4B active | 256K | The interesting serving experiment. |
| 31B Dense | 31B | 256K | The genuine quality upgrade. |

PLE = Per-Layer Embeddings. Embeddings stored per layer rather than
shared globally, trading VRAM for representational capacity. Lets a
small model punch above its weight without paying the per-token
compute cost of a bigger one.

---

## Per-variant analysis on Spark

### E4B — the cheap multimodal sidecar

**Footprint**: ~5 GB at 4-bit. 128K context. Native ASR + speech-to-
translated-text.

**Bandwidth math**: 273 GB/s ÷ 5 GB ≈ 50 tok/s ceiling. Realistically
30–40 tok/s. That's 2–3× current Qwen 14B decode rate.

**Where it wins**:

- Native audio at this size class — neither Qwen 3.5 nor Llama 4 ship
  audio at 4B. The single most novel capability across the Gemma 4
  family for a workstation deployment.
- High concurrency. Tiny weights → massive KV-cache headroom.
- PLE architecture is genuinely novel — worth touching to understand.

**Where it loses**:

- Coding quality. The Codeforces 2150 / LiveCodeBench numbers belong
  to the 31B Dense, not E4B. Don't conflate them.
- Long-document reasoning. 4B models hit "plausible but shallow"
  failure modes that 14B-class models avoid.
- Structured output (JSON, tool calls) — historical weak spot for
  small Gemma instruct variants vs Qwen.

**How it fits the stack**: NOT a vllm-chat replacement. Deploy as a
third container on port 8002, `--gpu-memory-utilization 0.1` (~13 GB
cap), used for cheap routing/classification, audio transcription,
anything where Qwen 14B is overkill. Replaces a hypothetical Whisper
sidecar if local speech becomes relevant.

**Learning payoff**:

1. PLE in practice — does the "effective capacity" claim hold on
   tasks you care about, or is it benchmark theater?
2. Multi-model serving — Qwen + E4B + nomic-embed concurrently, with
   client-side routing. A real pattern you don't get from one model.
3. Audio modality wiring — most novel addition of the Gemma 4 lineup.

### 26B MoE (A4B) — the architectural curiosity

**Footprint**: ~26 GB total weights, ~4B active per token. 256K
context.

**Why it's the most interesting experiment**: MoE serving on vLLM is
a different shape — expert routing, batch-size sensitivity, the
decoupling of VRAM cost from active-param compute cost. You can't
learn this from another dense model.

**Throughput hypothesis**: if active params are ~4B, per-token decode
*could* beat Qwen 14B despite higher total VRAM. In practice early
reports suggest MoE on current vLLM is slower than the active-param
math predicts — routing overhead and batch dynamics eat the win.

**Quality**: rivals 30B+ dense models on benchmarks. Strong "intelligence
per dollar of VRAM" pick.

**Caveats**: vLLM MoE support is still maturing — expect rough edges
on expert-parallel routing, especially with AWQ kernels.

### 31B Dense — the boring upgrade

**Footprint**: at AWQ ~16–20 GB. 256K context.

**Bandwidth math**: ~8–10 tok/s decode. Slower than Qwen 14B AWQ.

**Why pick it**: Codeforces 2150, LiveCodeBench edging Qwen 3.5-32B,
top-3 open model on Arena. The clean "same setup, better answers,
slower" trade.

**Caveats**: Need to confirm AWQ (or QAT-int4) availability on HF.
FP16 won't fit comfortably under the current 64 GB vllm-chat cap with
usable KV cache at 256K.

### E2B — skip

Smaller version of E4B. No reason to prefer it on Spark — the
hardware can trivially absorb E4B's extra 1 GB, and the quality gap
matters more than the speed gap.

---

## Cross-cutting considerations

### Tradeoffs vs Qwen 2.5-14B-Instruct-AWQ baseline

| dimension | Qwen 2.5-14B | Gemma 4 31B | Gemma 4 26B MoE | Gemma 4 E4B |
|---|---|---|---|---|
| Decode tok/s on Spark | ~15 | ~8–10 | uncertain (~10–20?) | ~30–40 |
| Quality (general) | strong | better | comparable to 31B | step down |
| Quality (coding) | strong | better | comparable | weaker |
| JSON / tool calls | very clean | unknown | unknown | risk area |
| Multimodal | text only | vision + audio | vision + audio | vision + audio |
| Native context | 32K | 256K | 256K | 128K |
| VRAM headroom on Spark | comfortable | tight at 256K | comfortable | trivial |

### Operational costs

- vLLM restart cost: ~3 min per swap (torch.compile). Pick flags
  generously up front.
- Gemma 4 is a month old (as of 2026-05-10). vLLM support exists but
  expect edge-case bugs, especially on AWQ + MoE.
- AWQ availability on HF needs verification per variant before any
  swap.

### Instruction-tuning style

Gemma models historically lean more refusal-prone and verbose than
Qwen. For agentic / structured-output workloads (llm_wiki, MemPalace
mining), Qwen is usually cleaner out of the box. Worth measuring on
real workloads before committing.

---

## Recommendations

**If picking one variant to deploy:**

- **For learning value**: 26B MoE. MoE serving patterns carry forward
  to other architectures and you won't learn them from another dense
  model.
- **For drop-in quality lift**: 31B Dense, provided an AWQ build
  exists. Same shape as current setup, better answers, slower.
- **For a complementary sidecar**: E4B on a third container. Doesn't
  replace Qwen 14B; adds audio and high-concurrency cheap inference
  to the stack.

**The honest call**: deploy E4B alongside the current setup first.
It's the cheapest experiment (low VRAM, novel modality, no risk to
working workloads) and answers the most interesting calibration
question — what does multi-model serving on one Spark actually feel
like, and is audio-in-one-model genuinely useful?

If E4B proves out the multi-model pattern, the natural next step is
26B MoE on a fourth container to learn the MoE serving shape.

31B Dense is the answer if and only if the goal is "make vllm-chat
smarter" and you're willing to give up ~5 tok/s.

---

## Open questions / things to verify before swapping

- [ ] Which Gemma 4 variants have AWQ or QAT-int4 builds on HF as of
  the swap date?
- [ ] Current vLLM version's Gemma 4 MoE support — known bugs?
- [ ] How does PLE interact with AWQ quantization? (PLE adds per-layer
  embedding tables — does AWQ quantize them well?)
- [ ] Audio input path through vLLM's OpenAI-compatible API — is it
  exposed via `/v1/audio/transcriptions` or a custom endpoint?
- [ ] Tool-call / JSON-mode reliability vs Qwen 2.5 on real
  CampaignGenerator and llm_wiki prompts.

---

---

## Nemotron 3 family (released 2026, NVIDIA)

Added 2026-05-18 after deploying Nemotron 3 Nano 30B A3B to the
`vllm-chat` slot (replacing Gemma 4 26B MoE pending Phase B
validation — see `nemotron3-nano-30b-observations.md`).

### The key distinguisher

**First model family where NVIDIA publishes a vLLM recipe with a
specific "DGX Spark / Jetson Thor" section.** Every other model
we've deployed runs on the GB10 (sm_121) fallback path because vLLM
doesn't ship tuned kernel configs for this hardware. Nemotron 3 on
the same hardware lands on the **tuned** path:

| | Gemma 4 26B MoE | Nemotron 3 Nano 30B |
|---|---|---|
| MoE backend | TRITON (fallback) | **FlashInfer CUTLASS** (tuned) |
| Attention backend | TRITON_ATTN (forced by head-dim mismatch) | **FlashAttention 2** |

This makes Nemotron 3 the first calibration data point we have for
"what does the tuned path actually look like on a Spark."

### Architecture (the second distinguisher)

Hybrid Mamba-2 + Transformer MoE. 30B Nano variant:
- 52 layers total
- 23 Mamba-2 layers (linear in context, constant per-sequence state)
- 23 MoE layers (128 experts, 6 active per token, +1 shared)
- **6** attention layers with GQA (2 groups)
- 30B total / 3.5B active per token (A3B)
- Native 256K context, 1M via env override
- Reasoning baked in (`<think>...</think>` blocks emitted before
  the final answer)

The 6-of-52 attention ratio means traditional quadratic KV cost
applies to only ~12% of layers. **Per-token KV is ~6 KB vs Gemma 4's
~73 KB on the same hardware** — measured at deploy, not estimated.

### Variants worth knowing

| variant | params | precision | weights size | notes |
|---|---:|---|---:|---|
| Nano 30B A3B BF16 | 30B / 3.5B | bfloat16 | ~60 GB | **currently deployed** |
| Nano 30B A3B FP8 | 30B / 3.5B | FP8 | ~30 GB | Faster decode on Blackwell FP8 cores. Try if BF16 is promising. |
| Nano 30B A3B NVFP4 | 30B / 3.5B | NVFP4 | ~15 GB | More lossy. Skip unless fitting alongside another model. |
| Nano 4B BF16 | 4B dense | bfloat16 | ~8 GB | Dense small sibling. Different shape; not a substitute. |
| Nano Omni 30B A3B | 30B / 3.5B | various | varies | Multimodal (text + image + audio + video). Separate experiment. |
| Super, Ultra | TBD | TBD | TBD | "Available in H1 2026" per NVIDIA's launch post. |

### Tradeoffs vs Gemma 4 26B MoE baseline

| dimension | Gemma 4 26B MoE | Nemotron 3 Nano 30B |
|---|---|---|
| Active params | ~4B | 3.5B |
| MoE kernel on Spark | TRITON fallback | **FlashInfer CUTLASS (tuned)** |
| Attention kernel on Spark | TRITON_ATTN (forced) | **FA2 (fast path)** |
| Single-stream decode tok/s | ~23 | **~23** (wash — bandwidth-bound) |
| Per-token KV cost | ~73 KB | **~6 KB** (12× cheaper) |
| Practical context | 128K @ ~4 sessions | **256K @ 8 sessions** (recipe) |
| Native max context | 256K | 1M (env-gated) |
| Reasoning behavior | none | **baked in** — `reasoning` field on responses |
| Tool calling | dedicated `gemma4` parser | `qwen3_coder` parser + reasoning plugin |
| Multimodal | vision (encoder loads even when unused) | text only (Omni variant is separate) |
| Cold start | ~16.5 min | ~18 min |
| Warm-restart init | ~10 min | ~1.2 min (post-load init 73 s) |

### Where Nemotron 3 wins

- **Prefill at long context** — fewer attention layers AND tuned
  kernel. Both effects compound; the architectural one grows with
  context length. The synthetic probe didn't produce clean enough
  data to publish a number, but the qualitative case is settled by
  the startup logs.
- **Long-context capacity** — 22× theoretical concurrency at 256K
  on the same KV budget that constrains Gemma 4 to 4 sessions at
  128K.
- **Warm restart cost** — ~1.2 min for the init phase vs Gemma 4's
  ~10 min compile cost. Once weights are cached, swapping in is much
  cheaper.

### Where Nemotron 3 loses (or is uncertain)

- **Time-to-first-useful-token under reasoning.** Even on "say OK",
  the model spends 232 tokens thinking before emitting 2 content
  tokens. Each opencode tool-call turn costs ~5 s under reasoning.
  For interactive feel this is a real regression vs Gemma 4.
- **Token economics worsen.** You pay for the reasoning tokens
  whether you read them or not. Smoke-test ratios are extreme
  (116:1); real workloads less extreme but still meaningful.
- **Tool-call + reasoning has an open HF bug.** Doesn't reproduce in
  the single-turn probe but might bite on long sessions. Phase B2
  will test.
- **Quality on real workloads is unmeasured.** Phase B1
  (CampaignGenerator) and Phase B2 (opencode) are still pending.
  Until then, "Nemotron is better" is a hypothesis from kernel
  selection, not a measurement on the workloads that matter.

### Operational status

- **Currently deployed in `vllm-chat` slot** as of 2026-05-18.
- **Client configs not yet flipped** — MemPalace, llm_wiki,
  CampaignGenerator, opencode still send `google/gemma-4-26b-a4b-it`
  and will 400 against the current deployment. Don't promote client
  ids until Phase B clears.
- **Revert path**:
  `ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-gemma4-26b-moe-longctx.sh'`

### Open questions for follow-up experiments

- [ ] How big is the FlashInfer CUTLASS vs TRITON-fallback prefill
      gap on real workloads? (Phase B1 wallclock comparison answers
      this.)
- [ ] Does the reasoning overhead net positive or negative for D&D
      narration workloads (multi-character voice consistency might
      genuinely benefit)?
- [ ] Does FP8 variant produce comparable quality at half the
      weights footprint? (Defer to a follow-up if BF16 lands well.)
- [ ] Does 1M context mode actually work with the recipe-stock
      `--max-num-seqs 8`? (Would need `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
      and roughly 2× the current KV pool — fits at the current util
      since pool is only 36% used.)
- [ ] How does the Omni multimodal variant compare for desktop chat
      use cases involving images?
- [ ] How does Super / Ultra (when released) change the picture?

---

## See also

- `current-setup.md` — what's actually running on the Spark today
- `nemotron3-nano-30b-observations.md` — detailed experimental
  record for the current deployment
- `nemotron3-nano-30b-test-plan.md` — methodology + decision
  criteria for the active Phase A/B comparison
- `gemma4-26b-moe-observations.md` — full record for the prior
  vllm-chat default (baseline for the Nemotron comparison)
- `spark-llm-serving-learnings.md` — bandwidth ceiling math and the
  reasoning behind current model choices
- `desktop-chat-clients.md` — Windows-side wiring for vllm-chat
