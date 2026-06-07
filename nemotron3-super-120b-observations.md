# Nemotron-3-Super-120B-A12B-NVFP4 — observations

Append-only experiment log for the single-box NVFP4 Nemotron-Super run
on spark1. Companion to `nemotron3-nano-30b-observations.md` (the 30B
sibling), `qwen35-122b-2box-observations.md` (the cross-box coding bar),
and `qwen3-next-80b-observations.md` (the on-box A3B reference this was
A/B'd against live). Setup/runbook facts live in `current-setup.md` §3
and the memory note `project_nemotron3_super_nvfp4`.

Don't rewrite entries — add new dated sections.

---

## Why this model was on the box

`Qwen/Qwen3.5-122B-A10B-FP8` (cross-box, TP=2 over the RoCE cable) was
the first local model that subjectively "could actually code" — the
quality bar to beat (`qwen35-122b-2box-observations.md`).

The open question: can a **single-box** NVFP4 hybrid hold that bar?
Nemotron-3-Super is 120B total / **12B active** (vs Qwen3.5's 10B
active), so on paper it's in the same activated-parameter class while
fitting on **one** Spark — freeing the second box and idling the cable.
If it held the bar, that's a strictly better operating point: same
coding quality, half the hardware, no inter-node NCCL.

Brought up 2026-06-06 via `spin-up-vllm-nemotron3-super-120b.sh` on
spark1:8001. The risky infra all worked first try (see next section),
so what's left below is a clean capability read on the 12B-active path.

---

## 2026-06-06 — infra validated (the risky parts)

All of these were genuine unknowns going in; all passed. Recording so a
future run doesn't re-litigate them:

- **NVFP4 inference runs on GB10 / sm_121 with real CUTLASS FP4
  kernels**, not the emulation fallback: `FlashInferCutlassNvFp4` GEMM +
  `FLASHINFER_CUTLASS` NvFp4 MoE backend. NVFP4 auto-detected from the
  checkpoint's `hf_quant_config.json` — no `--quantization` flag needed.
- **Nemotron-H hybrid (Mamba-2 + latent-MoE) loads and serves at 120B**
  on the NGC image `nvcr.io/nvidia/vllm:26.05-py3`. ~70 GB NVFP4 weights,
  TP=1, gpu-util 0.85, 128K context, fp8 KV. Cold load ~30 min on a
  fresh ~70 GB pull.
- **Reasoning parser:** the downloaded plugin registers the name
  **`super_v3`** (class `SuperV3ReasoningParser`), NOT `nemotron_v3`
  (the built-in name NVIDIA's generic recipe assumes). Same reasoning
  *leak* shape as the Nano: the trace lands in a `reasoning` field that
  opencode drops silently — harmless for coding (code is in `content`),
  but budget `max_tokens` ~1.5–2× to leave room for the hidden trace.
- Smoke test + tool-call (`qwen3_coder` parser) both PASS.

MTP (`num_nextn_predict_layers=1`) is in the checkpoint but was left
**OFF** — it's a self-speculative *decode-speed* knob, not a capability
lever, so it was out of scope for the quality question.

---

## 2026-06-06 — preliminary verdict: misses the Qwen coding bar

Driven live in opencode (default set to `dgx/nemotron3-super-120b`).
Judged on coding **output**, not tok/s.

**Verdict: does not clear the Qwen3.5-122B bar. Reverting spark1 to the
Qwen3-Next-80B reference.**

What it did well:
- **Reasoning is genuinely good.** It reads as a capable model when it's
  thinking about a problem in the abstract.
- **It correctly *sees the scope* of a task.** Given a non-trivial piece
  of work, its framing of "here is what needs to change and why" was
  sound — it understood the shape of the problem.

Where it broke:
- **It gets lost *executing* a long-horizon change.** Concrete case: a
  rewrite of a Python parser. It correctly sized up the whole job up
  front, then **bogged down and lost the thread partway through the
  rewrite** — the kind of failure where the model can describe the
  destination but can't carry the full file/state through the multi-step
  edit to get there.

Read on the failure mode:
- This is **plan-good / execution-bad**, which is the *expected* weakness
  of a 12B-active path on a single long artifact. Seeing scope is a
  one-shot reasoning act; holding an entire parser coherent across a
  multi-edit rewrite is a working-state-capacity problem, and that's
  where the thin activated path appears to spill.
- It is a **capability gap, not a latency gap.** MTP / `ENABLE_MTP=1`
  would make it *faster* at being lost — it does not address this. No
  point chasing the speed knob for this workload.
- Caveat worth a follow-up: a parser is a large single artifact. "Sees
  scope, loses the thread" could be partly a working-context-spill
  effect rather than pure capability. A more *bounded* rewrite (a single
  function, a well-fenced module) would isolate whether the ceiling is
  the task size or the model. Not yet run.

Net: Qwen3.5-122B remains the coding bar; Nemotron-Super doesn't reach
it on real multi-step coding work. The single-box NVFP4 hybrid is a
real, working serving path — just not (yet) a replacement for the
cross-box coder on the workload that matters here. spark1 returned to
`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` (the single-box reference) as the
daily driver.

---

## 2026-06-06 — sharper verdict: plans wide, executes narrow

Watching the two models on the *same* parser-rewrite task, the
difference wasn't quality-per-token, it was **strategy**:

- **Nemotron-Super went one-shot and exploded.** It sized up the whole
  job, committed to rendering the entire fix in a single pass, and ran
  out of working capacity partway — renamed a thing early, forgot it
  later. Big plan, small hands.
- **Qwen3-Next-80B-A3B attacked piecemeal.** (The on-box single-box
  reference — 3B active — i.e. the model spark1 actually A/B'd against,
  not the documented cross-box 122B bar.) It decomposed into small steps,
  finished each one locally-consistent, then moved on. Each completed
  piece is an implicit checkpoint, so errors got caught at the boundary
  instead of compounding to the end of the file.

Why this is architectural, not temperament:

- **It is _not_ active-parameter count** — and our own data forces this
  correction. The tempting story was "Super's 12B active can't hold the
  interdependencies while rendering." But the model that executed
  *better*, Qwen3-Next-80B, is **A3B — 3B active**, far *sparser* than
  Super's 12B. Fewer active params, better execution. The active-count
  axis points the wrong way; whatever separates these two, it is not how
  much compute fires per token.
- **The likelier driver is the architecture of _memory_, not size.**
  Nemotron-H is a Mamba-2 hybrid: prior context is compressed into a
  fixed-size recurrent state. Qwen3-Next is attention-based and can
  re-read its own earlier edits exactly. "Loses the thread partway
  through a long rewrite" is the textbook Mamba weakness — lossy
  long-range associative recall — and it fits the evidence far better
  than any active-param argument. Cheap discriminator: feed Super its own
  prior edits back in-context at each step; if that rescues it, the
  failure was recall (Mamba state), not capacity.
- This is the global-CLAUDE.md **LLM pipeline rule playing out inside a
  single model.** A one-shot rewrite is "extract → structure → render
  with no checkpoint" — the model inherits its own step-3 error into
  step-9 and amplifies it. Piecemeal execution inserts the checkpoints
  the one-shot run skipped.

Actionable read (supersedes the looser "misses the coding bar"):
**plans wide, executes narrow — force decomposition.** A 12B-active
model can *plan* a 122B-class problem but must be *driven* to execute it
in small, verified steps: decompose explicitly, ask for one bounded edit
at a time, verify each before the next, never request the whole rewrite
in one shot.

**Status: hypothesis, not yet measured.** What we *observed* is that
Super, left alone, chose one-shot and lost the thread while Qwen chose
piecemeal and held it. That forced decomposition would close the gap is
an *inference* from the architecture argument, not a run we've done. It
can fail two ways:

1. **Steps that are "small" by line count may still be wide by
   context.** A parser where every function references the shared
   grammar/state means even a one-function edit drags in the whole
   interdependency web — possibly still past 12B-active coherence.
2. **The hand-holding may eat the point.** If you have to pre-decompose
   the problem finely enough for Super to hold each step, *you* are doing
   the architectural work and the model is just rendering. At that point
   Qwen-that-decomposes-itself is strictly less babysitting for the same
   output.

The second failure mode is the real test. It is **not** "can Super do
piecemeal" — it almost certainly can. It is: **how fine does the
decomposition have to be before Super holds, and is that finer than the
point where you'd rather use the model that decomposes itself?** That
ratio is the calibration datapoint.

### Failure shape: Super *floods*, Qwen *stalls*

The two models don't just fail in different *places* (Super in
execution, Qwen in framing) — they fail with very different **blast
radius**, and Super's is worse.

- **Qwen3-Next fails *unproductively*: it loops.** Spins on the same
  dead end, produces *nothing*. Loud, obvious, and leaves **no mess** —
  the cost is wasted cycles, not bad code in the tree. Stop it,
  re-survey, move on; nothing to undo.
- **Super fails *productively*: it floods.** When it loses the thread it
  keeps *generating* — oodles of plausible-looking, worthless code. It
  doesn't stall, so you can't detect failure by watching it stop. Now you
  own a pile of artifact you have to read, distrust, and discard — and
  some of it can survive review *because it looks like work.*

This is the global rule biting: *"if a first-pass output looks
impressive, that's the best it can do."* Super's volume of code is that
trap — impressive-looking output is the ceiling, and the volume makes the
garbage harder to spot, not easier.

Operational pairing:
- **Qwen loops → stop it.** Cheap. Nothing to clean up.
- **Super floods → stop it *and* distrust the artifact.** Expensive. You
  have a cleanup pass, and you cannot assume any of the generated code is
  load-bearing.

**The deeper cut: containment determines whether recovery is _surgical
or wholesale_.** This is the part that actually matters operationally.
Neither model self-recovered — the distinction is what the *human* (or an
escalated Claude) could do once it got stuck.

- A **contained** failure (Qwen's loop) leaves a **coherent, inspectable
  work-state**. The human broke the loop — read the code, found the one
  real bug, handed it back — and the model *proceeded and finished* (see
  `qwen3-next-80b-observations.md`, "Resolved"). Recovery is **surgical**:
  *"what's wrong with this function?"* Cost: time you get back.
- An **uncontained** failure (Super's flood) **poisons the artifact** —
  there's no coherent state to inspect. Recovery is **wholesale**: the
  two paths actually observed were **throw the code away**, or
  **escalate to Opus** (off the Spark, to the Anthropic API) to *redo the
  whole thing.* Cost: the work itself.

So the flood doesn't just cost a cleanup pass — it costs you *the option
of recovering on-box at all*, and forces *wholesale* escalation rather
than a targeted one-bug fix. (Calibration datapoint worth naming: the
local NVFP4 model's failure forced an *off-box, whole-task escalation* —
the suboptimal-by-design local path handed the entire job back to the
API, not just the broken piece.)

### Next experiment (not yet run)

Same parser rewrite, but *you* supply the step boundaries:

- Re-run the identical parser-rewrite task on Super, this time driving it
  piecemeal — hand it one bounded edit at a time, verify each before
  issuing the next, never ask for the whole file at once.
- **Measure: how coarse can the steps be before it loses the thread?**
  Start with a generous decomposition (e.g. one function per step) and
  widen until it spills; the granularity at which it breaks is the
  number we want.
- Compare that required granularity against how Qwen self-decomposed the
  same job unaided. If Super needs materially finer steps than Qwen
  volunteers, that's the cost of the single-box NVFP4 operating point in
  babysitting terms — and it tells you whether "free the second box" is
  actually free.
- Keep MTP off; this is a capability/strategy read, not a speed one.
