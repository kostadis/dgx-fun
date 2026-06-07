# Super-plans / Qwen-executes — experiment plan

**Status: PLAN. Not deployed.** Nothing in `current-setup.md` changes
until this actually runs on the boxes. This doc is the design + runbook
for a two-model pipeline; revisit live box state before executing.

## The idea, in one line

Use **Nemotron-3-Super-120B** (the architect) to produce a decomposed
plan, gate that plan through a **human review**, then use
**Qwen3-Next-80B-A3B** (the executor) to carry out each step inside the
verified frame.

Companion reading — the two failure-mode logs this design is built on:
- `nemotron3-super-120b-observations.md` — Super: plans wide, executes
  narrow; *floods* worthless code when it loses the thread.
- `qwen3-next-80b-observations.md` — Qwen3-Next: decomposes + executes
  coherently at 3B active, but *frame-locks* on a wrong premise and
  loops.

The governing constraint is the global-CLAUDE.md **LLM Pipeline Design
Rule**: an LLM scope decision that feeds the next LLM step *automatically*
is the bad pattern. The human gate below is not optional polish — it is
the thing that keeps this combination from being worse than either model
alone (see "Why the gate is load-bearing").

---

## Why this split (it *subtracts* failure modes, not just adds strengths)

The naive read is "architect + executor = each model in its strength
role." The real reason it's a good split: **it parks each model where its
own worst failure physically cannot fire.**

- **Super plans → it can't flood.** The flooding failure only happens
  when Super is *executing* a long artifact and loses the thread. A plan
  is bounded output — there is nothing to flood. Never asking it to write
  code takes its dangerous failure mode off the table.
- **Qwen executes a written plan → its frame is pinned.** Qwen
  frame-locked on the JSON-blob task because *nothing told it which input
  was authoritative* — it had to infer the frame and inferred wrong. If
  the plan *names the frame explicitly per step* ("input is blob A; the
  test loads it from X"), Qwen's frame-locking has nothing to latch onto
  wrongly. Its weakness is mitigated by the structure it's executing.

So the pairing is *subtractive*: Super's flood is suppressed (bounded
output), Qwen's frame-lock is suppressed (frame handed, not guessed).

**Recoverability seals the assignment — Super-as-executor is
*disqualified*, not just worse.** The two models' failures differ in
whether recovery is even possible on-box:

- Qwen's executor failure is *contained* (a loop). It won't break its
  own frame, but it leaves a **coherent, inspectable state** — so you (or
  a surgically-escalated Claude) read it, find the one bug, hand it back,
  and the model proceeds. Safe to have inside an execution loop: recovery
  is targeted and the failure costs *time you get back*, not the work.
- Super's executor failure is *uncontained* (a flood). It poisons the
  artifact; the observed recovery was **discard the code or escalate to
  Opus off-box.** A failure with no cheap on-box recovery cannot sit in
  the execution loop.

So this isn't "Qwen happens to fit the executor role better." It's:
**never put Super in the execution loop** — its failure mode has no cheap
recovery, and you'd only discover that *after* it has already produced a
pile of poisoned code. Super earns its keep strictly upstream of the
gate, producing bounded plans it cannot flood.

---

## Why the gate is load-bearing (not ceremony)

The one residual exposure after the role split is **Super's plan being
wrong** — a scope/ordering/attribution error. Per the global rule, scope
is a *precision decision*; Super "sees scope" but its plan is a *draft*,
not ground truth.

And Qwen is the worst possible model to hand an unreviewed plan to,
because frame-locking means it **will not catch the error** — it executes
a wrong plan cleanly and confidently. That is strictly more dangerous
than either model alone:

- Super alone fails *visibly* (a flood of obvious garbage you'd never
  ship).
- Qwen-executing-Super's-bad-plan fails *invisibly* (clean, confident,
  wrong code). Qwen's coherence *removes the flailing* that would
  otherwise signal "this plan is wrong."

**The pipeline can convert a loud failure into a silent one.** The human
gate on the plan is what prevents that conversion. It is the irreducible
checkpoint.

---

## Data flow

```
  problem statement
        │
        ▼
  ┌───────────────┐
  │  SUPER (plan) │  spark1 — Nemotron-3-Super-120B-A12B-NVFP4
  └───────────────┘
        │  plan = ordered, bounded steps, each with an acceptance check
        ▼
  ┌───────────────┐
  │  HUMAN  GATE  │  ← review scope / ordering / attribution. CORRECT here.
  └───────────────┘
        │  verified plan
        ▼
  ┌───────────────┐
  │ QWEN (execute)│  spark2 — Qwen3-Next-80B-A3B-Instruct-FP8
  └───────────────┘   one step at a time; run each step's acceptance check
        │
        ▼  step check fails ──► STOP, re-survey (don't let Qwen loop)
   working code
```

Two independent endpoints — **not** tensor-parallel, no cross-box NCCL.
The orchestration glue (and the gate) is the actual subject of the
experiment; the models are off-the-shelf.

---

## The plan-format contract (what Super must emit)

Prompt Super to return the plan as an ordered list of **bounded steps**,
each carrying:

1. **Goal** — one sentence, what this step changes.
2. **Frame** — the explicit facts Qwen must not re-question: which file,
   which input is authoritative, which function owns the behaviour. *This
   is the field that pre-empts Qwen's frame-locking.*
3. **Acceptance check** — a concrete, runnable test of done-ness for this
   step alone (a command, an assertion, an expected output). Local to the
   step, not the whole task.
4. **Touch list** — the files/symbols this step is allowed to modify
   (bounds the blast radius; a step that wants to touch more is a smell to
   catch at the gate).

Bounded + per-step acceptance is what makes the human gate tractable
(review *structure*, not prose) and gives the executor an external signal
per step.

---

## The gate (what the human checks — fast)

Review the *plan*, not the code. Specifically the precision decisions
Super is unreliable at:

- **Scope** — does each step's touch list belong to that step? Anything
  reaching past its boundary?
- **Ordering** — does any step depend on a later step's output?
- **Attribution / frame** — is the "authoritative input / owner" in each
  step's Frame field actually correct? (This is the blob-ambiguity class
  of bug — catch it *here*, before Qwen cements it.)

Correct the plan in place, then release it to Qwen. This is the
human-review-imposes-structure step from the global rule's good pattern.

---

## Execution loop + frame-lock recovery

For each verified step, in order:

1. Hand Qwen **one** step (goal + frame + touch list). Not the whole
   plan — keep its working set small.
2. Run the step's **acceptance check**.
3. **Pass** → next step. **Fail** → do *not* let Qwen keep debugging; it
   will **not** break its own frame (confirmed — see the Qwen log's
   "Resolved"). **An external agent breaks the loop, not the model.** Two
   recovery options, both cheap *because the failed step's state is
   contained and inspectable*:
   - **You** read the (small, coherent) diff for that step, find the real
     bug, hand the correction back — Qwen proceeds.
   - **Escalate that one step to Claude/Opus**: *"what's wrong with this
     function?"* — a **surgical** escalation of the single broken step,
     not the whole task. Fold the fix back and Qwen continues.
   This surgical recovery is *only* possible because the executor's
   failure is contained. (It is the inverse of a Super flood, where the
   poisoned artifact forces a *wholesale* redo — which is the whole reason
   Super is barred from this role.)
4. If the inspection reveals the *plan* was wrong (not the execution),
   bounce back to the gate. Don't patch the plan inside the executor —
   that re-couples the failure modes.

Watch for the executor tell: **repeated competent-looking debugging that
re-derives the same dead end** = locked frame. Interrupt it and inspect
the contained state; do not wait for it to self-correct.

---

## Hardware wiring — VERIFY LIVE BEFORE RUNNING

The two models **cannot share the `vllm-chat` slot**; only one model per
port per box. As of this writing the slots are contended:

- spark1:8001 — alternates between Super and the Qwen3-Next daily driver.
- spark2 — currently runs Nemotron-3-Nano-30B (per memory
  `reference_spark2`).

So this pipeline needs a deliberate box assignment, e.g.:

- **Super on spark1:8001**, **Qwen3-Next on spark2** (displacing Nano, or
  on a second port if VRAM allows).

**Before running, confirm reality** (per the repo's hard rule):

```bash
curl -sS http://192.168.1.147:8001/v1/models   # spark1 vllm-chat
curl -sS http://192.168.1.69:8001/v1/models    # spark2 vllm-chat
```

The orchestrator points the *plan* call at whichever endpoint serves
Super and the *execute* calls at whichever serves Qwen3-Next. If a model
swap happens to bring this up live, **that** is the change that updates
`current-setup.md` — not this plan doc.

---

## First test task

Re-run the task that exposed both failure modes: **the parser rewrite
with the two embedded JSON blobs** (one correct, one wrong).

- It's the natural A/B: Super alone lost the thread; Qwen alone
  frame-locked on the wrong blob.
- Success criterion isn't just "tests pass" — it's whether the **gate +
  per-step Frame field** catches the blob-authority decision that sank
  Qwen solo, and whether bounded steps keep Super from flooding.

---

## What this experiment actually measures

Not tok/s. The questions:

1. **Does the role split suppress both failure modes as predicted?** Does
   Super-as-planner avoid flooding, and does an explicit per-step Frame
   actually pre-empt Qwen's frame-locking?
2. **How heavy is the gate?** If reviewing/correcting Super's plan is so
   much work that *you* are effectively doing the architecture, the
   pipeline collapses to "human plans, Qwen executes" and Super isn't
   earning its box. The gate-effort/quality ratio is the real datapoint —
   same calibration question as the Nemotron doc's "how fine must
   decomposition be before you'd rather use the self-decomposing model."
3. **Is two-box orchestration glue worth it** vs just driving Qwen3-Next
   solo with a human-written plan? Super has to add enough planning value
   to justify the second endpoint and the handoff plumbing.

The honest null result to watch for: **Super's plan needs so much
correction at the gate that it's cheaper to skip Super and hand Qwen a
human plan.** That's a real possible outcome and the experiment should be
willing to return it.
