# Qwen3-Next-80B-A3B-Instruct-FP8 — observations

Append-only experiment log for the **on-box single-box reference** that
runs in the `vllm-chat` slot on spark1:8001 — the daily driver spark1
returns to between experiments. Model id
`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`: 80B total, **A3B = 3B active**,
attention-based MoE, FP8. Setup/runbook facts live in `current-setup.md`
§3.

Companion logs:
- `qwen35-122b-2box-observations.md` — the *cross-box* 122B-A10B coding
  bar (a different, larger model; don't conflate).
- `nemotron3-super-120b-observations.md` — the NVFP4 Mamba-hybrid this
  model was A/B'd against on 2026-06-06; that doc carries the
  head-to-head and the architecture argument.

Don't rewrite entries — add new dated sections.

---

## 2026-06-06 — character sketch: decomposes well, but frame-locks

First real read on this model as a coding driver, from the same session
that evaluated Nemotron-3-Super on spark1 (see the Nemotron doc's
"plans wide, executes narrow" entry for the other side of the A/B).

### What it does well

- **It decomposes on its own.** Given a non-trivial change it attacked
  the problem *piecemeal* — small steps, each finished locally-consistent
  before moving on — rather than committing to a single one-shot rewrite.
  Each completed piece acts as an implicit checkpoint, so execution
  errors get caught at the boundary instead of compounding to the end of
  the file.
- **It executes coherently at only 3B active.** It held the thread across
  a multi-step change that Nemotron-Super (12B active, Mamba hybrid) lost
  partway through. Fewer active params, *better* execution — see the
  Nemotron doc for why that kills the "active-param count = execution
  capacity" story and points at memory architecture (full attention vs
  Mamba's fixed recurrent state) instead.

### Where it broke: frame-locking on a wrong premise

Concrete failure. Test code keyed its expected behaviour to an **embedded
JSON blob** — and there were **two** blobs present, one correct, one
wrong. The test was failing.

Qwen anchored on the premise *"the input I'm looking at is correct"* and
then debugged **competently from inside that frame** — checked the logic,
re-read the code, re-checked the assertion — looping on *"why is this
failing when the input is right?"* The actual problem: **it was looking
at the wrong blob.** It never put its own premise on trial. The one move
that breaks the loop — *"wait, which blob does this test actually
load?"* — requires questioning its own assumption.

**Outcome correction (it finished):** the frame-lock was *not* terminal.
After the stall, **Qwen recovered and completed the task.** So the loop
is a recoverable detour, not a dead end — "debugs the wrong thing
forever" overstated it. That's a *better* executor profile than first
written: the model can get stuck on a premise but is not permanently
trapped by it.

**Resolved — it did NOT break its own loop; the human did.** Correcting
any "self-recovered" reading: the model stayed stuck. *I* broke the loop
— by going to look at the code myself (or handing the bug to Claude/Opus
to find), locating the real problem (the wrong blob), and feeding the fix
back. *Then*, and only then, Qwen proceeded and finished. So "won't
generate the doubt itself / won't self-correct unprompted" **stands —
confirmed, not softened.**

What *containment* bought was **not** self-recovery — it was that the
stuck work-state stayed **coherent and inspectable.** Because Qwen had
emitted no garbage, a human (or an escalated Claude) could read the code,
find the one real bug, and hand it back — and the model picked up from
there. The failure left something *workable to debug.*

> **Containment → recoverability — and specifically, *surgical*
> recovery.** A contained failure leaves a small, coherent state you (or
> a stronger model) can inspect and fix *at the point of breakage*. An
> uncontained failure (Nemotron-Super's flood) leaves no coherent state
> to inspect — recovery is *wholesale*: discard the code, or redo the
> whole task in Opus. Containment is the difference between *"Claude,
> what's wrong with this function?"* and *"Claude, just do the whole
> thing over."*

This makes Qwen3-Next a *safe executor*: when it fails, you debug a
contained state and it proceeds — cost is *time you get back.* Super (see
`nemotron3-super-120b-observations.md`, "Failure shape") leaves no such
state to debug — the cost of its failure is *the work itself.*

### Read on the failure mode

- **This is a different *class* of failure than Super's.** Super loses
  the thread *executing* (a recall/coherence spill). Qwen executes fine
  but locks onto a wrong *frame* — a premise / scope / attribution error
  about *which* input is authoritative.
- **It looks like competent work, which makes it more dangerous.** Super
  visibly falls apart; Qwen confidently, consistently debugs the wrong
  thing. The output reads as capable right up until you notice it's been
  reconciling a correct input against a failing test, looping until an
  *external* agent breaks the frame — it never broke it itself (see
  "Resolved" above).
- **Piecemeal discipline does not save you here.** Decomposition
  checkpoints *local consistency* — every step was correct against its
  reference. But the reference itself was wrong, so the model can be
  locally consistent all the way down and not converge *on its own until
  the premise is challenged*. Piecemeal protects *execution*; it does
  nothing for a bad *premise*.
- **This is exactly the precision-decision hole from the global rule.**
  "Which blob is the input" is an attribution/scope call — and the model
  rendered confidently past it without ever flagging it as a decision.

### Actionable

- The weakness is **frame-locking**, not capacity: it will not
  spontaneously challenge *"which thing am I even looking at?"*
- Watch for the tell: **repeated, competent debugging that keeps
  re-deriving the same dead end.** That's the signature of a locked
  frame, not a hard bug.
- **The intervention when you see the loop: stop it and force a
  re-survey.** *"Stop. Drop the current thread. Look over the whole
  problem from the top."* The model can't re-widen its own aperture once
  locked, and it won't re-question a premise it's still actively
  reasoning *from* — interrupting the loop is what dislodges that premise
  from "settled assumption" back to "thing to check." You're not handing
  it information; you're making it re-derive its own starting point, and
  it will often spot the wrong blob *itself* on the way back up.
- The targeted nudge — *"are you sure that's the input the test
  reads?"* — is the **special case** where you've *already* spotted the
  wrong premise yourself. When you haven't (the common case), the general
  "stop and re-survey" move is what you reach for, because it doesn't
  require you to know the answer first.
- **Upside: the loop is *contained*.** When this model fails it produces
  *nothing* — it spins, it doesn't flood. The cost is wasted cycles, not
  bad code in the tree, and there's nothing to clean up once you stop it.
  That makes it a *safer* failure than Nemotron-Super's, which keeps
  generating worthless code when it loses the thread (see
  `nemotron3-super-120b-observations.md`, "Failure shape: Super floods,
  Qwen stalls"). Qwen wastes *your time*; Super pollutes *your tree*.
