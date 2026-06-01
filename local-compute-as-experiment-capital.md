# Local compute as experiment capital

*What distilling a 56-session D&D campaign on two DGX Sparks taught about
when local models are the right tool — and why the answer is about cost
*structure*, not cost.*

Written 2026-06-01, after the CampaignGenerator "world_state" distillation
moved from the Claude API onto the Sparks. This is an append-only learnings
doc, not a spec.

---

## TL;DR

The interesting property of owning the hardware isn't that tokens are cheaper.
It's that **sunk capex removes the cost *gradient* on iteration**. A fixed
opex budget (API tokens) caps how many experiments you can run *and* makes
every null result a pure loss — which quietly pushes you toward timid,
high-confidence experiments and away from the weird ones where the wins hide.
Local capex makes the marginal cost of an experiment ≈ power, so you can take
many cheap shots on goal. The better-than-before result came from the
*aggregate of affordable experiments*, not from any single clever move or from
the local models being "better." The honest claim is **affordable iteration
beat unaffordable perfection** — a statement about cost structure, not model
quality.

---

## The case study: distilling a campaign into a canon doc

**Goal.** Turn ~56 sessions of an *Out of the Abyss* campaign into a
`world_state.md` — a compact (~7K-token) "current state of the world" canon doc
a GM scans during prep. The original tool (`distill.py` in CampaignGenerator)
did this with the Claude API: chunk the bible per chapter, extract per-chapter
state notes, then one big synthesis call.

**The local pipeline.** We rebuilt the front of that on two DGX Sparks
(`spark1` + `spark2`, both serving Qwen3-Next-80B-A3B-Instruct-FP8):

- `ensemble_extract.py` — 5 extraction "lenses" × 3 self-consistency samples
  per chapter, fanned across both boxes, deterministically merged (with an
  embedding-cosine pass to catch cross-subject dupes). Output: **22,240 atomic
  facts** across 56 chapters, each with a `source_quote` and an `n_samples`
  confidence signal.

**What broke, and what we built (the robustness arc).** Running the full 56
chapters surfaced exactly the failure modes local serving has and the API
hides:

- *Tail stragglers.* A plain work-stealing queue self-balances mid-run but
  leaves a hole at the tail: when the queue drains, a free box exits while the
  other is still wedged on its last unit. Fix: **speculative re-execution** — a
  free endpoint re-runs the longest-running in-flight unit and the first copy
  to finish wins.
- *A "stall" that wasn't degradation.* The first full run hung ~6.5h on one
  chapter. I confidently diagnosed vLLM degradation / runaway decode toward the
  16K-token ceiling. **I was wrong.** Live metrics showed 0 preemptions and
  ~0% KV cache, and the real cause was mundane: *the laptop driving the run
  went to sleep*, freezing the Python processes with their TCP sockets to vLLM
  half-open; on wake the sockets were dead and the blocked reads never
  returned. Nothing had "degraded." Fixes: a **per-unit wall-clock timeout**
  (kill + re-queue; the old `split_run.sh` had `timeout 600` and the rewrite
  had dropped it) and an explicit **read/connect timeout on the OpenAI-compat
  client** so a stale socket raises in minutes instead of never. Lesson worth
  keeping: *a frozen socket is neither a slow box you wait out nor an error you
  retry — only a wall-clock cap bounds it.* And: distrust your first
  degradation story; check the boring explanation first.

**The synthesis wall.** 22,240 facts render to ~537K tokens — far past a 200K
context. A single synthesis call is impossible, and even the *old* `distill`
extraction corpus was ~228K tokens (already near/over the ceiling). So the
naive "feed everything to one model" path was never viable at this scale.

**The aggregation insight.** Insert a compression layer the old pipeline
lacked: collapse the atomic facts **per entity** into a current-state dossier.
A nice inversion of an earlier learning — reasoning models (Nemotron's
always-on `<think>`) flunked *bulk extraction* because the think budget was
pure tax on high-volume verbatim work; but *aggregation* is judgment-bound
(resolve recency/contradiction/attribution for one entity), which is exactly
what a think trace is for. In the event, plain **Qwen Instruct** did the
aggregation well enough that the reasoning-model A/B was deferred — the user's
bet that it "won't buy much" looks right so far. `facts_to_state.py` bundles
facts by `(type, subject)` and aggregates only the *stateful* recurring
entities; 703 entities (≥3 facts) → 703 dossiers, **0 failures**, ~2–3h across
both boxes. The corpus dropped 537K → ~265K, and a significance floor
(≥40 facts ≈ the ~40 entities the human doc actually covers) brings the
synthesis input to **~40K tokens** — one cheap API call with huge headroom.

**The discipline that makes it safe.** Aggregation makes scope, ordering, and
attribution decisions — *precision* decisions. So every dossier ends with a
mandatory `## Uncertainty` block listing what the model couldn't resolve
(contradictions, ambiguous chronology, unclear attribution). One terse prompt
line — "don't assert another entity's current state" — got the local model to
correctly *hedge* a companion roster and flag dead/departed members to
Uncertainty instead of stating them as present. That's the
**extract → human review → render** pattern: the model drafts structure, a
human ratifies it, the big model renders. Errors get a checkpoint instead of
compounding silently. The atomic `source_quote`s make the review auditable.

---

## The argument: capex vs opex changes which experiments you run

Buying hardware is **capex**; buying tokens is **opex**. Even when the dollars
match, they behave differently, and the difference is the whole point.

**1. Sunk capex makes the marginal experiment ≈ free.** Once the box is
bought, a run costs power. A Spark pulls ~150–240W under load; a 3-hour
ensemble run is ~0.5–0.7 kWh — pennies. The API path charges real money for
the same work, *every run*.

**2. The real lever is amortization, not per-run price.** Capex amortizes
across unlimited re-runs; opex scales linearly with usage. So the place the
difference bites is **iteration**. Tuning this pipeline meant re-running
extraction and aggregation many times — self-consistency sampling, the
two-Spark ensemble, the embedding merge, a fact-atomicity prompt fix, then the
aggregation-prompt iterations (concrete-detail slots, the cross-entity guard,
repeated single-entity re-runs). Locally each iteration was ~free. Metered,
many would have failed an "is this worth $X" gate and never been tried.

**3. A fixed opex budget caps experiment *count* and penalizes null results.**
This is the sharpest version. Under a budget, a failed experiment is pure loss
— you paid hundreds of dollars to learn "no." That cost structure reshapes the
science you're willing to do: you run only experiments you're already fairly
sure will work, which is precisely backwards for exploration. The
high-variance, probably-won't-pan-out ideas — the tails of the search space,
where the surprising wins live — are the ones a budget tells you not to run.
**Local removes the penalty on null results,** so you can take the weird shots.

**4. Quality follows from count.** Empirical tuning is a search; the result is
roughly a function of how many informative samples you can draw. The
better-than-before `world_state` was the *stack* of cheap experiments, not one
move. Affordable iteration is the mechanism.

---

## Honest caveats (so the thesis doesn't overreach)

- **Exploration vs exploitation.** Cheap iteration dominates while you're
  *searching* for the approach. Once you've *found* it, economics flip back to
  per-run quality, and a frontier API model can be the right call for the
  final, low-volume, quality-critical pass. The pipeline does exactly this:
  local for the iterative bulk (extraction, aggregation), the big API model for
  the one final synthesis (now cheap, since the input is ~40K not 537K tokens).
  **Local-to-explore, API-to-finish.**
- **Engineering opex is the hidden cost.** Look at what this work *was*:
  building speculative execution, per-unit timeouts, a client read timeout, and
  debugging a sleep-induced stall. That's labor the API path externalizes to
  Anthropic's SRE team. Owning serving means you pay it — in time, not the
  power bill — and you pay it again every time the stack drifts. It's the
  largest real cost here.
- **Utilization.** Amortized capex-per-run is only low if the boxes stay busy.
  An idle Spark has a *high* effective cost-per-token. For an occasional
  personal pipeline the honest economic case is thin; the real justifications
  are data-locality, iteration-freedom, and learning.
- **The baseline caveat.** "Better than what I had" is partly "many iterations
  vs the one under-iterated run I could afford before." The old result wasn't
  the ceiling of the API approach — a well-iterated API run could plausibly
  have matched it, just unaffordably. The defensible claim is *affordable
  iteration beat unaffordable perfection*, not "local models are better."
- **Free experiments erode discipline.** A budget imposes a crude "think before
  you spend" rigor. Remove it and it's easy to run 50 variants where 5 would
  have been informative. Local makes *you* supply the experimental discipline —
  clear hypotheses, stopping rules — that the meter used to impose.

---

## A decision rule: when to route a workload to local capex

Route a task to local (capex) compute when it is:

1. **Verifiable / reviewable** — errors get caught downstream (e.g. the
   per-dossier Uncertainty blocks + `source_quote`s), so a smaller model's
   mistakes don't silently propagate.
2. **High-volume or iterative** — enough sustained work to amortize the capex,
   and where metered opex would compound across runs.
3. **Bounded** — small enough context / task that a smaller model handles it.
4. **Latency-tolerant** — minutes/hours is fine; you're not in a tight
   interactive loop.

In this project: **aggregation** hit all four → local. **Synthesis** hits only
(1) — low-volume, cross-entity, prose-quality-sensitive → keep on the big API
model (cheaply, now that its input is curated to ~40K tokens).

---

## P&L footnote

`spark1` was bought by me; `spark2` by my employer. So from my personal P&L the
second box's compute is *free* (modulo power). That's an allocation accident,
not a transferable principle — but the generalizable version is real and
common: **is there a sunk, underutilized inference asset I can route verifiable
bulk work onto?** For a lot of orgs sitting on idle GPUs, that question is the
unlock — the skill is knowing which workloads (verifiable, iterative, bounded)
can move from API opex to owned capex without losing quality.

---

*Provenance: the pipeline lives in the CampaignGenerator repo
(`ensemble_extract.py`, `facts_to_state.py`, `config/agents/state_aggregate.md`,
`synthesise_world_state.py`). The robustness fixes are commits on its
`feat/ensemble-speculative` branch. This doc is the retrospective argument, not
the code.*
