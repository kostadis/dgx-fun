# Qwen 2.5 14B AWQ on voice-anchored prose — what actually works

**Date:** 2026-05-16
**Hardware:** DGX Spark, vllm-chat container (port 8001), unchanged from `current-setup.md`
**Local model:** `Qwen/Qwen2.5-14B-Instruct-AWQ`
**Task tested:** Pass 5 of `CampaignGenerator/session_doc.py` — first-person memoir narration
of a single scene, with `--prose-mode` + `--reflections` + `--narration-genre` directive + voice
files. All runs against the same scene (Storm Giants, 2025-03-12, scene 2,
"The Escape Plan and the Call to Battle", narrator = Orsik).

## Revision history

- **v1**: claimed "Qwen 14B is a paraphrase engine, period." Wrong — overcorrection from two broken runs.
- **v2**: corrected to "Qwen has a default-register gravity well; register directives provide a thin overlay only." Also claimed "my prompt edits hurt Sonnet meaningfully" based on a single Sonnet+new-prompt run that produced 375 words of generic-adventure. **That claim is no longer supported** — a re-run of the same combination produced 737 words of solid comic-noir.
- **v3 (this version)**: re-evaluated after a full 2x2 prompt-vs-model experiment (`/tmp/prompt-tests/`). The strong anti-prompt-edit claim is walked back. The Qwen findings stand; the Sonnet/Opus + scaffold data is added as ceiling-anchor calibration.

---

## The headline finding (updated)

**The scaffold pattern works at every model tier we tested.** Given a hand-built scaffold (beats + attributed quotes + staging), three different model tiers produced structurally correct output:

- **Qwen 14B AWQ (local)**: structurally correct, comic-noir register fails to land (1-2% phrase-level overlay), defaults to generic literary-adventure prose, ~480 words.
- **Sonnet 4.6 (Anthropic)**: structurally correct, comic-noir register sustained throughout, ~750-810 words.
- **Opus 4.7 (Anthropic)**: structurally correct, comic-noir register sustained, deeper character/context weaving, ~1000-1037 words.

**The model axis dominates the prompt axis** on this task. Prompt edits cause modest compression (~5-10% word loss) and slight plan-focus attenuation, but do not damage register or attribution. The big variance in output quality comes from which model is rendering the scaffold.

**Qwen 14B's ceiling** is *not* "can't do prose" or "can't follow scaffolds" — it's specifically that the model lacks a steerable register engine. It defaults to generic literary-adventure prose, and that default cannot be overridden by genre directives (even when the directives are excellent, even when the scaffold is well-built, even when the prompt has length floors and tail-reminders).

## The 2x2 — prompt × model

All four cells use the same scaffold, same `--narrate-tokens 20000`, same `--prose-mode --reflections`, same contradictory genre directive verbatim:

> "High-fantasy epic adventure — First-person comic-noir fantasy memoir — observational, dry, irony-forward, alive to absurdity. NOT epic-fantasy adventure prose; NOT literary-introspective register."

| | **Sonnet-prompt** (CampaignGenerator/ checkout, production) | **DGX-prompt** (unified-pipeline checkout, + length floor + DM-attributor BAD example + genre tail-reminder) |
|---|---|---|
| **Sonnet 4.6** | **810 words.** Clean comic-noir. Closes with "Today was still good to die." Convincing layered closing on Uncle Joon's door. | **737 words.** Clean comic-noir. Drops the "Today is good to die" book-end. Slight compression. |
| **Opus 4.7** | **1037 words.** Deepest character weaving (Vardis, Waterdeep, Headband twice). Book-ends with "Today is good to die." **Production ceiling on this scene.** | **993 words.** Strong but slightly compressed. Opens with "the kind of thing you don't look away from" — adjacent to the directive's forbidden patterns. One Headband mention, no Waterdeep callback. |

**Reading the table:** Move horizontally (prompt axis) → modest changes. Move vertically (model axis) → meaningful changes in length and depth.

## Full run history (one per row, all same scene)

| run | input | model | prompt | words | result |
|---|---|---|---|---:|---|
| 1 | raw VTT | Qwen 14B | original | 380 | broken — verbatim source-quoting, mechanical language |
| 2 | raw VTT | Qwen 14B | DGX-prompt (my edits) | 430 | worse — more source-quoting, "Persuasion three" fabricated |
| 3 | hand-cleaned scene events | Qwen 14B | DGX-prompt | 270 | shorter — less material to mirror |
| 4 | **scaffold** | Qwen 14B | DGX-prompt | 482 | structurally correct, register stayed default |
| 5 | scaffold | Qwen 14B | DGX-prompt + epic-fantasy directive | 457 | structurally correct, ~5-8% epic markers (vs ~1-2% for comic-noir) |
| 6 | scaffold | Sonnet 4.6 | DGX-prompt | 375 | **anomaly** — generic adventure; not reproduced in run 9 |
| 7 | scaffold | Sonnet 4.6 | Sonnet-prompt | 843 | comic-noir landed throughout (original Anthropic baseline) |
| 8 | scaffold | Sonnet 4.6 | Sonnet-prompt | 810 | comic-noir, slight variant of run 7 |
| 9 | scaffold | Sonnet 4.6 | DGX-prompt | 737 | comic-noir landed throughout — **walks back v2's claim about run 6** |
| 10 | scaffold | Opus 4.7 | Sonnet-prompt | 1037 | deepest weaving observed, book-ended on "Today is good to die" |
| 11 | scaffold | Opus 4.7 | DGX-prompt | 993 | strong, slightly compressed, one near-miss on forbidden-pattern directive |

Run 6 (Sonnet+DGX-prompt → 375 generic-adventure) was either a sampling artifact, a transient state, or some uncontrolled difference between runs that we didn't isolate. The reproducibility evidence (run 9) puts the "DGX-prompt crashes Sonnet" hypothesis to bed.

## The architectural unlock: the scaffold pattern

This is the load-bearing finding. Before scaffolds (runs 1-3), even excellent models failed. After scaffolds (runs 4-11), every model produces structurally correct output. The scaffold is what makes the pipeline viable.

Scaffold structure (example: `campaigns/stormgiants/summaries/20250312/scene_extractions_new/02_*.scaffold.md`):

```
[Scene 2] The Escape Plan and the Call to Battle
Narrator: Orsik
Focus: Orsik is outside in the amphitheater. Thistle flies out. Orsik asks
       what happened. Thistle explains.

- Orsik is outside, looking into the hole that fake Laela flew out of.
- Thistle flies out solo.
- Orsik asks him what happened.
- Thistle tells him.
- [bulleted beats in narrative order]

## Quotes to place

### [Thistle's Failed Persuasion — One-Dimensional Chess]

<!-- Laela's Reaction -->
Laela: "We're gonna go up there where the big battle is going to occur?"

<!-- Thistle's retort -->
Thistle: "One-dimensional chess."
...
```

The scaffold IS the human-reviewed structure from the pattern in `~/.claude/CLAUDE.md` ("LLM extracts → human reviews and imposes structure → LLM renders inside that structure"). With it, every model from Qwen 14B to Opus 4.7 can render correctly. Without it, models fail in different but consistent ways (Qwen quotes source verbatim; Sonnet improvises wrong staging; Opus invents content).

The scaffold investment pays off at every tier. Build it once, plug in any renderer.

## What the prompt-edit axis actually does (revised)

The DGX-prompt adds three things to the production prompt:
- Length floor: "Target 600-900 words"
- DM-as-attributor BAD/GOOD examples
- Genre directive repeated at the tail of the prompt

Observed effects across runs 9-11 (Sonnet+DGX, Opus+Sonnet, Opus+DGX vs Sonnet+Sonnet baseline):

- **~5-10% word compression.** Sonnet: 810 → 737. Opus: 1037 → 993. Real but small. Probably driven by the explicit numerical range — capable models tend to read numerical targets as caps even when they're framed as floors.
- **Slight plan-focus attenuation.** Both Sonnet-prompt runs land on "Today is good to die" as a closer (or book-end). Sonnet+DGX-prompt drops it entirely; Opus+DGX uses it once at the open but not the close. Hypothesis: the added DM-attributor block and tail genre reminder crowd out attention to the plan's focus directive.
- **Slight reduction in context-weaving.** Opus+Sonnet-prompt threads Vardis, Waterdeep, and the Headband twice through the prose. Opus+DGX-prompt has one Headband mention and no Waterdeep. Probably the same cause as the focus attenuation — added prompt content reduces room for context callbacks.
- **No register damage.** Comic-noir lands convincingly in all four cells, including run 6's would-be reproduction (run 9 = 737 words of comic-noir).
- **No attribution damage.** All four runs handle the scaffold's quote attributions correctly.
- **Edge case: Opus+DGX-prompt opens with a forbidden-pattern near-miss.** *"The hole in the ground was the kind of thing you don't look away from once you've started looking."* The directive forbids "the shape of X / the quality of X / X had a [shape|quality] / that particular quality" — "the kind of thing" is structurally adjacent. The other three cells don't do this. Possibly the DGX-prompt's tail genre reminder is interpreted by Opus as license to be slightly more rhetorical than the directive intends.

**Net assessment:** the DGX-prompt edits are a modest negative on Anthropic models. Not catastrophic, but they don't earn their complexity for the API path. They were designed for Qwen 14B's failure modes (mechanical language survival, DM-attribution drift), and on a model that doesn't have those failure modes, they're prompt bloat.

## The Qwen findings (still hold)

The Qwen 14B AWQ findings from v2 still hold — only the prompt-edit-hurts-Sonnet claim was overstated:

- **Default-register gravity well.** Qwen always writes in generic literary-adventure register. Register directives provide a thin overlay (~5-10% for well-trained registers like epic-fantasy; ~1-2% for niche like comic-noir).
- **Stable failure modes under scaffold:** ~10-20% attribution drift, scaffold-bullet-as-dialogue artifacts, telling-not-showing dialogue tags.
- **Qwen 14B + scaffold IS viable** for tasks where "competent generic literary-adventure prose" is acceptable. Not viable for tasks where a specific niche register is load-bearing.

## Operational rules of thumb (revised)

**For session narration specifically:**

- **Production:** Sonnet 4.6 + production prompt + scaffold. ~810 words of clean comic-noir, attribution-faithful, plan-focus respected, good context-weaving. This is what works today.
- **Premium tier:** Opus 4.7 + production prompt + scaffold. ~1037 words, deeper character/context weaving, book-ends with plan-focus lines. Costs ~5-7× per token. Worth it for capstone scenes (session opener, climax, character departures). Not worth it for every scene.
- **Budget tier (experimental):** Qwen 14B AWQ on the DGX + scaffold. ~480 words of generic literary-adventure prose. Structurally correct. Useful as a first-draft engine if you're willing to register-rewrite in a second pass.

**For the prompt-edit axis specifically:**

- **The DGX-prompt edits are a slight net negative on Anthropic models.** They were designed for Qwen failure modes. Consider gating them behind `--dgx-endpoint` (or `--strict-prose-mode`, or similar) rather than running them on the Anthropic path. Production prompts should stay clean.

**For task-type generalization:**

- Use Qwen 14B + scaffold for: closets, dossiers, paraphrase-shaped content, anything where "competent generic literary register" is the deliverable.
- Use Anthropic + scaffold for: anything where specific register adherence is load-bearing (comic-noir, Hemingway, niche tones).
- The scaffold is universal. Build it once per scene; plug in the appropriate renderer.

## Workflow the data supports

The actual end-to-end pipeline this experimental record points at, in two modes depending on mood and time:

### Mode A — Qwen-iterate → Opus-polish (when you want a finished render)

1. Hand-build the scaffold (beats + attributed quotes + staging).
2. Render with Qwen 14B on the DGX. Read the output.
3. If the output is structurally incoherent (beats jumbled, attribution drift, staging error), the **scaffold has a gap.** Fix the scaffold, re-render with Qwen, repeat. This iteration loop is zero-API-cost and sub-minute per pass.
4. Once Qwen renders cleanly, the scaffold is ready. Make one final call to Opus 4.7 with the same scaffold for the polished comic-noir output.

The thing being iterated on is the scaffold, not the prose. Qwen output is a **test of scaffold quality** — if it renders cleanly in Qwen, it'll render brilliantly in Opus. Cost: ~one Opus call per scene plus unbounded free Qwen iteration. Probably 5-10× cheaper than naive Sonnet-from-scratch iteration where every attempt costs tokens.

### Mode B — Qwen-draft → human-write (when you want to write it yourself)

1. Hand-build the scaffold.
2. Render with Qwen 14B on the DGX.
3. Use the Qwen output as a launching point — beats sequenced, dialogue placed, staging respected. The register/voice work that Qwen can't do is exactly what you do anyway.

The local model functions as a competent first-draft assistant rather than an unreliable substitute for finished prose. Useful when you want the scene in your own voice but don't want to start from a blank page.

### The underlying insight

**The scaffold is the most valuable artifact in the pipeline.** More valuable than any rendered output. A good scaffold can be rendered by Qwen, Sonnet, Opus, or by hand. It survives model swaps, register changes, even mood-driven workflow changes. Investing tooling effort in scaffold construction — a scaffold-generation pass, scaffold validation, scaffold-aware review — has higher long-run ROI than investing in better renderers, because the scaffold is what makes every renderer viable.

The architectural rule from `~/.claude/CLAUDE.md` ("LLM extracts → human reviews and imposes structure → LLM renders inside that structure") names the scaffold as the human-reviewed structure. This workflow makes the scaffold the central artifact and lets the renderer choice be tactical — driven by cost, mood, and how finished the output needs to be.

## Open experiments

- **Two-pass workflow.** Qwen drafts the structural render (cheap, fast); Sonnet/Opus does a register-pass rewrite given the scaffold + Qwen draft. Should be cheaper per scene than full-Sonnet while keeping register quality. Untested.
- **Qwen 72B AWQ on the same scaffold.** Tests whether the default-register gravity well is scale-bound inside Qwen-family. Hypothesis says no — it's a training-distribution issue, not a size issue. Worth one ~3-min docker swap to confirm.
- **Llama 3.3 70B Instruct on the same scaffold.** Different training distribution. Tests whether the gravity well is Qwen-specific.
- **Gate the DGX-prompt edits behind `--dgx-endpoint`.** Cleanly separates the Qwen-specific bureaucracy from the Anthropic path. Roughly: in `session_doc.py`, check if `endpoint` is set before applying the new length floor, DM-attributor BAD example, and tail genre reminder. Anthropic path keeps the cleaner production prompt; DGX path gets the heavier scaffold rules.

## See also

- `current-setup.md` — vllm-chat container config
- `gemma-vs-qwen-ab.md` — earlier Qwen-works finding for closet generation (consistent with the rules of thumb above)
- `model-comparisons.md` — candidate next models with VRAM math
- `dnd-session-prep-with-opus.md` — the workflow being localised
- `CampaignGenerator-unified-pipeline/session_doc.py` — `build_narrate_system` and `PROSE_MODE_INSTRUCTION` (Pass 5 prompt assembly with the DGX-prompt edits)
- `CampaignGenerator/session_doc.py` — production version without the DGX-prompt edits
- The scaffold format example: `campaigns/stormgiants/summaries/20250312/scene_extractions_new/02_the_escape_plan_and_the_call_to_battle.scaffold.md`
- The 2x2 experiment outputs:
  - `/tmp/prompt-tests/sonnet-test-with-dgx-spark-prompt/` (737 words)
  - `/tmp/prompt-test/sonnet-test-with-sonnet-prompt/` (810 words, note: singular `prompt-test`)
  - `/tmp/prompt-tests/opus-test-with-dgx-spark-prompt/` (993 words)
  - `/tmp/prompt-tests/opus-test-with-sonnet-prompt/` (1037 words)
