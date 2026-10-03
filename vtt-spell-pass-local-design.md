# vtt-spell-pass on the Spark — design

**Status:** design, not built. Evidence gathered 2026-10-02/03 against Out of the
Abyss review records. Nothing in the `vtt-spell-pass` skill has changed yet.

**Goal:** run the judgment steps of `/vtt-spell-pass` (batch mode) on local
models — Clef on spark2, `qwen3.8-flash-next` on spark1 — instead of Sonnet,
**without moving any ruling away from the GM.** As with every Spark project, the
point is calibration: learning where local models hold up and where they break.
The Anthropic path would still be faster and stronger; this doc does not claim
otherwise.

## The shape

```
batch_scan.py                deterministic — unchanged (Phase 0/1, sibling lookups)
      │  every unknown capitalised token + tape context
      ▼
Clef 27B  — TRIAGE           choice {filler, rules, realworld, campaign}
      │  forward if P(campaign) ≥ cutoff (~0.06–0.10)      ~0.4 s/token
      │  everything else → listed in the page footer ("Left alone" / "Table chatter"),
      │                     never silently dropped
      ▼
qwen3.8-flash-next, thinking ON — PICK
      │  "which canon spelling, word for word?"  over deterministic top-8
      │  candidates + OTHER + LEAVE                         ~33 s/item
      │  ran out of thinking budget → card flagged "model unsure", no proposal
      ▼
merge_proposals.py / stage_c.py → review artifact → GM rules every card
```

## What decision each step removes from the GM

Per the global LLM-pipeline rule.

| Step | Decision | Removed from GM? | Why that is acceptable |
|---|---|---|---|
| Clef triage | which tokens get a model's attention | **No** — it is ordering, not dropping | Cleared tokens still appear in the footer lists the GM already reads today; any can be pulled back. A miss here is the dangerous failure ("a wrongly rejected name deletes a real one"), so the cutoff is set for **zero** missed names, not for throughput. |
| qwen-think pick | draft spelling on a card | **No** | Every card is still Approve / Reject / Discuss with keep-verbatim. Abstentions are surfaced as abstentions. |
| Second-transcription reading (`ordinary_words` etc.) | feeds the auto-dismiss gate | **Yes, today** | **Not moved to a local model by this design.** Untested; stays where it is until measured. |

## Evidence

All tests replay rulings the GM already made. Scripts and labels:
`clef/spell-pass-eval/` (see *Reproduce*).

### 1. Clef triage — the step it is good at

864 tokens from the 9 OOTA sessions whose review footers enumerate what was
left alone: 628 left alone, 152 table chatter, 84 that became approved name cards.
A **miss** = one of the 84 filtered out before reaching the expensive step.

| Triage model | Forwards | Real names missed / 84 | Latency |
|---|---|---|---|
| **clef 27B**, P(campaign) ≥ 0.10 | **37 %** | **0** | 0.4 s |
| clef 27B, P(campaign) ≥ 0.05 | 56 % | 0 | 0.4 s |
| clef-flash, best cutoff | 66 % | 2 | 0.13 s |
| qwen, thinking off | 24–38 % | 5–6 | 2.0 s |

Held-out check: cutoff tuned on half the sessions (then halved for margin),
tested on the other half → **0 misses**, forwarding 44 % and 51 %.
Names the weaker models drop: `Heel Strike`, `Custadis`, `Pick / Shine`, `Zoom`.

### 2. Picking the canon spelling — the step Clef is bad at

335 items: GM-approved name rulings (gold = approved canonical) plus 12 GM-ruled
chatter (gold = LEAVE). Options = deterministic top-8 candidates from canon
**names only** (glossary bold column, registry names+aliases, module inventories —
never the glossary wrong-form column, which contains these very rulings) +
`OTHER` + `LEAVE`. "Wrong name" = a confident pick of the wrong canon spelling,
the error a tired reviewer approves.

| Model | Correct | Wrong names | Confidence informative? | Latency |
|---|---|---|---|---|
| clef-flash | 30 % | 52 | no | 0.25 s |
| clef 27B | 66 % | 71 | barely (top half 78 %) | 0.75 s |
| qwen, thinking off | 76 % | 56 | no (top half 80 %) | 3.8 s |
| **qwen, thinking on** (4K budget) | **94.5 % of the 220 answered** | **7** | yes | ~33 s; ~4 min when it runs out |
| clef ∧ qwen-think agree | 95.8 % of 166 | 5 | — | — |

- Of qwen-think's 115 non-answers, 51 are the 65 items whose right name was not
  in the candidate list — it runs out of budget on exactly the hard ones. That is
  a safe abstain, not a failure.
- Clef ignores word-level instructions: told not to add words, it still turned a
  spoken `Misrim` into **Ilvara Mizzrym** (the "inserted a surname nobody spoke"
  failure the skill warns about). qwen obeyed.
- Gating qwen with Clef's confidence added nothing.

### 3. Clef predicting the GM's verdict — rejected, and why

First experiment: Clef scored "will the GM approve this card?" on 525 cards.
AUC 0.887 on Claude-written card text, 0.911 on raw-source cards — but a one-line
rule ("does the card propose a fix at all?") scores **0.908**. Split by card
type: ~0.66 on proposals (only 10 non-approvals — not measurable), and on
open questions 0.80 *with* Claude's prose vs **0.40** without it. It was reading
Claude's hedges back. The verdict target is also saturated: the GM approves
97.6 % of proposed fixes.

## Known gaps (do these before trusting the pipeline)

1. **Candidate recall caps the pick step at ~80 %.** The right name is in the
   deterministic top-8 only 258/323 times (spelling + exact Double Metaphone).
   Misses: `Vira`→Ilvara, `Peto`/`Boo-boy`→Buppido, `Darrow`→derro,
   `Crag Gorm`→Cairngorm, and the house convention of writing Kostadis as `GM`.
   OTHER catches these as abstains, but every miss is a card the model cannot help with.
2. **qwen-think budget.** Re-run at 8K: does it convert abstains into right
   answers, or into wrong names? The answer decides the budget.
3. **The test set flatters Sonnet.** Every pick item is a Sonnet proposal the GM
   approved, so the set cannot show Sonnet's misses — this is "how close do the
   local models get to approved work", not a head-to-head.
4. **Triage labels are weaker than card rulings** ("left alone, not pulled back
   from the footer"), and 9 sessions / 84 names is small. Re-measure on the next
   campaign's records (Phandalin, obelisk) before reusing the cutoff there.
5. **Second-transcription reading** (feeds auto-dismiss) — untested locally.
6. **Capacity.** spark2 serving Clef leaves spark1 as the only qwen endpoint.
   At ~33 s/item, qwen-think is fine for the backlog and slow for a live session.
7. **Clef ≠ Workers AI not verified** (see `clef-observations.md`). A silently
   wrong head load would make every number above meaningless; the triage result
   is the one to re-run after that check.

## Operational notes from the run

- `sibling_context.py` re-reads and fuzzy-matches the whole sibling transcript on
  every call (~30–60 s each on long Zoom tapes); 221 lookups took ~2 h serially,
  ~17 min with 14 threads. In a live local pipeline this, not the model, is the
  bottleneck — load the sibling once.
- Never wait on a chained job with `pgrep -f "<cmd>"`: the waiting shell's own
  command line contains the string, so it matches itself forever (cost 3.5 h).

## Reproduce

Scripts in `clef/spell-pass-eval/`, run from that directory; they read the OOTA
campaign directly from `~/out-of-the-abyss/out-of-the-abyss`.

| Script | Produces |
|---|---|
| `triage.py <clef\|clef-flash\|qwen>` + `tri_score.py` | §1 — needs `triage_labels.json` |
| `build2.py` → `build3.py` → `build_pick.py` (`K=8`) | `pick.json` for §2 (build2 is the slow sibling step) |
| `run_pick.py <clef\|clef-flash\|qwen\|qwen-think>` + `score_pick.py` | §2 |
| `build.py`, `run.py`, `run2.py`, `analyze.py` | §3 |

Endpoints: Clef `http://192.168.1.121:8002/v1/systemone`, qwen
`http://192.168.1.147:8001/v1/chat/completions` (model `qwen3.8-flash-next`,
`chat_template_kwargs.enable_thinking`).

## Related negative result — Clef for CampaignGenerator ensemble typing

Tested 2026-10-03 because ensemble extraction assigns every fact a `type`
(npc/monster/faction/location/object/event/thread/date) and `ensemble_merge.py`
groups on (type, subject): an entity typed two ways becomes two dossiers. The GM
had hand-merged 109 such groups (`docs/ensemble/.type_merge_decisions.json`), which
supply the labels. Scripts: `clef/ensemble-typing-eval/`.

**Per entity** (92 merged groups, facts canonicalised the way `facts_to_state.py` does):

| Method | Matches GM primary type |
|---|---|
| majority vote of the passes' own types (free) | 82 / 92 |
| clef 27B / clef-flash / fast qwen | 82 / 80 / 80 |
| vote, model breaks ties under 60 % | 87 / 92 — same for all three models |

Most "errors" are GM filing conventions, not model confusion — demon lords filed
`monster` (Zuggtmoy, Juiblex, Demogorgon, Ogrémoch, Yestabrod) except Yeenoghu
(`npc`); temples and Deepking Tarngardt filed `faction`. That inconsistency is the
GM's to settle.

**Per fact** (Clef re-types each fact after qwen extracts it — the extraction-phase idea):

| Re-typer | Split entities among the 92 (was 78) | New splits among 55 consistent entities | Facts drained to event/thread/date |
|---|---|---|---|
| clef 27B | 29 | 7 | 255 + 177 |
| clef-flash | 46 | 13 | 120 + 68 |
| fast qwen | 38 | 13 | 28 + 19 |

The 92-entity set was selected on extraction's mistakes, so any re-typer looks good
there; the control shows each one splits entities that were fine. Projected over
OOTA's ~470 consistent dossiers, re-typing creates about as many splits as it fixes
(clef 27B) or more (flash, qwen). **Per-fact typing splits long-lived entities
whichever model does it** — an entity with dozens of facts gets one odd label
eventually, and Clef's determinism does not help because every fact is a different
input.

**Fix (no Clef needed):** type once per canonical subject — vote, a tie-break for
close votes, and a GM-written filing-convention list — then stamp that type on every
fact before bundling. See the CampaignGenerator design
`docs/design/EntityLevelTyping_proposal.md` (branch `design/entity-level-typing`, worktree `~/src/CampaignGenerator-entity-typing`).

## Next step if this goes ahead

Implement in a scratch worktree of `~/src/mytools` (never the live checkout):
a `--triage clef` stage in `batch/batch_scan.py` that annotates rather than
filters, and a local Stage A runner that calls qwen-think in place of the Sonnet
agent and writes the same `proposals.json` contract — so `merge_proposals.py`,
the artifact, and the GM's review are untouched.
