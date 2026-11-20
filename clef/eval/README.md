# Frozen-dataset rerun of the decision-models paper

`paper/main.tex` reports experiments E1–E7. The original scripts in `../spell-pass-eval`,
`../ensemble-typing-eval`, `../social-npc-eval` and `../finetune` rebuilt their inputs from the live
campaign tree on every run, so results drifted as the campaign grew. This directory freezes the
inputs once and runs every model against the same bytes.

| file | what |
|---|---|
| `freeze.py` | snapshot every experiment input (with the model-facing `state` already rendered) into `~/data/decision-eval/<version>/`, read-only, plus `MANIFEST.json` (sha256 + counts + source commits) |
| `tasks.py` | questions, prompts and model endpoints, copied verbatim from the original scripts |
| `run.py MODEL EXP...` | run one model on E1–E5; results to `<data>/results/<part>/<model>.json`; never overwrites |
| `e6.sh` | E6 fine-tuning of Decision-2.0-Nox-4B on spark2 from the frozen items |
| `score.py` | every table in the paper, from the results |
| `MANIFEST-v1.json` | copy of the frozen set's manifest (hashes only, no content) |

The dataset itself is **not in the repo**: it quotes play sessions verbatim and the table-chatter
labels name real people. It lives at `~/data/decision-eval/v1/` on the workstation.

## Reproduce

```bash
# 1. build inputs from the campaign (slow: build2.py shells out per card)
cd ../spell-pass-eval && python3 build.py && python3 build2.py && python3 build3.py && python3 build_pick.py
cd ../ensemble-typing-eval && python3 build_ent.py && python3 build_fact.py && python3 build_ctrl.py \
  && for t in 0.5 0.55 0.6 0.67 0.75; do python3 replay_489.py $t; done
# 2. freeze
cd ../eval && python3 freeze.py ~/data/decision-eval/v2
# 3. run (serve each model first; see current-setup.md and spin-up-clef.sh / spin-up-decision2.sh)
DATA=~/data/decision-eval/v2 python3 run.py nox e1 e2 e3 e4 e5     # ... per model
DATA=~/data/decision-eval/v2 ./e6.sh
DATA=~/data/decision-eval/v2 python3 score.py --json scores.json
```

Models: `clef`, `clef-flash` (spark2:8002), `lux` (:8003), `kai` (:8004), `nox` (:8005),
`jev` (TypeSafe hosted API, key in `~/.jev-api`), `qwen` / `qwen-think` (qwen3.8-flash-next on
spark1:8001, reasoning off / on). Local SystemOne models run single-stream; Jev's latency
includes the internet round trip and is not comparable to the GB10 numbers.

Running `jev` sends the frozen states (campaign transcript excerpts) to TypeSafe's API.

## Google Gemini Decision 1

`gemini` runs the same frozen E1–E5 states and unchanged question definitions through
[Google's Decisions API](https://aistudio.google.com/docs/decisions-api), using
`gemini-decision-1`. The adapter maps `state` to `input`, authenticates with
`x-goog-api-key`, and retains the returned probabilities, confidence and token usage.
It uses REST directly, so no additional SDK is required.

```bash
# Key lookup: GEMINI_API_KEY, GOOGLE_API_KEY, then GEMINI_API_KEY_FILE (default ~/.gemini-key).
# The key file may contain a bare key or KEY=value. Never commit it.
DATA=~/data/decision-eval/v1 python3 -u run.py gemini e1 e2 e3 e4 e5
DATA=~/data/decision-eval/v1 python3 score.py --json /tmp/decision-scores.json
```

The runner checks API access with a synthetic input before writing any results.
Results use the existing `results/<part>/gemini.json` schema and never overwrite
previous files. E1 prose keeps its two questions together; E5 keeps outcome and
target together per scenario. Different frozen rows remain separate requests.
Unfinished Gemini parts checkpoint each completed row to `gemini.partial.jsonl`
beside their final result; rerunning resumes that prefix. Each Gemini request has
a 60-second socket timeout, and failed rows use the runner's existing retries.
Like the other decision models this runs single-stream. Its latency includes the
internet round trip, so compare accuracy on the same items and report hosted
latency separately from GB10 service latency. E6 trains local weights and does not
apply to this hosted model; E7 latency is computed from the E1–E5 results.

Running `gemini` sends frozen states (including campaign transcript excerpts and
table chatter) to Google's API. Google's launch price is $0.10 per million input
tokens and no output-token charge; the saved `usage` records actual input counts.

### Frozen v1 results — 2026-11-11

Completed all nine E1–E5 parts: **6,316 scored requests, zero failed rows**.
The input files still match every hash in the frozen manifest. Raw results remain
beside the private dataset; the repository contains only
[aggregate scores for all available models](gemini-v1-scores.json) and a
[run audit with counts, usage, latency and result hashes](gemini-v1-run.json).

| Measure | Gemini Decision 1 | Clef 27B | Jev | Qwen fast |
|---|---:|---:|---:|---:|
| E2 spelling accuracy, 497 items | **82.5%** | 66.4% | 69.0% | 76.1% |
| E2 wrong-name substitutions (lower is better) | 53 | 109 | **28** | 87 |
| E3 held-out forwarding, two folds | **27% / 29%, zero missed names** | 43% / 49%, zero missed | 51% / 47%, zero missed | 24%, 6 missed (hard classifier) |
| E4 repair-set fact accuracy | 87.8% | 76.8% | 84.2% | **88.3%** |
| E4 residual split entities, repair set | **25** | 29 | 35 | 38 |
| E4 new split entities, control set | **4** | 7 | 9 | 12 |
| E5 expected directional changes, 12 pairs | **12/12** | 11/12 | 11/12 | 9/12 |

Gemini is strongest here at spelling selection and triage. Its most-confident
half of spelling answers is 93.5% accurate, but it still makes more wrong-name
substitutions than Jev. On E1, AUC among proposals containing a correction is
0.476 on upstream prose, 0.684 on raw sources, and 0.742 with wider context
(Clef: 0.549 / 0.722 / 0.745). The original warning about verdict prediction
depending on presentation still applies. On E4 entity-level typing, Gemini alone
gets 80/92; using it only below a 60% extraction-vote majority gets 87/92,
matching the best saved hybrids. Per-fact retyping still creates four new control
splits, and the split metric excludes non-entity labels: Gemini assigns 42 repair
facts and 17 control facts to event/thread/date. These are bounded campaign tests,
including only 84 real names in E3 and 12 hand-written NPC pairs in E5.

The saved requests have **275 ms overall median latency**, with per-part medians
267–285 ms, including the hosted round trip. Recorded input usage is **2,891,653
tokens**, or **$0.289** at the launch price; this excludes smoke/preflight calls,
the abandoned first E2 attempt, and any unreported usage from retries. E2 initially
stalled under the original 600-second timeout; it was restarted with a 60-second
socket timeout and per-row checkpoints, retaining all completed E1 results.
Later stalls recovered on retry. Saved request timings exclude socket-timeout
attempts retried by the outer row runner, so they do not describe those pauses.
The Qwen baseline used its original concurrent generative run; hosted and GB10
latencies should be interpreted with their respective serving configurations.

### Candidate-retrieval follow-up

The paired follow-up holds the 497 E2 examples, Gemini model, question wording
and transcript states fixed and changes only candidate options. All lists were
prepared before the API sweep. Canonical names come from the original campaign
commit, with the glossary's wrong-form column excluded. The four alternatives
completed **1,988 requests with zero failed rows**.

| Retrieval | Correct name offered / 485 | Actual correct spellings / 485 | Wrong-name substitutions |
|---|---:|---:|---:|
| Original top 8 | 385 (79.4%) | 347 (71.5%) | 53 |
| Original top 8 + next nearest names, total 32 | 428 (88.2%) | 387 (79.8%) | 49 |
| Sibling/context ranking, top 8 | 388 (80.0%) | 352 (72.6%) | 49 |
| Sibling/context ranking, top 32 | 428 (88.2%) | 389 (80.2%) | 49 |
| Context + earlier-session corrections, top 32 | **442 (91.1%)** | **399 (82.3%)** | **47** |

Here **actual correct spellings** excludes correct OTHER decisions: admitting
that a name is missing is useful, but it does not recover its spelling. That
distinguishes these percentages from the earlier 82.5% E2 decision accuracy,
which includes OTHER and LEAVE across all 497 items. The follow-up still scores
all 497 items, including the same 12 table-chatter controls; wrong-name counts
include those controls.

The simple widening accounts for most of the improvement. It recovers 40 net
additional spellings without adding context machinery. Adding context at 32
options adds two more, and earlier-session correction variants add another ten.
The strongest arm gains 63 correct spellings and loses 11 previously correct
ones, for a net gain of 52. It correctly spells 48 of the original 100 names
that were missing from the options. Examples include Peto → Buppido,
Vira → Ilvara, and Crag Gorm → Cairngorm. At confidence ≥0.9 it proposes 172
names, of which 167 are correct; confidence still does not justify auto-applying
changes. Median request latency remains 263–287 ms across arms. Recorded input
cost for all four arms is $0.253, excluding preflight and unreported retry usage.

**Leakage and workflow checks.** Sibling retrieval aligns the token's surrounding
raw transcript against the independent transcript; it does not inspect gold
answers. Context ranking adds a small bonus for canonical names already visible
in the raw window. The history arm learns token → canonical associations only
from approved examples with strictly earlier session dates, excluding the same
date and future dates. This simulates sequential review; it does not reconstruct
the dates when corrections actually entered the glossary. Of the test rows, 24
already have an exact token correction in an earlier session and could normally
be handled by the deterministic glossary pass. Excluding those rows leaves 461
name examples: correct spellings rise from 326 to 378, so the full net gain of
52 persists on that subset. On the context8 control's 472 unchanged option maps,
all returned choices match the original run.

These are retrospective results on one campaign. They support testing 32
candidates in the existing GM-reviewed flow, followed by earlier-correction
retrieval; fresh sessions should validate the benefit before changing defaults.
The live spell-pass skill, second-transcript dismissal rules and approval gates
have not been changed.

Reproduce with `retrieval_experiment.py prepare`, `run`, then `score` and a
private `--workdir`. Preparation uses the campaign Git snapshot plus the original
`dmetaphone.py` helper (override its location with `--phonetics`). For example:

```bash
python3 retrieval_experiment.py prepare --workdir /tmp/retrieval-v2
python3 -u retrieval_experiment.py run --workdir /tmp/retrieval-v2
python3 retrieval_experiment.py score --workdir /tmp/retrieval-v2 --output /tmp/retrieval-v2-scores.json
```

Private candidate lists, row-level responses, manifest and run audit are archived
at `~/data/decision-eval/retrieval-v1/`. Only
[aggregate results and audit hashes](gemini-retrieval-v1-scores.json) are tracked
here. Each arm checkpoints completed rows and resumes them on rerun.

### Likely-yes list with one bulk approval

`card_filter_review.py` builds the requested review-mode experiment: model-yes
proposals appear as compact token → replacement rows, with one explicit bulk
approval. All other items retain Approve / Reject / Discuss cards. Any list row
can be pulled into individual review, including after bulk approval; doing so
revokes that row's previous approval. Both groups keep the original proposal
and source evidence available. No prediction is pre-marked as a human ruling.
The export records a separate decision for each original card.

This is a **historical replay**, reusing the E1 wide-source Gemini scores. It
does not modify the live skill, glossary or transcripts, and its distinct
review IDs prevent replay exports from matching a current session's items.
Historical verdicts are used only in the aggregate audit, never embedded in
the browser's review data. The page starts unmarked and reload clears unsaved
choices. Downloaded exports validate against the shared review-decision schema.

| Minimum model yes score | Flat-list rows | Individual cards | Historical nonapprovals placed in flat list |
|---|---:|---:|---:|
| 0.50 | 518 | 248 | 7 |
| 0.70 | 464 | 302 | 6 |
| 0.80 | 382 | 384 | 6 |
| **0.90 (prototype)** | **210** | **556** | **2** |
| 0.95 | 4 | 762 | 0 |

The dataset has 643 concrete proposals (627 approvals and 16 other rulings)
plus 123 open/no-change cards, which always stay in individual review. At 0.90,
208 of 210 flat-list proposals were historically approved and 14 of the 16
exceptions remain on cards. This is useful routing evidence with imperfect
separation; zero observed misses at a very restrictive cutoff on this same
dataset does not validate it for unseen cards. The old E1 warning still applies:
prose-only verdict scores separate the exceptions poorly, and the class balance
already makes an always-yes guess 97.5% accurate on concrete proposals.

```bash
python3 card_filter_review.py --output /tmp/card-replay.html
# Smaller single-session example:
python3 card_filter_review.py \
  --session 2026-01-12-chapter_43-harvesting-horror-in-the-underdark \
  --output /tmp/card-example.html
```

The saved example at `~/data/decision-eval/card-filter-v1/replay.html` has 13
flat-list proposals and 11 individual cards. `full-replay.html` contains all
766 items. Three Python checks and a headless browser test cover routing,
transcript escaping, bulk approval scope, exceptions before and after bulk
approval, notes, export, initially unmarked state, reload and mobile overflow.
[Threshold audit](gemini-card-filter-v1-scores.json).

### Expanded card-verdict corpus

`expand_card_corpus.py` inventories the spelling rulings under
`~/out-of-the-abyss/out-of-the-abyss/summaries/` and freezes an expanded corpus.
The original benchmark already used 59 sessions from this campaign. The wider
scan finds **1,118 labeled cards across 65 sessions**, adding **352**:
298 approve, 44 discuss and 10 reject. This includes a legacy spelling-review
layout skipped by the original `*/spell_review/decisions.json` glob. Three
backup decision exports contain the same labels as their current primary files
and are not counted again. Other review tasks (speaker attribution, consistency,
NPC dossiers) are not pooled into spelling approval labels.

The private snapshot is `~/data/decision-eval/card-filter-expanded-v1/`:

| Split | Cards | Approve | Discuss | Reject | Chatter |
|---|---:|---:|---:|---:|---:|
| Development, 60 sessions | 812 | 692 | 108 | 8 | 4 |
| Held out, five newer sessions | 306 | 268 | 34 | 4 | 0 |

Sessions recorded after the last original benchmark session are reserved whole
for validation, preventing cards from the same session appearing on both sides.
No exact feature duplicates cross the splits. Human labels and review notes are
stored under `target`, separate from an explicit feature allow-list. The 46
additional legacy queue items have a structured reconstruction rather than a
claim to the exact historical card wording. The 306 held-out examples retain
their original shared-card text. This is dataset preparation: no model has been
fitted and no additional data has been sent to Google.

Ten current older queue/decision pairs have differing review IDs. Their labels
match the frozen benchmark, so those examples retain the original v1 feature
text; the mismatches are recorded in the manifest rather than silently treating
the current queues as the historical cards. New records with mismatching review
IDs, unmatched card IDs or changed historical labels cause the builder to stop.
Every primary source and mapping file is hashed.

**The hard cases remain scarce.** Using the stored proposal mappings, 929 cards
contain a concrete correction: 907 approve and only 22 other rulings (16 discuss,
two reject, four chatter). The original concrete-proposal subset had 16 such
exceptions; the additional data adds six. Most of the 158 total non-approvals are
open questions, which are easier to keep out of a bulk-correction list anyway.
The held-out block contains only four non-approved concrete proposals out of
255, so validation must report their individual routing and not rely on a high
overall yes/no accuracy. Proposal counts are mapping-derived audit metadata,
not model features.

```bash
python3 expand_card_corpus.py --output /tmp/card-filter-expanded-v2 \
  --report /tmp/card-filter-expanded-v2-inventory.json
```

[Aggregate inventory and provenance audit](card-filter-expanded-v1-inventory.json).

### Input-repair rerun: the cue-627 location correction

The original E1 wide-source request represented the complete replacement
`Entemok's moon` → `Entémoch's Boon` as the shorter token `Entemok` →
`Entémoch's Boon`, omitted the card's cue-only/no-glossary scope, and truncated
the second-transcript lookup at 900 characters. `card_input_ablation.py` tests
those three repairs in all eight combinations, plus a version containing the
literal before/after sentences. Three unchanged comparison cards are controls.

All 12 inputs run three times, with order shuffled within each repetition:
**36 successful requests**, identical scores across repeats. The original and
all three unchanged comparison cards reproduce their previous scores exactly.
The model and Q_E1_RAW approval question are unchanged; no historical verdict
or GM note is included in a request.

| Target-card input | Estimated P(approve) |
|---|---:|
| Original request | 78.08% |
| Full replacement phrase only | 87.55% |
| Cue-only scope only | 78.12% |
| Complete sibling excerpt only | 83.75% |
| Full phrase + complete sibling | 90.81% |
| Full phrase + cue-only scope | 86.00% |
| Cue-only scope + complete sibling | 77.21% |
| All three repairs | 89.46% |
| All three + exact before/after sentences | 86.78% |

The recovered sibling text comes from the same campaign Git snapshot as the
frozen benchmark. Its match still starts at line 2201 with alignment score 0.903.
The full 1,225-character lookup output extends the exact original 900-character
prefix, now including “the path to Anthemok's boon was lost.” The edit's full
span, cue number and literal before/after sentences come from the review map at
that same commit, and the builder checks that the sentence changes contain only
the proposed replacement.

The replacement-span defect and truncated evidence suppressed this card's
score. The fully explicit descriptions still stay below the prototype's 90%
cutoff, while the shorter phrase-plus-sibling repair crosses it. That is input
sensitivity around a routing threshold, not evidence to select whichever prompt
gets the highest score. This one-card experiment does not establish whether
fixing inputs improves routing of the rare non-approved proposals; that needs
evaluation on additional cards. Existing benchmark results and replay groupings
are retained unchanged.

Private prompts, snapshot evidence and response rows are saved under
`~/data/decision-eval/card-input-ablation-v1/`;
[aggregate repeated scores and input hashes](gemini-card-input-ablation-v1.json)
are tracked here. Three focused tests check factor isolation, exact preservation
of the original control, sibling-prefix preservation and before/after fidelity.

```bash
python3 card_input_ablation.py --workdir /tmp/card-input-ablation-v2 \
  --report /tmp/card-input-ablation-v2.json
```

### Three-section review prototype

The revised prototype uses **over 90%**, **80–90% inclusive**, and regular cards
below 80%. Each of the two score groups has its own explicit bulk-approval
button. Neither button touches the other group or any regular card. Pulling a
row out of either list revokes its approval and moves it into individual review.
Open questions without a concrete correction always remain regular cards,
regardless of score. Every page starts unmarked and retains per-item exports.

The v2 replay uses the corrected `faithful_edit` score (86.78%) and matching
input from the cue-627 rerun for that one example; all other scores are unchanged.
The displayed replacement phrase comes from the original card title, preserving
the full `Entemok's moon` span instead of the old shortened token label.

| Group | Example session | Full historical set |
|---|---:|---:|
| Over 90%, independent bulk approval | 13 | 210 |
| 80–90%, independent bulk approval | 4 | 173 |
| Regular cards, individual decisions | 7 | 383 |

Historical non-approvals in the two lists are two and four respectively; these
are review-layout groups, not validated automatic-approval guarantees. The
earlier two-section replay and recorded benchmark scores remain archived.
No additional inference calls were needed for this UI change.

The example is `~/data/decision-eval/card-filter-v2/replay.html`; the complete
set is `full-replay.html` in the same directory.
[Counts and score provenance](gemini-card-filter-three-tier-v2.json).

```bash
python3 card_filter_review.py \
  --session 2026-01-12-chapter_43-harvesting-horror-in-the-underdark \
  --ablation-dir ~/data/decision-eval/card-input-ablation-v1 \
  --output /tmp/three-section-review.html
```

Boundary tests cover exactly 80% and 90%, invalid scores and open questions.
The browser test verifies independent bulk actions, revoking an approval from
either group, preserving individual decisions and notes, export validation,
unmarked reload and mobile layout.


### Stock Clef versus Gemini, three-group card filter

The retained Clef server was started on spark2 for a fresh, paired replay of
all 766 E1 wide-source cards. Both models receive the same faithful cue-627
input in place of that example's old malformed input. This baseline uses
**stock Clef 27B**, with no locally trained head.

| Model | Over 90% / non-approvals | 80–90% / non-approvals | Regular cards |
|---|---:|---:|---:|
| Clef 27B | 209 / 2 | 130 / 2 | 427 |
| Gemini Decision 1 | 210 / 2 | 173 / 4 | 383 |

Clef offers 339 bulk-list proposals versus Gemini's 383 and keeps 12 of the
16 historical non-approved concrete proposals on regular cards versus 10.
The rare exceptions are too few to establish a reliable winner. Despite similar
group sizes, 232 cards change tiers. Fixed 80/90 cutoffs need not be equally
calibrated for different models.

Clef scores the fully specified cue-627 example at 93.34%, versus Gemini's
86.78%. All three repeats agree across the 12 input-ablation variants and
controls. All 765 unchanged full-set inputs reproduce Clef's earlier scores
exactly. No cards failed. Current median Clef round trip is 731 ms (local LAN;
Gemini's earlier measurements include its hosted internet round trip).

Private inputs, responses, ablation and replay are archived at
`~/data/decision-eval/clef-card-filter-v1/`; the example page has 12 high-list,
three middle-list and nine regular cards. Browser and shared-export checks
passed. [Aggregate comparison](clef-card-filter-v1-comparison.json).

```bash
python3 clef_card_filter.py --workdir /tmp/clef-card-filter-v2 \
  --report /tmp/clef-card-filter-v2.json
```


### First personalized Clef readout training run

The requested supervised experiment trained **1,024 final readout weights**
inside Clef's decision head; the 27B backbone and all other head tensors stayed
frozen. This is a small readout fine-tune, not full-head or backbone training.
The checkpoint audit found only `residual_scorer.3.weight` changed; BF16 rounding
left 1,002 values different from the released tensor.

Training used 812 development cards (60 sessions), including 674 concrete
proposals and 18 concrete non-approvals. Backbone/routing features were cached
once. A regularized logistic correction to the released readout was fitted
using CPU LBFGS. Open/no-concrete-proposal examples have weight 0.25; concrete
proposals have weight 1. Five L2 values (0.001, 0.01, 0.1, 1, 10) were compared
in three session-grouped development folds. Pooled concrete-proposal AUC chose
0.001, with log loss breaking ties. Selection and final fitting were saved
before opening holdout labels.

Both heads were then evaluated with **actual full-model forward passes** on the
reserved 306 cards from five newer sessions (255 concrete proposals, only four
concrete non-approvals). These use the stored original card text and the general
approve-as-written question for both models, rather than the earlier E1 raw
source prompt; their numbers are not directly comparable to the 766-card table.

| Held-out group | Stock cards / exceptions | Tuned cards / exceptions |
|---|---:|---:|
| Over 90% | 149 / 2 | 127 / 3 |
| 80–90% | 33 / 1 | 73 / 1 |
| Regular cards | 124 / 35 | 106 / 34 |

Regular-card exception counts include open questions that always remain on
cards. Of the **four non-approved concrete proposals**, stock keeps one on
regular cards; tuned keeps none. The trained head offers 18 more bulk-list
items overall but misroutes all four hard exceptions, including three in its
highest group. It is **not an upgrade for this filter at the chosen cutoffs**.
The sample has too few exceptions to establish a general model-quality claim.

On development folds, AUC improved 0.573 → 0.628. That did not transfer:
held-out AUC fell 0.528 → 0.306. Held-out concrete log loss improved
0.324 → 0.198 because predictions of the 251 ordinary approvals improved
(loss 0.295 → 0.153), while loss on the four exceptions worsened 2.194 → 3.009.
This demonstrates why overall probability loss cannot select this workflow's
upgrade by itself. Four-decimal API rounding leaves all group counts unchanged.

The complete private experiment, including frozen inputs, feature cache, code,
checkpoint, model selection, weight audit and per-card predictions, is archived
at `~/data/decision-eval/clef-card-training-v1/` and retained on spark2 at
`~/clef-card-training-v1/`. [Audited metrics](clef-readout-training-v1.json).
The training implementation is `clef/finetune/clef_train_and_serve.py`; its
regularized optimizer and metrics are `clef/finetune/clef_readout.py`.

The trial container serves both heads on spark2:8002: existing model ID `clef`
remains stock; `clef-card-tuned` explicitly selects the experimental head and
supports one noul approval question. Health, model discovery and stock/tuned
serving parity checks passed. No production client or review rule was switched
to the tuned head. See `current-setup.md` for deployment and restoration.
