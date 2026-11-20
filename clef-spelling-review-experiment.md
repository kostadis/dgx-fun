# Clef and Gemini for spelling correction review

Experiments completed November 12, 2026.

**Stock Clef was competitive with Gemini Decision 1 at predicting which spelling corrections the GM would approve.** It is a sensible default for this review filter on the existing Spark hardware. The first personalized fine-tune did not improve exception handling, so the released Clef head remains the default.

“Competitive” applies to this historical approval-filter experiment. It does not establish equal performance across other tasks or prove better spelling accuracy. Local inference avoids a per-request hosted API charge, while still consuming electricity and Spark capacity.

# The problem and proposed review flow

The existing spelling workflow presents a separate card for each proposed correction. Most receive “yes,” so the opportunity is to reduce repetitive review while keeping ambiguous changes visible.

The decision model answers: **Will the GM approve this correction exactly as written, without changes or discussion?** It scores an existing proposal; it does not generate the replacement spelling.

The prototype routes concrete corrections into three groups:

- **Above 90%:** a flat list with one bulk approval.

- **80% through 90%:** a separate flat list with its own bulk approval.

- **Below 80%, plus open questions:** individual review cards.

Every list row can be moved to individual review, including after bulk approval. Nothing starts approved. These are model score thresholds, not verified guarantees of real-world approval rates. The historical prototype changed no campaign files.

# Stock Clef compared with Gemini

Both models scored the same 766 historical card states, using the same question and the corrected faithful input for the Entémoch’s Boon example. All requests completed successfully.

An “exception” below means a historical outcome other than approval exactly as proposed. It can include discussion or chatter, not just explicit rejection.

| Review group | Stock Clef cards | Clef exceptions | Gemini cards | Gemini exceptions |
| --- | ---: | ---: | ---: | ---: |
| Above 90% | 209 | 2 | 210 | 2 |
| 80% through 90% | 130 | 2 | 173 | 4 |
| Individual cards | 427 | 100 | 383 | 98 |

The high-confidence groups were almost identical in size and exception count. Across both bulk lists, Clef grouped **339 proposals with four exceptions**; Gemini grouped **383 with six exceptions**. Gemini offered more bulk-review coverage; Clef left more work on individual cards and included fewer exceptions in the bulk lists.

These results support using stock Clef for this workflow. They do not identify a universal winner: the two systems changed the review tier for 232 cards, and the practical tradeoff depends on how much the GM values fewer cards versus catching exceptions. Clef’s measured median request time in this run was about 731 milliseconds.

# Why the exact input matters

The correction **“Entemok’s moon” → “Entémoch’s Boon”** exposed an input problem. The earlier request represented only the token “Entemok,” omitted the cue-only scope, and truncated supporting context. That did not faithfully describe the proposed edit.

With the complete before-and-after phrase, intended scope, and restored supporting context, stock Clef scored approval at **93.34%** and Gemini at **86.78%**. Under the chosen rules, Clef placed it in the above-90% list and Gemini in the 80–90% list. The historical GM decision was approval.

This example shows why a decision model should receive the same concrete edit and scope the reviewer sees. A score can change because the request describes a different decision, even when the underlying transcript is unchanged.

# The personalized training experiment

The expanded corpus contained **1,118 cards from 65 sessions**, including material from Out of the Abyss. We used 812 cards from 60 sessions for development and reserved 306 cards from five newer sessions for testing. Entire sessions stayed together, and held-out labels were excluded from fitting and hyperparameter selection.

This was a small readout fine-tune: only the final 1,024 residual-scorer weights were trainable. The 27-billion-parameter backbone and the rest of the decision head stayed frozen. Three-fold validation grouped by session selected the regularization setting before the held-out evaluation.

| Held-out review group | Stock Clef cards | Stock exceptions | Tuned Clef cards | Tuned exceptions |
| --- | ---: | ---: | ---: | ---: |
| Above 90% | 149 | 2 | 127 | 3 |
| 80% through 90% | 33 | 1 | 73 | 1 |
| Individual cards | 124 | 35 | 106 | 34 |

The held-out set contained 255 concrete proposals, only four of which were not approved. Stock Clef put three of those four into the bulk lists; tuned Clef put all four there. The tuned head increased combined bulk coverage from **182 to 200 cards**, but increased included exceptions from **three to four**.

Average prediction loss improved from 0.324 to 0.198 because the tuned model better predicted the many approved proposals. Its predictions on the rare non-approved proposals worsened. For a filter whose purpose is to preserve review of exceptions, this was not an improvement.

Only four difficult concrete cases were available in the test, so the comparison is limited. The result applies to this readout training method and dataset; it does not establish that larger head or backbone training cannot help. The training trial also used a different card-text representation from the earlier 766-card comparison, so its stock and tuned results should be compared within this experiment.

# Decision and practical implications

**Keep stock Clef as the default approval scorer.** Retain the three review groups and require explicit bulk approval. Collect more concrete proposals that the GM rejected, modified, or discussed before attempting a larger personalized fine-tune; those rare decisions are the examples the filter most needs to learn.

At the end of the experiment, Spark 2 served the released head as `clef` and the experimental head separately as `clef-card-tuned`, sharing one backbone on port 8002. The default model ID continued to select stock Clef. This is the recorded experiment deployment, rather than a fresh service-health check.

The economic benefit is avoiding per-request hosted API charges on hardware already owned. Local inference still has electricity and capacity costs. The approval-filter experiment measures review routing, not whether proposed replacement spellings are correct; that needs a separate spelling-quality evaluation.

# Evidence and reproducibility

The aggregate reports are saved in the local repository:

- Stock Clef versus Gemini: `clef/eval/clef-card-filter-v1-comparison.json`

- Training method and held-out results: `clef/eval/clef-readout-training-v1.json`

- Experiment details: `clef/eval/README.md`

Private inputs, predictions, checkpoints, and review pages remain under `/home/kostadis/data/decision-eval/`, including `clef-card-filter-v1/` and `clef-card-training-v1/`. The saved checkpoint audit confirmed that only the intended final weight tensor changed, and serving checks matched the saved predictions.
