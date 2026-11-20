# Adaptive chunking on Temple of Elemental Evil

The upstream recursive splitter improved evidence retrieval on this local
Temple of Elemental Evil conversion compared with fixed 600-token windows.
Adaptive selection also improved on that baseline, but did not outperform the
best static setting. **Try the simple upstream recursive splitter first; this
pilot does not justify adding the full adaptive selector to production.**

This was an isolated experiment. Campaign files, production indexes, Cognee
datasets and serving configuration were preserved. Source text, questions,
retrieved passages, embeddings and model answers remain local in ignored
experiment storage.

## Retrieval result

Forty source-grounded questions were frozen before scoring. A complete result
requires all designated source evidence for that question, including every
span of a multi-span question, within the retrieved context budget.

| Chunking arm | Chunks | Complete at 1,200 tokens | Complete at 2,400 tokens | Complete in top 5 chunks |
|---|---:|---:|---:|---:|
| Fixed 600-token windows | 257 | 22/40 — 55% | 34/40 — 85% | 36/40 |
| Upstream recursive, 600 | 904 | **37/40 — 92.5%** | **40/40 — 100%** | 37/40 |
| Upstream recursive, 1,100 | 886 | **37/40 — 92.5%** | **40/40 — 100%** | 37/40 |
| Heading-aware upstream splitter | 621 | **37/40 — 92.5%** | 39/40 — 97.5% | **38/40** |
| Adaptive selection, four metrics | 620 | **37/40 — 92.5%** | 39/40 — 97.5% | **38/40** |

At 1,200 tokens, adaptive retrieval gained 15 complete questions over fixed
windows and lost none. At 2,400 tokens it gained five and lost none. The
heading-aware variant had the same complete-question outcomes as adaptive.
The static recursive settings recovered the remaining trap question at 2,400
tokens, although the intrinsic selector preferred another setting for that
section. Higher intrinsic scores therefore did not select the highest-recall
configuration in this pilot.

All arms used the same Qwen embedding model, dense cosine ranking, query
instruction, source-breadcrumb rule and context-budget accounting. Whole
chunks were packed from the top 20 candidates. Overlap, breadcrumbs and
separators consumed budget. The top-five column is a separate diagnostic:
those five chunks have different total token lengths across arms.

The comparison changes granularity as well as boundaries. Recursive small-only
merging produced many more, shorter chunks than fixed windows. At the
1,200-token cap, mean actual retrieved text was **968 tokens for fixed windows
and 1,136 for adaptive**. At 2,400, it was 2,165 and 2,322 respectively. Better
packing contributes to the result; this is not an isolated measurement of an
adaptive-selection algorithm.

## Answer spot check

Nine question IDs and fact checks were fixed before retrieval. The same local
vLLM model answered both arms' 2,400-token contexts, with temperature zero,
thinking disabled and instructions to abstain when evidence was missing.
Arms alternated for each question. All 18 calls finished naturally.

| Manual source review | Fixed windows | Adaptive retrieval |
|---|---:|---:|
| Complete requested information | 7/9 | **9/9** |
| Partial answer | 1/9 | 0/9 |
| Answered unknown | 1/9 | 0/9 |
| Nonempty supporting quotations all verbatim | 7/9 | 8/9 |

On the trading-post question, fixed retrieval found the operators' names but
missed their physical descriptions; the reader explicitly acknowledged the
missing information. Adaptive retrieval supplied both. On a dungeon room's
dimensions, the fixed reader answered unknown while the adaptive reader
recovered both values from its retrieved passage.

These were retrieval failures followed by partial answers or abstention, not
observed invented factual answers. No unsupported factual answer claims were
found in this small manual review. One answer in each arm used an ellipsis in
a purported quotation, failing strict verbatim support even when the facts
were correct. A name/number pattern check gave fixed windows 8/9, illustrating
why that weaker automated check cannot replace checking every requested part.
This is a bounded spot check, not exhaustive answer-quality evaluation.

## What was tested

- [Ekimetrics adaptive-chunking](https://github.com/ekimetrics/adaptive-chunking),
  clean commit `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d`.
- The user's local `adventure-t14-5e.json` conversion: **31 sections, 2,674
  rendered structural blocks, 133,137 o200k tokens**. Tables and lists were
  preserved during a single shared rendering pass.
- Forty manually selected questions and 42 source spans: NPC identities,
  relationships, conditions, room mechanisms, item rules, numbers and node
  hazards. The selection metrics never used these questions or answers.
- Local Ollama `qwen3-embedding:0.6b` on spark2; model digest and input hashes
  are recorded in the [aggregate receipt](experiments/adaptive-chunking/results.json).
  Input truncation was disabled and requests were serialized in batches.
- The paired reader was the existing `qwen3.8-flash-next` vLLM service on
  spark1. Hibana was not used as the reader.

Adaptive selection chose the heading-aware splitter for 23 sections,
recursive-600 for seven and recursive-1100 for one. It used four upstream
metrics with equal weights: size compliance, block integrity, intrachunk
cohesion and contextual coherence. Reference completeness from coreference
resolution was omitted. Coherence sampled at most 16 starting windows per
section. The core splitters, metric functions and selection function came
from the pinned upstream implementation.

This is a **four-metric adaptation with selection per module section**, not a
reproduction of the full paper's per-document experiment. It omits the
coreference model, LLM-regex and semantic splitter candidates. JSON conversion
was held constant, so the trial does not measure PDF parsing quality or verify
edition conversions against the printed original.

## Additional budget sensitivity

After the primary result, the same cached vectors and unchanged questions were
used to examine packing effects at additional budgets. No new model calls or
chunking choices were made. These are explicitly post-hoc diagnostics.

| Budget | Fixed windows | Recursive 600 | Adaptive |
|---|---:|---:|---:|
| 1,500 | 33/40 | 38/40 | 38/40 |
| 1,800 | 34/40 | 39/40 | 38/40 |
| 3,000 | 35/40 | 40/40 | 39/40 |
| 4,800 | 38/40 | 40/40 | 40/40 |

The benefit persists beyond the exact primary caps, but shrinks when the reader
can take more context. This supports more efficient retrieval under a constrained
context budget, rather than a universal claim about recall at every setting.

## Practical next step and limits

Use recursive-600 with the same breadcrumbs as the first candidate for an
isolated Cognee module index, keeping the current production index for comparison.
Measure full ingestion cost and graph-backed recall before promotion: this
candidate makes **904 chunks versus 257**, which can substantially increase
graph-extraction calls even when vector retrieval improves. The present trial
measured direct chunk retrieval, not Cognee graph ingestion or recall.

One module conversion and 40 targeted questions do not establish broad D&D
module performance. Evidence scoring requires designated source spans; another
passage may independently support an answer. No tuning on these question
outcomes was performed. Before choosing a production policy, repeat on a second
module with new questions and compare against the actual production chunker.

The five-arm retrieval experiment completed in **271.3 seconds**, making 567
embedding requests for 7,558 unique text inputs. The 18 serialized reader calls
totaled **62.7 seconds**. These are costs of this experiment, including multiple
candidates and intrinsic scoring, not a production ingestion benchmark.

Validation: **61 Python tests passed**, including source identity, overlapping
evidence coverage, context accounting, Unicode preservation and redacted exports.
All source spans matched the frozen rendering, and every chunking arm passed
complete source-coverage checks.

Artifacts: [runbook and scripts](experiments/adaptive-chunking/README.md),
[aggregate results](experiments/adaptive-chunking/results.json),
[manual answer review](experiments/adaptive-chunking/quality-review.json).
