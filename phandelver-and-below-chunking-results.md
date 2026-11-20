# Chunking Phandelver and Below: The Shattered Obelisk

The user's preferred module confirms that the upstream recursive splitter is
worth testing for context-limited retrieval. On 48 frozen questions,
recursive-600 recovered complete designated evidence for **42 questions within
1,200 tokens and 46 within 2,400**, compared with fixed windows' **28 and 40**.
The four-metric adaptive selector recovered **40 and 45**, matching a static
heading-aware setting and trailing plain recursive splitting.

The benefit is clearest as **context-budget efficiency**. At a fixed five
retrieved chunks, recursive and fixed windows tied. The answer spot checks
did not establish an overall answer-quality winner. This is evidence for a
candidate chunking configuration, not qualification of a production Cognee change.

## Primary retrieval results

| Method | Indexed chunks | Complete evidence at 1,200 tokens | At 2,400 tokens | In top five chunks |
|---|---:|---:|---:|---:|
| Fixed 600-token windows | 296 | 28/48 — 58.3% | 40/48 — 83.3% | 42/48 |
| Upstream recursive, 600 | 1,161 | **42/48 — 87.5%** | **46/48 — 95.8%** | **42/48** |
| Upstream recursive, 1,100 | 1,156 | **42/48 — 87.5%** | **46/48 — 95.8%** | **42/48** |
| Heading-aware upstream splitter | 761 | 40/48 — 83.3% | 45/48 — 93.8% | 41/48 |
| Adaptive selection, four metrics | 759 | 40/48 — 83.3% | 45/48 — 93.8% | 41/48 |

All budgets use the same o200k tokenizer. A complete question requires the
union of retrieved source intervals to cover all its designated evidence
spans. Overlap cannot count twice, and another chapter's matching offsets
cannot satisfy a span. Forty-eight questions use 51 spans.

At 1,200 tokens, recursive-600 gained 14 complete questions over fixed windows
and lost none. At 2,400, it gained seven and lost one, for a net gain of six.
The result is not uniform across questions. Adaptive gained 13 and lost one
at the smaller budget, then gained seven and lost two at the larger budget.

The adaptive selector chose heading-aware splitting for 11 sections, recursive-600
for the story tracker and recursive-1100 for credits. All question-bearing
sections therefore used the heading-aware candidate. Its higher intrinsic
scores did not identify the best downstream retrieval configuration.

## Why the budget matters

All arms use the same local embeddings, query instruction, cosine ranking and
breadcrumb attachment rule. Whole chunks are packed from the top 20 candidates,
including breadcrumb, overlap and separator costs. Smaller chunks let more
candidate passages fit before the context cap.

| Primary cap | Mean actual tokens, fixed | Recursive 600 | Adaptive |
|---|---:|---:|---:|
| 1,200 | 811 | 1,119 | 1,119 |
| 2,400 | 2,076 | 2,329 | 2,315 |

These are matched caps, not identical consumed token counts. The recursive
configuration also indexes nearly four times as many chunks as fixed windows.
Granularity, packing and the precision of attached headings change together;
the experiment does not isolate an adaptive algorithm's contribution.

The top-five diagnostic has no common token cap: five large fixed chunks carry
more text than five small recursive chunks. Nevertheless, its tie at 42/48
shows that this trial does not demonstrate better recall at the same top-k.

After the primary run, additional caps were checked using the exact cached
vectors and unchanged questions, with no new model calls or chunking choices:

| Additional cap, post-hoc | Fixed | Recursive 600 | Adaptive |
|---|---:|---:|---:|
| 1,500 | 35/48 | 43/48 | 42/48 |
| 1,800 | 35/48 | 45/48 | 44/48 |
| 3,000 | 42/48 | 46/48 | 46/48 |
| 4,800 | 43/48 | 47/48 | 47/48 |

The budget advantage persists beyond the two primary caps and becomes smaller
with more available context. These additional results are exploratory.

## What was ingested

The source is the local `5e-tools-kostadis/data/adventure/adventure-pabtso.json`:
13 sections, **4,749 rendered blocks and 158,712 o200k tokens**. The representation
includes the adventure narrative, boxed text, tables, named list entries,
gallery captions and resolved appendix content. Map pixels are not interpreted.

The adventure's 82 direct creature, creature-description, item and reward
references were expanded with 5etools' own local copy/merge implementation.
Seven child mind-crystal variants were also resolved, for **89 resolved entities**.
The export records SHA256 hashes for 20 resource files and blocks HTTP loading.
All exported entities were rendered; source files were preserved.

Questions were frozen before selection or retrieval: 16 cover chapters 1–4,
24 cover chapters 5–8, and eight cover magic items and creatures. The test
therefore includes substantial material beyond the opening Lost Mine story.

The pinned chunking implementation is
[Ekimetrics adaptive-chunking](https://github.com/ekimetrics/adaptive-chunking)
at `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d`. It uses the same local adaptation
as the [ToEE pilot](adaptive-chunking-toee-results.md): three splitter candidates,
equal weights for size compliance, block integrity, intrachunk cohesion and
contextual coherence, and at most 16 coherence starting windows per section.
Coreference scoring, LLM-regex and semantic splitter candidates are omitted.
This is not a full paper reproduction or a PDF parser comparison.

Embeddings came from the existing local Ollama `qwen3-embedding:0.6b` service,
with input truncation disabled and serialized batches. The model digest, input
hashes, code hash and primary parameters are in the
[aggregate receipt](experiments/adaptive-chunking/pabtso-results.json).

## Answer checks and grounding

Twelve question IDs, their fact checks, the three arms and the 2,400-token
budget were recorded before selection began. The unchanged spark1 vLLM reader,
`qwen3.8-flash-next`, answered 36 requests, rotating arm order per question.
Temperature was zero and thinking disabled. All calls finished naturally.
Codex inspected the answers against the retrieved text and full source;
this was not independent human adjudication.

The check did **not establish a generated-answer winner**:

- Fixed windows omitted the requested manifestation on one question and
  explicitly acknowledged the missing information.
- Recursive and adaptive answers omitted a captive's room identifier from
  their prose despite having it in context. Recursive's supporting quotation
  supplied the identifier, making its full response more useful than its summary.
- One recursive answer put the final refraction confrontation in the Endless
  Void. The encounter section places it at the Briny Pool in B22 after the
  party returns. The introductory passage and question compress this geography,
  so this is a diagnostic grounding finding, not a reliable comparative error rate.
- Several quotations were abridged with ellipses despite the verbatim-quote
  instruction. All quotes were nonempty and verbatim in 10/12 fixed responses,
  9/12 recursive responses and 10/12 adaptive responses.

The predefined string checker returned 9/12 in every arm, but that is not an
answer-accuracy score. It unnecessarily required the town name when the named
destination business was supplied, and extra words when the one-word answer
already identified the requested creature. The
[source review](experiments/adaptive-chunking/pabtso-quality-review.json)
preserves these distinctions and receipt hashes.

Designated-span scoring is also stricter than answerability. For example, the
fixed-window answer to Halia's offer was complete even though its selected text
did not cover both entire designated source spans. Alternative or partial
passages can supply all requested facts. The primary scores were retained
unchanged rather than relabeling cases after seeing results.

## Practical interpretation

Recursive-600 is the strongest tested retrieval candidate under these caps,
consistent with the earlier ToEE result. Adaptive selection added no observed
benefit over a static candidate. Try the simpler setting in an isolated index
and match the actual application's retrieval budget before choosing a policy.

For Cognee, full ingestion and graph-backed recall remain untested. Moving
from 296 to 1,161 chunks could multiply graph-extraction work; a vector-retrieval
gain does not establish a graph-quality or end-to-end latency gain. Existing
Cognee datasets, indexes and serving configuration were preserved.

These are 48 targeted questions on one JSON representation, not a random
sample of D&D queries. The answer review is small and includes an ambiguous
geography question. The two module pilots support further testing rather than
a broad accuracy claim or production promotion.

The five-arm selection/retrieval run took **433.1 seconds**, with 710 embedding
requests covering 10,796 unique text inputs. The 36 serialized reader calls
totaled **127.4 seconds**. These are observed experiment costs, not production
ingestion or sustained throughput measurements.

Validation: **63 Python tests passed**. All gold spans matched the frozen
rendering, all chunking arms passed complete source-coverage checks, the source
hashes remained unchanged, and the native JavaScript exporter passed syntax checks.
The earlier ToEE receipts and original script snapshots were preserved separately.

Artifacts: [runbook and scripts](experiments/adaptive-chunking/README.md),
[aggregate results](experiments/adaptive-chunking/pabtso-results.json),
[source review](experiments/adaptive-chunking/pabtso-quality-review.json).
