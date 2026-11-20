# Adaptive chunking on D&D modules

These isolated pilots test [Ekimetrics adaptive-chunking](https://github.com/ekimetrics/adaptive-chunking)
against a fixed-window baseline on the user's local adventure sources.
The question is whether its intrinsic scoring selects chunks that retrieve
more of the evidence needed to answer actual module questions.

**Completed pilot:** adaptive retrieved complete designated evidence for 37/40
questions at 1,200 tokens and 39/40 at 2,400, versus fixed windows' 22/40 and
34/40. Static upstream recursive splitting reached 37/40 and 40/40, so adaptive
selection did not improve on the best static setting. Nine paired answers were
manually reviewed. [Results and limitations](../../adaptive-chunking-toee-results.md),
[aggregate receipt](results.json), [answer review](quality-review.json).

## First pilot: Temple of Elemental Evil

### Frozen inputs and comparison

- Upstream commit: `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d`.
- Source: local `campaigns/toee/docs/background/adventure-t14-5e.json`;
  **31 sections, 2,674 rendered blocks, 133,137 o200k tokens**.
- Forty manually written questions, with 42 exact source spans, frozen before
  running selection or retrieval. Categories include identities, relationships,
  conditional behavior, room mechanisms, item rules, numbers and node hazards.
- Embeddings: existing local Ollama `qwen3-embedding:0.6b` on spark2, one batch
  at a time. Record its model digest. Disable silent embedding-input truncation.
- Dense cosine retrieval with the same query instruction and the same source
  breadcrumb rule for every arm. Pack whole chunks from the top 20 candidates
  into **1,200-token and 2,400-token context budgets**, counting breadcrumbs,
  overlaps and separators. Also record complete-evidence retrieval in the top five.

| Arm | Construction |
|---|---|
| `fixed_600` | 600-token source windows, 50-token overlap |
| `recursive_600` | Upstream recursive splitter, 600-token bound |
| `recursive_1100` | Upstream recursive splitter, 1,100-token bound |
| `module_headings_600` | Upstream splitter with Markdown-heading separators first |
| `adaptive_4metric` | Select one of the three upstream-based candidates independently per source section |

The three candidates use `small_only` merging, 100-token minimum, 50-token
overlap and `max_tokens_strategy="chunk_size"`. The adaptive selector uses
equal weights for upstream size compliance, block integrity, intrachunk
cohesion and contextual coherence. Missing metrics are ignored as in the
upstream selector. Coherence evaluates at most 16 sliding-window starting
positions per section to bound the local embedding workload.

This is a **four-metric local adaptation**, not a reproduction of the full
paper. Coreference-based reference completeness, LLM-regex and semantic
splitters are outside this pilot. Existing JSON parsing is held constant;
this does not compare PDF parsers or verify the conversion against the
original printed module. Section-level selection is finer-grained than
choosing one method for an entire book. The fixed-window baseline is a
controlled reference, not a claim to reproduce a production Cognee index.

## Scoring

Primary outcome: all designated evidence spans for a question are completely
covered by the union of retrieved source spans, under the context budget.
Also report mean evidence coverage and actual retrieved tokens. Overlap cannot
count twice; matching offsets in another section cannot satisfy a source span.
Multi-span questions require every span. Designated-span completeness is stricter
than answerability: another passage may independently support the same answer.

Nine question IDs and their expected fact checks are fixed before retrieval
for a paired answer spot check. Both arms use the same unchanged vLLM reader,
`qwen3.8-flash-next` on spark1, at the 2,400-token budget. Alternate arms per
question, disable thinking, use temperature zero and require exact supporting
quotations. Fact-pattern checks and verbatim quotation checks are diagnostic;
they do not establish correctness of every generated claim. Review the private
answers against their source before interpreting them as quality results.

No campaign files, existing indexes, Cognee datasets or serving configurations
are modified. Module text, questions, answers, source excerpts and vectors stay
under ignored `raw/`; `.upstream/` is also ignored.

## Run locally

The runner needs Python 3.11+ with NumPy, pandas and tiktoken; see
[`requirements.txt`](requirements.txt). It loads the upstream
splitters and metrics directly. The selector's unchanged function body is loaded
from its AST to avoid importing unused plotting dependencies. A clean upstream
checkout at the pin is required.

```bash
git clone https://github.com/ekimetrics/adaptive-chunking.git \
  experiments/adaptive-chunking/.upstream
git -C experiments/adaptive-chunking/.upstream checkout \
  ea87ce8e1a97888f3f179e7f1359ff7f43fb179d

python3 experiments/adaptive-chunking/prepare.py \
  --source /home/kostadis/src/campaigns/toee/docs/background/adventure-t14-5e.json \
  --output experiments/adaptive-chunking/raw/corpus.json
```

Create a private `raw/cases.json`: each record has an ID, category, question,
and `evidence` records containing `doc_id`, `start`, `end` and exact `quote`.
`answer_checks` contains groups of alternative fact strings for the spot check.
The local first trial retains this frozen fixture and its builder in `raw/`.

```bash
export TIKTOKEN_CACHE_DIR="$PWD/experiments/adaptive-chunking/raw/tiktoken-cache"
python3 experiments/adaptive-chunking/run.py \
  --corpus experiments/adaptive-chunking/raw/corpus.json \
  --cases experiments/adaptive-chunking/raw/cases.json \
  --output experiments/adaptive-chunking/raw/pilot-20261120
python3 experiments/adaptive-chunking/check_answers.py \
  --run experiments/adaptive-chunking/raw/pilot-20261120 \
  --cases experiments/adaptive-chunking/raw/cases.json
```

Embedding and section-metric caches are resumable. The runner rejects changed
inputs, model digest or code when resuming a result directory. Use a new run
directory for a changed configuration. Query evidence is validated against the
frozen source before work starts. Individual reader responses are also resumable
only when their request hash and endpoint match.

The optional `sensitivity.py --run ... --cases ...` command reuses cached
vectors for post-hoc budgets of 1,500, 1,800, 3,000 and 4,800 tokens, without
new model calls. `analyze.py --run ... --output results.json` exports aggregate
receipts, paired case IDs and sensitivity results without module text,
questions, passages, vectors or answer text.

## Second pilot: Phandelver and Below

**Completed:** recursive-600 retrieved complete designated evidence for 42/48
questions at 1,200 tokens and 46/48 at 2,400; fixed windows reached 28/48 and
40/48, while adaptive reached 40/48 and 45/48. Recursive and fixed tied at a
fixed five chunks. The 36 answer checks did not establish an overall answer-quality
winner. [Results and limits](../../phandelver-and-below-chunking-results.md),
[aggregate receipt](pabtso-results.json), [source review](pabtso-quality-review.json).

The second source is the user's preferred **Phandelver and Below: The Shattered
Obelisk**, from `5e-tools-kostadis/data/adventure/adventure-pabtso.json`.
It uses the same five arms, embedding model, metrics, retrieval candidate pool
and primary context budgets, with a new 48-question fixture. Sixteen questions
cover chapters 1–4, 24 cover chapters 5–8, and eight cover magic items and creatures.
Twelve question IDs are fixed for answer checks across fixed-window, recursive-600
and adaptive contexts, using the same vLLM reader at 2,400 tokens.

`export_references.mjs` runs 5etools' native local JSON copy/merge implementation,
with network loading disabled. It resolves all 82 direct statblock/fluff/item/
reward references plus seven child mind-crystal variants. The export records
hashes of 20 local resource files. The renderer preserves named list entries,
roll cells, gallery captions, source page metadata, resolved creature abilities,
and item rules. Map pixels are not interpreted.

```bash
node experiments/adaptive-chunking/export_references.mjs \
  /home/kostadis/src/5e-tools-kostadis \
  /home/kostadis/src/5e-tools-kostadis/data/adventure/adventure-pabtso.json \
  "$PWD/experiments/adaptive-chunking/raw/pabtso/references.json"
python3 experiments/adaptive-chunking/prepare.py \
  --source /home/kostadis/src/5e-tools-kostadis/data/adventure/adventure-pabtso.json \
  --module-title 'Phandelver and Below: The Shattered Obelisk' \
  --references experiments/adaptive-chunking/raw/pabtso/references.json \
  --output experiments/adaptive-chunking/raw/pabtso/corpus.json
```

The frozen module has 13 sections, 4,749 rendered blocks and 158,712 o200k
tokens. Its private fixture has 48 questions and 51 exact evidence spans.
The private `raw/pabtso/reader-plan.json` records the answer-check IDs, arms,
budget and question-file hash before selection starts.

```bash
python3 experiments/adaptive-chunking/run.py \
  --corpus experiments/adaptive-chunking/raw/pabtso/corpus.json \
  --cases experiments/adaptive-chunking/raw/pabtso/cases.json \
  --output experiments/adaptive-chunking/raw/pabtso/pilot-20261120
python3 experiments/adaptive-chunking/check_answers.py \
  --run experiments/adaptive-chunking/raw/pabtso/pilot-20261120 \
  --cases experiments/adaptive-chunking/raw/pabtso/cases.json \
  --case-ids q01 q05 q09 q13 q17 q21 q25 q29 q33 q37 q43 q47 \
  --methods fixed_600 recursive_600 adaptive_4metric
```

Use fresh output filenames/directories for changed inputs. The ToEE receipts,
private source, questions and answers remain separate. Its original runner,
renderer and reader scripts are retained under that run's ignored
`code-snapshot/`; the runner snapshot matches the recorded code hash.

## Mempalace chunker arms (2026-11-20)

Adds `mempalace_800`, `mempalace_800_crumb` and `recursive_600_nocrumb`. Seed each
new output dir with the earlier run's `embeddings.sqlite3` so old arms are cached.

```bash
python3 experiments/adaptive-chunking/run.py \
  --corpus experiments/adaptive-chunking/raw/corpus.json \
  --cases experiments/adaptive-chunking/raw/cases.json \
  --output experiments/adaptive-chunking/raw/mempalace-20261120
python3 experiments/adaptive-chunking/run.py \
  --corpus experiments/adaptive-chunking/raw/pabtso/corpus.json \
  --cases experiments/adaptive-chunking/raw/pabtso/cases.json \
  --output experiments/adaptive-chunking/raw/pabtso/mempalace-20261120
```

Write-up: [mempalace-chunker-arm-results.md](../../mempalace-chunker-arm-results.md).
