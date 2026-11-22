# Phandelver adaptive chunking in Cognee

For a combined account of the factual benchmark and all model/engine narrative
experiments, see the [linked write-up](phandelver-cognee-agent-results.md).

The later reader-only comparison found **3.41 s median and 44/47 complete
factual answers with Hibana fast**, versus **4.68 s and 45/47 with pinned Qwen
vLLM**, and **12.10 s with Inkling vLLM** (44/47 strict, 45/47 lenient).
Hibana fast also ran the general module question in one pass and through an
agent. Both were faster than the corresponding Qwen runs but retained grounding
failures. See the [Hibana fast follow-up](#hibana-fast-original-questions-and-module-synthesis-2026-11-21).

The subsequent [Unsloth Inkling DSpark/no-thinking Cognee test](experiments/inkling/dspark/COGNEE.md)
completed all 48 reader queries at **10.35 s p50 / 15.13 s p95**, with **44/47
strict** and 515.36 s total query time. Its 221.57 s module agent missed the
actual Chapter 4-to-5 handoff. The separate report records the effort-setting
bug, verified thinking-channel suppression, preserved stalled attempts and
quality limits; historical results below retain their original configurations.

A completed [paired native thinking test](experiments/inkling/native-thinking/RESULTS.md)
then held the checkpoint, process and reader input constant. Thinking off cut
factual p50 **10.38 → 4.51 s** and the 48-query sum **564.26 → 222.25 s**.
Both arms scored **44/47 complete**, but fully grounded complete answers fell
**44/47 → 43/47**. Agent time fell **579.03 → 164.69 s**, with fewer searches
and source errors in both answers. DSpark was restored afterward; these native
results are separate from the GGUF and AtlasKV experiments.

The heading-loss defect is repaired in the experimental Cognee adapter. The
full corrected follow-up, completed on **2026-11-21**, produced **45/47 complete
correct answers**, up from **43/47** for raw adaptive chunks. Stock Cognee
remains ahead at **46/47**. Corrected adaptive p95 query latency was **6.16 s**,
versus **14.04 s** for stock: about **2.28× faster** in these observed runs.

The adapter now passes each chunk's existing frozen heading breadcrumb into
graph extraction, summarization, and vector indexing. It retains the original
source text and offsets separately; boundaries and question-independent
selection remain unchanged. This restores the context used in the earlier
[vector retrieval trial](phandelver-and-below-chunking-results.md).

## Corrected follow-up

All 13 sections and 48 questions completed. The original stock and raw adaptive
indexes and results were preserved. The same q37 source ambiguity remains
excluded, and answer review uses the original rubric.

| Measurement | Stock Cognee | Adaptive, raw | Adaptive, headings preserved |
|---|---:|---:|---:|
| Complete correct answers, Codex review | 46/47 | 43/47 | **45/47** |
| Complete correct answers supported by supplied context | 46/47 | 43/47 | **45/47** |
| Clean abstentions | 0 | 3 | 2 |
| Incorrect responses under the question rubric | 1 | 1 | 0 |
| Query latency p50 | 9.87 s | 4.84 s | **4.68 s** |
| Query latency p95 | 14.04 s | 6.65 s | **6.16 s** |
| Median query prompt tokens | 17,084.5 | 3,635.5 | 3,928 |
| Persisted chunks | 78 | 759 | 759 |
| Ingestion, successful document attempts | 64.85 min | 230.28 min | 244.52 min* |
| Maximum concurrent model calls | 4 | 4 | 8 |

The corrected run recovered **q26** (secret burial names), **q29** (the three
fanatics and their beliefs), and **q46** (Careful Mind Crystal). It introduced a
new miss on **q22**: the reader did not receive Ruxithid's clairvoyance passage
and abstained. **q04** still lacked the flood timing passage and also abstained.
Both misses have zero frozen-gold coverage in raw context and source lineage.

The original item failure is repaired: Careful protects up to three chosen
creatures, whose saves automatically succeed. Reviewing all six affected
adjacent-item fragments found the correct effect owner in both graph extraction
and summaries. The six-question item smoke test also passed.

This is not exhaustive graph correctness. One Quickened graph description still
calls its rarity common, although the book says rare; the following Subtle item
is common. Breadcrumbs restore the parent name but do not carry every parent
attribute. That residual extraction error is outside the frozen questions and
remains a limitation.

Validation verified all **759** original source spans, persisted contextual
chunks, and contextual embedding inputs; every chunk reached both ingestion
model stages. All **329** retrieved chunk appearances across the 48 questions
were matched against actual reader requests. All 48 sessions were independent.
The graph contains 4,335 nodes and 19,700 edges. All 1,592 saved completed model
calls returned HTTP 200 and stopped naturally, with no schema-validation retry.
Saved completed calls do not account for requests lost in the VM crash.

### Recovery and timing limits

The client VM crashed after six sections completed. A storage backup was made,
then the identical code, inputs, model and eight-call setting resumed through a
user service. The six completed sections were retained; the interrupted seventh
section was reprocessed. Its 119 chunk IDs and texts were stable, and the final
759-chunk graph census passed. The runner returned zero. All **4,430** inventoried
original stock/raw JSON audit artifacts remained unchanged. The DGX container, image, startup
time, embedding digest, and Cognee source commit were unchanged at the end.
The recovered client service used 12 min 30 s of CPU over 2 h 35 min 38 s of
wall time, with a journal-reported 2.9G memory peak. This does not diagnose the
earlier VM crash.

\* The 244.52-minute value sums successful per-document `remember` attempts. It
excludes VM downtime and the interrupted chapter attempt; captured model-call
totals retain 24 completed calls from that partial attempt. This recovered run
does **not** establish a controlled ingestion speed comparison. The four-call
corrected rebuild stopped before the eight-call restart is a separate partial
pilot and is excluded from these totals.

At the user's request, the corrected run used eight concurrent calls, compared
with four in the earlier arms. Query execution remained sequential. Latencies
are observations on a shared server at different times, rather than a causal
measurement of heading context alone.

The bottleneck profile supports model inference as the main ingestion cost:
full-book preparation and token counting took 7.22 seconds, versus median
upstream times of 45.24 seconds for graph extraction and 10.89 seconds for
summaries. Embedding requests had a 0.37-second median. For the first two matched
sections, raising the request limit from four to eight reduced time from 19.33
to 18.49 minutes, about 4.35% in a single-pass pilot. This is not a repeatable
speedup estimate.

Review remains a Codex assessment, not independent human GM adjudication. q38
retains the original scoring of its requested trigger and combat round; its
battle label is imprecise and should be clarified in a future fixture. Questions
and rubric were not retuned for the follow-up.

The [corrected metrics and labels](experiments/adaptive-chunking/cognee/context-results.json)
and [runbook](experiments/adaptive-chunking/cognee/README.md) include configuration,
recovery, and input-audit details. Seventeen local checks passed.

## Exploratory narrative question (2026-11-21)

A subsequent user question asked how the first half, culminating in Wave Echo
Cave, feeds into the second half. This is outside the frozen 48-question score.
Both saved indexes were copied into separate query roots and asked the same
question with `GRAPH_COMPLETION`, `top_k=15`, a fresh session, and up to 600
answer words. Original indexes were preserved; both copied graph counts stayed
unchanged. Every raw chunk marked as context was matched in the actual reader input.

The stock answer took 35.07 seconds and broadly recovered the shared-town,
psionic-goblin, and return-to-Phandalin transition. It overreached when describing
the first-half enemies as distractions or minor players in the larger
conspiracy; the source does not establish that hierarchy.

The corrected adaptive answer took 27.76 seconds but **failed the synthesis**.
It called the Forge of Spells the primary bridge and speculated about a
Forge/Obelisk/Drow Vault connection. Six of its seven retrieved raw chunks came
from Wave Echo Cave, with one from Chapter 1; none came from the introduction
or Chapter 5. The actual transition was absent from its prompt. The answer
partly admitted missing evidence, but still asserted the unsupported connection.

Source checking establishes the useful links: early psionic-goblin clues,
including the goblin corpse in W1; a return to Phandalin after rescuing Nundro;
the party's mine-profit stake; and the new goblin crimes that launch Chapter 5's
investigation. The W1 scouting group explicitly found no obelisk shards. That
is a clue to the parallel shard hunt, not evidence that the Forge powers the
obelisk ritual.

This probe shows that improved fact-question results do not establish reliable
cross-chapter synthesis under the original retrieval settings. The two response
times are single observations with a longer answer budget, not comparable to
the earlier short-answer p95 metrics.
The [probe receipt](experiments/adaptive-chunking/cognee/narrative-probe-results.json)
records settings and retrieval coverage; complete answers and prompts remain
private alongside the experiment artifacts.

### Retrieval-budget follow-up

The user challenged whether the harness had simply stopped retrieving too early.
Both original probes used one plain `GRAPH_COMPLETION` pass and one generated
answer, with a final budget of 15 graph relationships. They did not use iterative
context extension or query decomposition. A fixed relationship count does not
equalize evidence volume across chunk sizes.

Repeating the adaptive question with `top_k=60`, preserving the model, question,
index, session isolation, and answer instructions, raised input from 4,215 to
16,550 tokens. Time rose from 27.76 to 36.90 seconds. It recovered the actual
Chapter 4 handoff: the goblin scouting clue, escorting Nundro home, and the mine
profit share. It dropped the claimed Forge/Obelisk causal connection. All
retrieved raw context appearances passed the actual-request audit.

This substantially repaired the core synthesis, though the answer still
incorrectly called the second half Chapters 5–7 and admitted missing Chapter 5
details. The larger one-pass budget supports the user's explanation. The first
failure cannot establish an inherent narrative weakness of adaptive chunks;
retrieval budget and the harness's stopping rule are confounded with chunking.
That diagnostic still used one pass. The following test continued retrieval.

### Continued retrieval

An explicit gap-driven loop started from the saved adaptive `top_k=60` answer.
It asked the same DGX model to identify missing evidence and propose follow-up
queries, then ran those queries through Cognee on a fresh copy of the same index.
It did not inject the source-checked answer or manually selected book passages.
Final synthesis used the union of actually retrieved raw source chunks, with
intermediate generated answers excluded as factual evidence. This is a wrapper
around Cognee retrieval, not a test of its native context-extension mode.

The planner identified the missing Chapter 5 material and asked:

- `Chapter 5 Paths of Peril Indigo Sanctum`
- `Chapter 6 The Shattered Obelisk Drow Vault`

One additional round with these two searches added 14 passages, expanding the
source union from 26 to 40. Six new model calls covered planning, both retrieval
answers, synthesis, and two sufficiency checks. The successful attempt took
88.12 seconds beyond the 36.90-second seed query, or 125.02 seconds including it.
A first wrapper attempt failed while treating a typed Cognee response as a
dictionary; that attempt was preserved separately and excluded from these times.

The new answer recovered Chapter 5's crimes, the search for shards, and the
investigation leading to Ruxithid. It no longer claimed that Chapter 5 evidence
was missing, and the Forge/Obelisk causal invention was absent. All retrieved
context appearances passed the actual-request audit, and all 40 accumulated
raw passages were verified in the final synthesis input. Copied graph counts
and original benchmark artifacts stayed unchanged.

The model's sufficiency checks declared no gaps, but source review still found
overstatement: the corpse does not reveal its full story to the characters,
and Nezznar's defeat should not be described as causing the already-running
supernatural plot.
The corpse's player-knowledge qualifier was already in the final input; the
exact Chapter 5 sentence about background activity was not. The remaining
issues require both a closer source reading and a chronology check, beyond
the earlier recovery of Chapter 5's main transition.

A reviewer-directed second round therefore asked for passages establishing the
goblins' timeline relative to Nezznar's defeat and the limits of what characters
can learn from the W1 corpse. This third follow-up search added nine more
passages, raising the union to **49**. The final reader was explicitly asked to
distinguish chronology from causation and player knowledge from GM-only text.
Its revised answer states that the mind flayer plot was already operating in
the background, and that the characters cannot learn the corpse's full story
at that point. The corpse remains a clue rather than proof available to them.

The third search's generated answer itself still made a faulty timing inference;
it was not treated as source evidence. Final synthesis used the accumulated raw
passages, including the introduction's statement that the fanatics' activities
are invisible through most of the first four chapters. The exact Chapter 5
background-activity sentence was not retrieved. All 49 passages and the final
source citations passed the request/provenance audit.
The complete continuation used two additional retrieval rounds, three searches,
and eight new model calls; automatic sufficiency checks alone had stopped early.

This supports the user's stopping-policy explanation: continued retrieval
recovered the missing connection. It does not establish that iteration alone
caused every improvement, because final context assembly also changed to a raw
source union and the final grounding instructions were strengthened. The frozen
48-question scores remain unchanged.

### Relative performance of the narrative configurations

Stock was also run through the same gap-check and reviewer-directed source
validation procedure. Its planner proposed no automatic follow-ups; one common
validation search added seven passages to its five-source seed. Final synthesis
used 12 raw chunks. All retrieved context and final citations passed the source
audit, and the original indexes remained unchanged.

| Configuration | Time | Model calls | Total model tokens |
|---|---:|---:|---:|
| Stock, one pass, `top_k=15` | 35.07 s | 1 | 12,656 |
| Adaptive, one pass, `top_k=15` | 27.76 s | 1 | 4,981 |
| Adaptive, one pass, `top_k=60` | 36.90 s | 1 | 17,389 |
| Stock, gap checks plus source validation | 118.69 s | 5 | 81,125 |
| Adaptive, gap checks plus source validation | 182.00 s | 9 | 101,702 |

At the wider one-pass budget, adaptive's latency was approximately equal to
stock, rather than retaining the original under-retrieved answer's speed gain.
With the continuation procedure, adaptive took **1.53× as long** and used
**1.25× as many model tokens**. It made three follow-up searches versus stock's
one. Repeated planning, intermediate answers, and synthesis accounted for extra
model work.

Both final answers recovered the main transition. The adaptive final handled
the parallel plot and GM/player knowledge distinction more carefully; stock
still overstated the Spider's interference in the mind flayer plot. The stock
self-check also misused its `unsupported_claims` field to list supported facts.
Equal procedure therefore does not establish identical answer quality or a
reliable automatic stopping criterion.

These are single-query observations from a sequential prototype. Continuation
times include each seed answer, gap checks, follow-up searches, and the common
final validation/revision. Stock started at `top_k=15`, adaptive at `top_k=60`;
both used `top_k=15` for follow-ups. Indexing, storage copying/setup, manual
review, and the failed wrapper attempt are excluded. This is a configuration
comparison, not an isolated benchmark of chunking or native model throughput.

### One-question-at-a-time agent (2026-11-21)

The user's proposed agent was tested on fresh, isolated copies of both saved
indexes. Both started from the original narrative question at `top_k=15`,
without prior probe answers or human-selected follow-ups. After each Cognee
answer, a separate Qwen model call received that answer, the preceding
observations, and the accumulated raw source passages. It chose exactly one
next question or a final answer. The captured requests verify that every
decision followed the latest response and every follow-up followed that
decision. This is a Qwen-driven research wrapper, not Cognee's native
context-extension mode.

Both used the same policy, temperature 0, seed 604, disabled thinking, and
eight-search maximum. Repeated searches were rejected, with up to three
automatic protocol attempts before stopping. No reviewer intervened during
either measured run. Sources and generated tool answers were distinguished
explicitly in the agent instructions; only raw passages were factual evidence.

| Measurement | Stock index | Adaptive, headings preserved |
|---|---:|---:|
| Outcome | Repeated-question guard; no final answer | Agent finished; grounding errors remain |
| Searches | 2 | 5 |
| Model calls, including decisions and protocol retries | 6 | 10 |
| Client elapsed time* | 65.70 s until stall | 143.41 s to completion |
| Input tokens | 144,673 | 57,057 |
| Output tokens | 2,084 | 4,375 |
| Total model tokens | 146,757 | 61,432 |
| Accumulated raw source passages | 14 | 27 |

*These common elapsed measurements include the worker's Cognee startup but
exclude storage cloning, ingestion, excluded earlier attempts, and later source
review. Adaptive's research loop alone took 139.99 seconds. Stock's time is
cost until a stall, not latency to an answer. The shared model server was not
reserved; these are individual observations, not p50/p95 measurements.

Stock first asked for alternative Chapter 4/5 hooks if players missed the early
psionic-goblin clues. After receiving the answer, it proposed exactly the same
question three times despite protocol corrections. The guard stopped another
identical retrieval. All six model requests succeeded; this was a control-loop
failure. Stock's larger raw passages and repeated decision prompts explain
its high token cost despite fewer searches: agent calls consumed 110,019 of its
146,757 tokens, versus 40,396 of adaptive's 61,432.

Adaptive chose this sequence, reading each response before choosing the next:

1. The original Wave Echo Cave-to-second-half question.
2. The Chapter 4/5 transition, initially following the incorrect Drow Vault lead.
3. The Chapter 4 ending, Nezznar, and the Forge of Spells.
4. The actual Chapter 4 "What's Next?" section.
5. Chapter 5's hooks into the mind flayer plot.

The fourth search added no new raw passages and did not recover that section.
The fifth added seven passages, including the introduction's Chapter 5 summary,
and the agent chose to finish. It recovered the return to Phandalin, goblin
crimes, investigation, Ruxithid, and mind flayers. It dropped the original
Forge/Obelisk causal theory and correctly identified Chapters 5–8.

**The final adaptive answer still fails semantic grounding.** It invented a
note on the goblin corpse explaining the shard search, converting GM narration
into a discoverable clue. The source explicitly limits what characters can
learn then; that qualifier was outside the retrieved fragment and never entered
this run's source union. The final answer also claimed that the first-half
bandits were pawns or distractions in the mind flayer conspiracy. Retrieved
passages described the plots proceeding alongside one another, without that
hierarchy. The agent nevertheless declared no remaining gaps. The mine-profit
stake and Nundro handoff were also absent from its answer.

All seven search responses passed the actual-request source audit: 51 raw-chunk
appearances were verified. All nine agent calls, including stock's invalid
actions, received the latest answer and complete accumulated source union.
Adaptive's copied graph counts stayed unchanged; stock stopped before its final
graph-count check. All 4,430 previously protected historical JSON artifacts
remained byte-identical. Four local protocol checks passed. Provenance checks
establish what the model saw, not whether its interpretation was correct.

The first paired attempt is preserved separately: stock showed the same stall,
while adaptive hit a wrapper error because a search action omitted its unused
answer field. The parser was corrected to default unused fields to empty strings,
and both arms restarted fresh under the unchanged research policy. Those earlier
attempts used another 195,955 model tokens across 14 calls. Including them, this
experiment consumed **404,144 tokens across 30 calls**; their time and tokens are
excluded from the table and retained in the receipt.

The sequential agent therefore produced a different retrieval path and cost,
but did not produce two correct answers for a fair speed comparison. Adaptive
used fewer tokens than the earlier reviewer-assisted continuation (61,432 versus
101,702) and less research time (139.99 versus 182.00 seconds), while retaining
errors that review had previously corrected. This single question cannot establish
that either chunking method is better for agents. Frozen fact-question scores
remain unchanged.

The [sequential receipt](experiments/adaptive-chunking/cognee/sequential-results.json)
contains decisions, costs, source-audit counts, and the independent quality review.
Full answers, per-step observations, actual requests, and readable traces are saved
privately under `raw/pabtso/cognee-20261120/sequential-agent-20261121/`.

### Haiku on the adaptive index (2026-11-21)

At the user's request, the same sequential experiment was repeated with
**Claude Haiku 5.5** (`claude-haiku-5-5`) generating both Cognee reader answers
and research-agent decisions. The user explicitly approved sending retrieved
Phandelver passages and research prompts to Anthropic. The API key was read
from `~/.anthropic-key` into memory and was absent from all saved artifacts.

This reused an isolated copy of the existing corrected adaptive index. Stored
graph descriptions and summaries still came from Qwen ingestion; embeddings
remained on the same Ollama model. The question, research policy, `top_k=15`,
eight-search ceiling, fresh-session isolation, and automatic-feedback setting
were unchanged. The first reader's messages and seven raw chunks were exactly
identical to Qwen's first turn. Subsequent questions were chosen independently.

| Measurement | Qwen, adaptive index | Haiku, adaptive index |
|---|---:|---:|
| Searches | 5 | 2 |
| Model calls | 10 | 5 |
| Client time, including worker startup | 143.41 s | 28.72 s |
| Research loop, excluding worker startup | 139.99 s | 24.55 s |
| Input tokens | 57,057 | 28,633 |
| Output tokens | 4,375 | 3,418 |
| Total model tokens | 61,432 | 32,051 |
| Raw source union | 27 | 8 |
| Reviewed answer | Main handoff recovered; invented claims | Supported but substantially incomplete |

Haiku asked one follow-up about Chapters 5/6, the Drow Vault, and the transition
from Wave Echo Cave. That search added only the introduction's two-part summary.
It then attempted to finish while listing missing transition evidence. The
protocol guard rejected the contradictory gap list; Haiku removed the list and
kept the uncertainty in its answer, without retrieving again. It stopped with
six searches still available, despite acknowledging that further searching
could help. The fifth model call is this corrective final-answer call.

The final answer accurately described the two halves, level-5 handoff, and
Rockseeker/Spider setup, and explicitly left the narrative mechanism unverified.
It did **not** invent Qwen's corpse note, bandit/mind-flayer hierarchy, or
Forge/Obelisk causal connection. However, it missed the actual return-to-Phandalin
and goblin-crime hook, Nundro and the mine stake, and the investigation into
Ruxithid. No actual Chapter 5 passage or Chapter 4 ending entered its source
union. It also spent answer space on incidental streambed and rift details.
This is a cautious incomplete answer, not a correct complete answer to the question.

The measured successful run's estimated Anthropic API cost was **$0.0045723**,
using captured native usage and standard list rates. Two separately preserved
compatibility attempts added nine calls, 63,287 tokens and an estimated
$0.0086615. The first rejected a valid Markdown JSON fence; the second stopped
because prose around a corrected action bypassed the retry mechanism. Fence
parsing and bounded JSON-error feedback were repaired before the fresh measured
run, with the research policy unchanged. Across all Haiku attempts, usage was
**95,338 tokens across 14 calls**, estimated at **$0.0132338**. These are list-price
estimates, not billing statements; local retrieval and embedding costs are excluded.
[Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).

The native adapter disabled thinking and omitted unsupported Qwen sampling
options; Haiku used its provider-default sampling, without Qwen's seed or
temperature 0. JSON was requested by instruction rather than an enforced schema.
Reader replies had a 4,096-token ceiling and decisions retained 2,800 tokens;
no response reached either ceiling. No cache or thinking tokens were billed in
the captured usage. Provider tokenizers differ, so counts are not equal units
of text or compute. These settings follow the
[Haiku documentation](https://platform.claude.com/docs/en/models/haiku-5-5/overview).

All five actual native API bodies preserved the local message content; all
requests succeeded and ended naturally. The two searches' eight raw-source
appearances, and all three agent calls' latest observation and source union,
passed the request audit. Graph counts stayed unchanged, 30 protected adaptive
source artifacts and 4,430 historical JSON artifacts remained byte-identical,
and eight local protocol/provider checks passed.

Haiku's observed run was shorter and cheaper in model-token count, but answered
less of the question. Qwen retrieved more of the real handoff and then overreached;
Haiku abstained before using the available search budget. Neither outcome
establishes a reliable stopping rule or a speed advantage at equal quality.
This is one question, with both the reader and agent changed together; frozen
fact-question scores remain unchanged.

The [Haiku receipt](experiments/adaptive-chunking/cognee/haiku-sequential-results.json)
contains costs, decisions, provenance checks and source review. Full answers,
native requests and readable traces are private under
`raw/pabtso/cognee-20261120/haiku-sequential-20261121/`; the two excluded attempts
are retained in adjacent `-attempt1` and `-attempt2` directories.

### Sonnet on the adaptive index (2026-11-21)

The requested completeness check repeated the same adaptive-index experiment
with **Claude Sonnet 5.5** (`claude-sonnet-5-5`) as both reader and research
agent. It started fresh from the original question, using the same policy,
`top_k=15`, eight-search cap, saved Qwen-ingested graph and summaries, and Ollama
embeddings. The original reader messages and seven initial raw chunks were
identical to the other runs. No prior answers or reviewer-selected follow-ups
were supplied.

| Measurement | Qwen | Haiku 5.5 | Sonnet 5.5 |
|---|---:|---:|---:|
| Searches | 5 | 2 | 3 |
| Model calls | 10 | 5 | 6 |
| Client time, including worker startup | 143.41 s | 28.72 s | 52.98 s |
| Research loop, excluding worker startup | 139.99 s | 24.55 s | 48.86 s |
| Input tokens | 57,057 | 28,633 | 56,833 |
| Output tokens | 4,375 | 3,418 | 4,799 |
| Total model tokens | 61,432 | 32,051 | 61,632 |
| Raw source union | 27 | 8 | 24 |
| Reviewed outcome | Main bridge found; invented links | Cautious but incomplete | Main bridge found and supported |
| Estimated API charge, measured run only | Local compute, unpriced | $0.0045723 | $0.161656 |

Sonnet asked the original question, then asked what follows Nezznar's defeat
and what clues carry forward. That second search retrieved the actual Chapter
4 ending: the mine-profit share, optional Nundro homecoming, strange goblins,
and return to a town experiencing new crimes. It then asked how the goblins,
map and mind flayers connect Chapters 5 and 6. The third search added eleven
raw passages, including Chapter 5's investigation and fanatics background,
the return-to-town scene, and Chapter 6's shard hunt. It finished after reading
that response, with no protocol corrections or wrapper failures.

**This was the strongest reviewed answer of the three on this question.** It
correctly treats the Lost Mine/Spider arc as a self-contained plot with an
optional continuation. It connects early psionic-goblin clues, the return to
Phandalin, Chapter 5's investigation, Ruxithid and the later shard hunt. It also
includes the mine-profit reward and level handoff. Its 584-word answer cites
the retrieved evidence and marks its synthesis and remaining uncertainty.
It does not invent a corpse note, a bandit/mind-flayer hierarchy, or a magical
Forge/Obelisk causal mechanism. It does not leave the principal handoff
unanswered as Haiku did.

This is not an exhaustive correctness finding. The W1 sentence limiting what
characters can learn from the corpse was absent from the retrieved union;
Sonnet does not explicitly state that restriction, although it does not invent
a player-readable message. The exact Chapter 5 background-activity sentence
was also absent. The answer presents prior goblin scouting and separate arcs
without claiming Nezznar's defeat caused the fanatics' plot. The model answer
was preserved unchanged after Codex's source review.

Sonnet's supported minimum thinking setting is `between_tools`, whereas Haiku
accepts `disabled`; the API capabilities and actual native requests were
checked. Up-front thinking was suppressed and native usage reported **zero
thinking tokens** in all six calls. Sampling stayed at the provider default,
without Qwen's temperature or seed. No prompt-cache tokens or truncation were
recorded. Reader and agent token ceilings remained 4,096 and 2,800 respectively.
[Sonnet API controls](https://platform.claude.com/docs/en/models/sonnet-5-5/overview).

The estimated **$0.161656** API charge uses captured native input/output usage
and standard rates of $2/$10 per million tokens. There were no excluded Sonnet
attempts. This is a list-price estimate, excluding local retrieval and embedding
costs, rather than an invoice.
[Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).

All three search responses passed the request audit, with 28 contextual chunk
appearances verified. All three agent calls received the latest answer and the
complete accumulated raw source union; all six native API bodies preserved the
input messages and ended naturally. All 17 individual or grouped citation
labels exist in the retrieved source union. Cloned graph counts stayed unchanged;
30 protected adaptive artifacts and 4,430 historical JSON artifacts remained
byte-identical. The key was absent from saved artifacts. Nine relevant local
checks passed.

Sonnet used fewer searches and less observed time than Qwen while giving a
better-grounded answer here. Haiku remained quicker and cheaper but did not
complete the research. This is one question with different provider controls
and tokenizers, and both the reader and agent changed together. It supports
testing model-dependent search and stopping behavior; it does not establish
general model superiority or a chunking-specific gain. Frozen fact-question
scores are unchanged.

The [Sonnet receipt](experiments/adaptive-chunking/cognee/sonnet-sequential-results.json)
contains metrics, decisions and source review. Full answers, native captures,
evidence and readable traces are saved privately under
`raw/pabtso/cognee-20261120/sonnet-sequential-20261121/`.

### Inkling 2 Mini on Spark 2 (2026-11-21)

The user requested another local-model comparison after the Qwen, Haiku and
Sonnet experiments. The original `thinkingmachines/Inkling-2-Mini` checkpoint
was installed from revision `b85c3cfa615fc413cc9af3379f491b963f3b586a`, using
the official ARM64 vLLM image pinned by digest. Both the 19.5 GB main weights
and bundled 636 MB draft passed SHA-256 checks. The draft was downloaded but
speculative decoding was not enabled.

Inkling runs in its own container on Spark 2, alongside Clef, with a separate
OpenAI-compatible endpoint at `http://192.168.1.121:8004/v1` and served name
`inkling-2-mini`. The runtime reported vLLM `0.1.dev25855+g7cb5dae22`, PyTorch
`2.14.0+cu130`, and CUDA 13.0 on the GB10. The source checkpoint's native
mixed precision was retained. Text-only operation, a 65,536-token context,
one active request, eager execution and GPU-memory utilization 0.26 kept the
installation within the shared host's available memory. Model loading reported
17.64 GiB of weight memory and 12.92 GiB available for KV cache. These settings
differ from a dedicated-host performance configuration.

Plain-text and JSON smoke requests passed. The same adaptive index, original
question, `top_k=15`, eight-search cap, agent policy and Ollama embeddings were
then reused with Inkling as both reader and agent. The original reader messages
were exactly identical to Qwen's first turn. No previous answers or manually
selected follow-ups were supplied.

| Measurement | Inkling same-policy trial |
|---|---:|
| Completed searches | 2 |
| Model calls | 6: two readers and four agent calls |
| Client time until guard failure | 137.94 s |
| Input tokens | 26,982 |
| Output tokens, including reasoning | 3,447 |
| Reasoning tokens within output | 1,946 |
| Total tokens | 30,429 |
| Accumulated raw passages | 9 |
| Final answer | None |

The first decision selected a Chapter 5/6 follow-up and met the action contract.
After reading the second response, all three permitted attempts returned a
shape equivalent to:

```json
{"action": "search", "args": [{"query": "a new Chapter 5 question"}]}
```

The required top-level query, decision summary and gap list were missing. The
guard rejected the actions, and the run stopped. The model was choosing a new
question, but could not express it in the specified control schema despite
feedback. All six HTTP calls succeeded and ended naturally; this was not an
API transport, truncation or model-server crash. No final narrative answer
exists to grade. Native tool calling was not exercised by this custom JSON
contract, so its behavior remains unassessed.

`reasoning_effort="none"` was verified in all captured requests. It reduced the
requested effort without guaranteeing zero reasoning: native usage recorded
1,946 reasoning tokens, already included in the output and total counts.
The two excluded smoke calls took 10.83 seconds and 187 tokens. Installation,
download, initial kernel preparation, and source review are excluded from the
137.94-second guard-failure time. Local hardware and electricity were unpriced.
This time cannot be compared as latency to a successful answer.

Both searches passed the actual-request source audit, verifying nine raw-chunk
appearances. All four agent calls received the latest response and accumulated
raw source union. Thirty protected adaptive artifacts and 4,430 historical
JSON artifacts remained byte-identical. The failed worker stopped before its
final copied-graph count check, so that field remains unverified. All 26 relevant
local checks passed after adding the local endpoint and reasoning-effort options.

Inkling remained healthy after the trial. Clef's original container ID, image
and start time were unchanged, and the Clef and Spark 1 Qwen health endpoints
both returned HTTP 200. Shared-host available memory was about 54 GiB before
installation and 21 GiB afterward; swap use rose from roughly 0.65 to 2.2 GiB
across installation/testing. Those observations do not isolate model footprint
or effects of other concurrent activity.

The [Inkling receipt](experiments/adaptive-chunking/cognee/inkling-sequential-results.json)
contains the rejected actions, request audit and usage. The
[installation/client guide](experiments/inkling/README.md) provides the pinned
launcher and OpenAI example. Full observations, captures and a readable trace
are private under `raw/pabtso/cognee-20261120/inkling-sequential-20261121/`.
This is a completed negative experiment for the existing action protocol;
the factual scores and earlier model answers are unchanged.

### Inkling schema-guided workaround (2026-11-21)

The format failure was addressed with an opt-in `--action-format json_schema`
mode. The server now receives the existing five-field action contract as a
JSON schema with required fields and `additionalProperties:false`. Reader calls,
retrieval policy, reasoning effort, model weights and the semantic validator
are unchanged. No server restart was needed. The default remains plain
`json_object`, preserving the original comparison mode.

First, the exact previously failed decision context was replayed with only
`response_format` changed. The constrained response supplied a top-level query,
decision summary and gap list, and passed the original validator. This diagnostic
took 7.38 seconds and 5,497 tokens, including 45 reasoning tokens. It was not
used to seed the fresh research run.

| Measurement | Plain JSON baseline | Schema-guided fresh run |
|---|---:|---:|
| Outcome | Action-schema guard; no final answer | Agent finished autonomously |
| Searches | 2 | 7 |
| Model calls | 6 | 14 |
| Client time | 137.94 s until failure | 499.32 s to completion |
| Input tokens | 26,982 | 90,787 |
| Output tokens | 3,447 | 14,191 |
| Reasoning tokens within output | 1,946 | 8,589 |
| Total tokens | 30,429 | 104,978 |
| Raw source union | 9 | 31 |
| Invalid action attempts | 3 | 0 |

The schema-guided agent continued past the original failing decision, spent
several searches on Chapter 5/6 map details, then examined the Spider's relation
to the later plot and the actual Chapter 4 ending. Search 7 retrieved that
ending; the agent chose to finish before the eight-search cap. Its final answer
recovered Nundro's homecoming, the mine-profit stake, the return to Phandalin,
and the goblin crimes and investigation that lead into the mind flayer plot.
It qualified the lack of an explicit Nezznar-to-mind-flayer link and did not
claim a Forge/Obelisk power connection. The answer is 431 words and was not
revised after the run.

This fixes control-format interoperability. Grounding still required review.
The answer broadly locates the Chapter 6 dungeons beneath Phandalin, echoing
a retrieved caption, while the full map description more precisely places
them in the Starmetal Hills northeast of town. Its phrase "goblin-body note"
is ambiguous: the retrieved passage is GM narration, not a physical message.
Unlike Qwen's earlier claim, it does not explicitly assert that the corpse has
a player-readable note. The character-knowledge restriction remained outside
the retrieved union. These qualifications are recorded in the source assessment.

All seven reader responses passed the request audit, with 38 contextual chunk
appearances verified. All seven agent calls received the latest response and
full accumulated source union; all fourteen calls requested the pinned model
and lowest reasoning effort, and all agent requests used the schema. They
returned HTTP 200 and ended naturally. Copied graph counts and 30 protected
adaptive artifacts stayed unchanged, as did 4,430 historical JSON artifacts.

The 499.32-second client time includes worker startup; the research loop alone
took 494.64 seconds. The exact-context diagnostic, earlier smoke calls,
installation and review are excluded. The new run plus its format diagnostic
used 110,475 tokens. Local hardware/electricity remained unpriced. The failed
baseline is preserved, and cache state and server load were not reset or matched.
This is a changed-decoding-constraint diagnostic, with a different search path
and amount of work, rather than a speed comparison at equal quality.

The [schema-guided receipt](experiments/adaptive-chunking/cognee/inkling-schema-sequential-results.json)
contains the before/after metrics, replay check, schema and source review. The
[workaround guide](experiments/inkling/README.md#schema-guided-action-workaround)
shows the opt-in flag, following
[vLLM's structured-output interface](https://docs.vllm.ai/en/latest/features/structured_outputs/).
Full answers and traces remain private under
`raw/pabtso/cognee-20261120/inkling-schema-sequential-20261121/`.

### Inkling original 48 factual questions (2026-11-21)

The user also requested the original question benchmark, independent of the
agent flow. Inkling answered all 48 frozen factual questions against a fresh
copy of the corrected adaptive index. Each question made exactly one Cognee
`GRAPH_COMPLETION` recall, with `top_k=15`, the original 120-word instruction,
temperature 0, seed 604 and a fresh UUID session. Questions ran sequentially.
There were no agent decisions, follow-up searches, ingestion calls or output
schemas. The original Qwen graph, summaries and Ollama embeddings were reused.

All 48 actual reader message lists and all 48 retrieved raw-chunk sets were
identical to the corresponding Qwen baseline. This compares answer generation
over the same retrieved evidence, rather than independently rebuilt indexes.

| Measurement | Qwen reader, corrected adaptive | Inkling reader, same index |
|---|---:|---:|
| Completed questions / reader calls | 48 / 48 | 48 / 48 |
| Complete correct answers, strict source review | 45/47 | 44/47 |
| Complete correct answers, lenient q20 wording | 45/47 | 45/47 |
| Recall p50 | 4.68 s | 12.10 s |
| Recall p95 | 6.16 s | 18.60 s |
| Sum of all 48 query times | 226.79 s | 636.25 s |
| Input tokens | 186,523 | 204,946 |
| Output tokens | 2,808 | 14,674 |
| Reasoning tokens within output | Thinking disabled | 11,497 |
| Total model tokens | 189,331 | 219,620 |

Inkling was **2.59 times slower at the median**, **3.02 times slower at p95**,
and **2.81 times slower over the full question set**. Its query sum was
10 minutes 36 seconds, versus 3 minutes 47 seconds for Qwen. Including worker
startup and the final graph check, Inkling took 641.08 seconds. Installation,
index cloning, earlier agent tests and source review are outside these times.
All measured answers stayed within the requested word limit.

The two existing retrieval misses remained: q04 lacked the flood timing and
q22 lacked Ruxithid's rudimentary clairvoyance. Inkling additionally conflated
the flood warning/command location with the release point and used general
telepathy background as a substitute for the missing clairvoyance explanation.
All six adjacent-item questions, including Careful Mind Crystal, were answered
correctly. No new complete answer was recovered relative to Qwen.

The one-point scoring sensitivity is q20. The source says Ruxithid does not
know that his supposed gods are mind flayers. Inkling's sentence describes
him as misunderstanding his gods as being mind flayers, which can be read as
reversing his belief. The strict review counts this misleading wording as
incorrect; interpreting it as an awkward statement of the intended source
fact gives 45/47. Both interpretations are recorded, with the unchanged answer
preserved. Neither produces a factual accuracy gain over Qwen. The predeclared
q37 ambiguity remains excluded; q38 retains the original condition-and-round
rubric. This is Codex source review, not blinded human adjudication.

Despite `reasoning_effort:"none"`, Inkling reported 11,497 reasoning tokens:
78.35% of its output. Its input tokenizer counted 9.88% more tokens for the
same messages, while total output was 5.23 times Qwen's. Captured upstream
model time was 565.80 seconds, 88.93% of Inkling's query sum. These observations
suggest generation work is a substantial contributor to the latency; they do
not isolate tokenizer speed from model computation or server configuration.

Both models used vLLM. Qwen was the retained Spark 1
`RadixArk/Qwen3.8-Flash-Next-NVFP4` checkpoint at revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`, with FP8 hybrid side layers,
MTP-2 speculation and prefix caching. Inkling used the separate Spark 2
configuration documented above, with eager execution and no speculative
decoding, beside Clef. Thus these are deployed-configuration measurements,
with different runtimes, tokenizers and shared-device conditions, rather than
a controlled intrinsic model-speed benchmark. Hibana supplied the capture
harness; its inference engine was stopped. The earlier Hibana `fast` profile
used a separate W3-calibrated checkpoint,
`yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3-qad`; that copy was not used here.

All 48 reader calls returned HTTP 200 and stopped naturally. The audit verified
329 contextual chunk appearances, 48 unique sessions and unchanged cloned
graph counts. All 30 protected adaptive artifacts and 4,430 historical JSON
artifacts retained their hashes. Qwen, Inkling and Clef remained healthy with
unchanged container identities and start times.

The [factual-question receipt](experiments/adaptive-chunking/cognee/inkling-question-results.json)
contains per-question timings, token usage, labels, prompt-equality checks,
model identity and the scoring sensitivity. The
[question-only runner](experiments/adaptive-chunking/cognee/questions.py) and
[run instructions](experiments/adaptive-chunking/cognee/README.md#factual-questions-with-a-different-reader-model)
make this separate from the agent protocol. Full answers, prompts and review
notes remain under ignored
`raw/pabtso/cognee-20261120/inkling-questions-20261121/`.

### Hibana fast: original questions and module synthesis (2026-11-21)

Three fresh trials used independent copies of the corrected adaptive index:
48 single-recall factual questions, one single-recall Wave Echo Cave answer,
and the same sequential research agent. No documents were reingested. Graph
extraction, summaries and Ollama embeddings remained from Qwen. The live server
verified the pinned Hibana `fast` profile and W3 checkpoint
`yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3-qad` at
`5c4c62a62853caebea8cedda7fbed952efa1effe`. It used eight MTP draft tokens,
FP8 KV, expert sharing and relaxed acceptance. Weights, precision and decoding
policy change along with the engine; this does not isolate an engine-only gain.

| Factual benchmark | Qwen vLLM | Inkling vLLM | Qwen Hibana fast |
|---|---:|---:|---:|
| Complete correct answers | 45/47 | 44/47 strict; 45/47 lenient | 44/47 |
| Query p50 | 4.68 s | 12.10 s | **3.41 s** |
| Query p95 | 6.16 s | 18.60 s | **4.32 s** |
| Query sum, all 48 | 226.79 s | 636.25 s | **168.99 s** |
| Input tokens | 186,523 | 204,946 | 186,523 |
| Output tokens, including reported reasoning | 2,808 | 14,674 | 3,033 |
| Total tokens | 189,331 | 219,620 | 189,556 |

Hibana fast was **1.37 times faster at the median**, **1.43 times faster at
p95**, and **1.34 times faster over all 48 queries** than pinned Qwen vLLM.
The query sum was 2 minutes 49 seconds; worker startup and the final graph
check bring it to 173.82 seconds. All 48 actual reader messages matched Qwen's
exactly, and both counted the same input tokens. Tokenizer input length does
not explain this speed difference. Questions ran sequentially at `top_k=15`
with the unchanged 120-word instruction, temperature 0, seed 604 and fresh sessions.

The existing q04 flood-timing and q22 clairvoyance retrieval misses remained
clean abstentions. All six item questions passed. The new incorrect answer was
q09: after naming the historical spellbook recipient correctly, Hibana fast
claimed Agatha knew the book's location while also admitting she did not know
what happened afterward. A supplied graph node already conflated Agatha with
Old Owl Well; the raw passage explicitly limited her knowledge.

Two otherwise complete answers added unsupported qualifications. Q20 extended
Ruxithid's ignorance of his supposed gods' identity into ignorance of his own
words; q31 compressed the sinkhole route and named Illithinoch stronghold into
a precise location/name claim about the suspected elder brain. Requested facts
were correct in both, so they retain complete-answer credit. Strict review of
all extra assertions leaves **42/47 fully grounded complete answers**. The
predeclared q37 exclusion and q38 condition-and-round rubric remain unchanged.
These are Codex source judgments, not blinded human scores.

| General module question | Qwen vLLM | Hibana fast |
|---|---:|---:|
| One-pass query time | 27.76 s | **14.87 s** |
| One-pass calls / tokens | 1 / 4,981 | 1 / 5,023 |
| Agent client time | 143.41 s | **84.51 s** |
| Agent searches / model calls | 5 / 10 | 5 / 10 |
| Agent total tokens | 61,432 | 60,389 |
| Agent raw source union | 27 | 24 |

The one-pass answer received the same seven raw passages and identical reader
messages as Qwen. It still missed the actual transition, guessed that the
second part began at Chapter 6, and treated Nezznar, the Forge and deeper Drow
Vault content as the bridge. Its missing-evidence paragraph did not retract
those claims. Faster generation did not repair missing retrieval.

The agent used the unchanged policy, `top_k=15`, eight-search cap and plain
`json_object` actions. Five searches and ten calls completed with zero protocol
corrections. Its final 432-word answer recovered the chapter ranges, level
progression and broad introduction summary about psionic-goblin crimes. It
did not invent a physical corpse message or Forge-to-Obelisk power link.

However, it made the absence of shards in Wave Echo Cave into a discovery
prompting the party to investigate elsewhere. Retrieved source S009 contained
the full W1 restriction that characters cannot learn anything further then;
the shard-scout explanation is GM narration. The distinction was available
and still lost. The agent never retrieved the actual Chapter 4 ending, Nundro
homecoming, mine-profit stake or precise return-to-Phandalin hook. It stopped
after identifying that gap, with three searches unused. This is a broader but
still flawed answer; Sonnet remains the strongest reviewed handoff.

The agent research loop took 75.42 seconds, or 84.51 seconds including worker
startup and final checks. Its 1.70-times shorter client time than Qwen does not
establish equal-quality speed: query paths and source unions differed. The
one-pass 1.87-times speedup likewise compares two flawed answers. Timings
exclude model switching, source review, cloning, prior ingestion and the
separate arithmetic smoke request. Cache state and host conditions were not matched.

Audits verified all 48 factual prompts and 329 source appearances, the one-pass
reader's seven appearances, and the agent's 36 appearances and five
latest-answer/source-union inputs. All measured calls returned HTTP 200 and
stopped naturally. Cloned graph counts and protected original artifacts stayed
unchanged. Full answers, captures and review notes remain under ignored
`raw/pabtso/cognee-20261120/hibana-fast-20261121/`.

Separate receipts preserve the [48-question results](experiments/adaptive-chunking/cognee/hibana-fast-question-results.json),
[one-pass module answer](experiments/adaptive-chunking/cognee/hibana-fast-onepass-results.json)
and [sequential agent](experiments/adaptive-chunking/cognee/hibana-fast-sequential-results.json).

Spark 2's Clef and Inkling containers were temporarily stopped for Hibana's
memory requirement, then restored with their original IDs and images. Both
are healthy; Hibana is stopped and the tunnel closed. Spark 1's Qwen container,
image and start time stayed unchanged. Restoration plus readiness verification
took 651.09 seconds, including correction of a probe that used localhost for
Inkling even though its published endpoint binds to the Spark's LAN address.
This operational time is excluded from query timings; exact first-available
time was not measured. The [operations receipt](experiments/adaptive-chunking/cognee/hibana-fast-operations.json)
also reconciles all 60 server requests with 59 measured calls and one smoke.

## Original raw-chunk comparison (2026-11-20)

The original adapter fed selected raw chunks into Cognee without their heading
breadcrumbs. It reduced complete correct answers from 46/47 to 43/47, while
queries were about 2.1× faster at p95 and ingestion took 3.55× longer. The
historical measurements and diagnostic below are preserved.

Both arms ingested all 13 expanded module sections and answered the same 48
independent questions. The predeclared q37 geography ambiguity is excluded
from correctness, leaving 47 adjudicable questions.

| Measurement | Current Cognee | Adaptive, raw chunks |
|---|---:|---:|
| Persisted chunks, exact census verified | 78 | 759 |
| Ingestion | 64.85 min | 230.28 min (3 h 50 min) |
| Complete correct answers, Codex review | 46/47 | 43/47 |
| Complete correct answers supported by supplied context | 46/47 | 43/47 |
| Clean abstentions | 0 | 3 |
| Incorrect or misleading responses | 1 | 1 |
| Recall latency p50 | 9.87 s | 4.84 s |
| Recall latency p95 | 14.04 s | 6.65 s |
| Recall time, all 48 questions | 458.54 s | 240.74 s |
| Median recall prompt tokens | 17,084.5 | 3,635.5 |
| Complete frozen gold passage in raw context | 46/48 | 45/48 |
| Existing lexical spot checks | 12/12 | 11/12 |
| Graph nodes / edges | 1,410 / 5,287 | 4,702 / 19,124 |
| Primary captured LLM calls | 206 | 1,568 |
| Primary API tokens, input + output | 1,513,455 | 2,877,998 |
| Recovered schema-validation retries | 1 | 1 |

Primary metrics exclude the archived default-session pilot and the two
post-trial diagnostic calls. Both arms had 48 distinct query sessions and 48
captured-prompt history audits. Every primary captured model call returned
HTTP 200 and finished naturally. No terminal schema-validation failure was
logged.

The review is a Codex assessment against frozen source evidence and actual
retrieval prompts, rather than independent human GM adjudication. “Complete”
means the answer delivers the requested facts without a material factual
contradiction. Context support alone is insufficient when the supplied graph
or summary already contains an error.

## Original answer changes and grounding failure

Current Cognee supplied the correct facts for q26, q29 and q46; adaptive did
not. Adaptive gained no newly complete answers.

- **q04, hideout flood:** current Cognee confidently asserted immediate
  flooding. The module specifies knocking supports away in the round after
  a warning, then flooding in the following round. Adaptive correctly
  admitted that its retrieved evidence did not establish the answer. Both
  failed to retrieve the timing passage.
- **q26, burial names:** adaptive missed the explanation that Dumathoin
  grants the Talhund new, secret names in death and abstained.
- **q29, fanatics:** adaptive retrieved collective background about the
  fanatics, but omitted their names and beliefs about being Ilvaash's
  favorite. It abstained. Current Cognee answered fully.
- **q46, Careful Mind Crystal:** adaptive declined the requested mechanics
  and repeated incorrect rules for other variants. This failure began in
  ingestion: a chunk carried Careful's effect without its parent heading,
  followed by the next item's Distant heading. Its generated summary
  attributed Careful's effect to Distant. Another summary similarly assigned
  Heightened's effect to Quickened. The reader repeated these corrupted
  summaries while claiming Careful's effect was unavailable.

The q46 source rule was present in the captured prompt. The problem was its
association with the correct item. All **321** current and **329** adaptive
raw-chunk appearances marked as context were matched in full against the
actual SDK request text. Provenance and valid JSON therefore did not guarantee
correct entity attribution.

q27 illustrates a different measurement limit: current Cognee covered 94.9%
of the frozen gold passage and still answered every requested fact correctly.
The missing tail was unasked background. Whole-passage coverage is a
retrieval diagnostic rather than semantic answer accuracy.

### Focused heading diagnostic

After the scored trial, the exact problematic summary request was replayed
with the same model, temperature and seed, using a common 2,048-token output
cap. Both calls finished naturally:

| Input | Observed summary attribution |
|---|---|
| Original raw fragment | Again assigned Careful's effect to Distant |
| Same fragment with its existing frozen breadcrumb | Correctly assigned that effect to Careful |

The breadcrumb was already present in the earlier frozen retrieval plan:
module → Appendix B → Mind Crystal → Mind Crystal (Careful). This diagnostic
supports retaining parent context for both graph extraction and summaries.
That diagnostic comprised two summary calls and rebuilt no index. The full
corrected follow-up above is a separate experiment.

## Original controlled setup

- Full expanded text of *Phandelver and Below: The Shattered Obelisk*,
  including appendices, story tracker and credits. Maps and images are
  represented by captions.
- Current arm: installed Cognee `TextChunker`, automatic 4,096-token budget.
  Adaptive arm: the earlier question-independent per-section four-metric
  selection, 759 frozen chunks. The coreference metric is omitted. Both arms
  use raw source chunks without added heading breadcrumbs.
- LLM: Spark 1's existing vLLM `qwen3.8-flash-next`, through the existing
  Hibana/Cognee `Capture` proxy. Hibana itself is not serving this comparison,
  following the user's explicit backend selection.
- Embeddings: Ollama at `192.168.56.1:11434`, `qwen3-embedding:0.6b`,
  1,024 dimensions. Native `/api/embed` calls use `truncate:false` and validate
  counts, dimensions and finite nonzero vectors.
- Cognee source `b32d8afc59e1064d9291b9828a8a147be9cc8bab`, existing Python
  environment reused read-only. Installed distribution metadata reports
  1.5.3; the editable source commit is authoritative. Both arms used matching
  dependency versions, tokenizer and chunk cap.
- Separate relational, graph, vector, cache and log roots. Four LLM calls at
  a time, four chunks per batch, one source section at a time. Temperature 0,
  seed 604, thinking disabled.
- `remember(self_improvement=False)` and
  `recall(GRAPH_COMPLETION, auto_route=False, top_k=15, include_references=True)`.
  Automatic feedback is disabled. Every question receives a fresh UUID
  session, and captured requests are checked for other benchmark questions.

The frozen expanded-corpus JSON fingerprint is
`415f55e8cd9bf8ea9f60b0d53c9b1bb3e78f50db67c6358c38afcc1e0a861df0`.
The question-fixture fingerprint is
`0d346af78ffc9dd18cc9ca0b3ba12a62a6ed7793a98597c633a11faf5d296359`.
The embedding digest is
`ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d`.
The serving container and image pin, and the embedding digest, were verified
unchanged at the end.

## Original operational findings and limits

An initial 22 baseline responses were archived after request inspection showed
that Cognee's default cached session inserts earlier generated answers into
later prompts, despite `AUTO_FEEDBACK=false`. Independent recall was restarted
with fresh sessions. The complete saved graph was verified byte-for-byte
identical across that restart, with no ingestion rerun. Original manifests,
code snapshots and pilot requests remain private. Pilot responses are excluded
from primary scores and latency; total wall time and embedding counts retain
their overhead. The unmatched pilot warmup also limits latency interpretation.

Native SDK calls used JSON-object mode and omitted an explicit output-token
parameter. The configured `LLM_MAX_COMPLETION_TOKENS=16384` was not a request
output cap. Each arm recovered one schema-validation error. The current
credits extraction alone took 285 seconds and 13,289 output tokens.

Both arm summaries and the controller recorded completion. When collected
afterward, the long-lived outer tool session returned exit code 143. Its cause
is unestablished; all source censuses, 96 question checkpoints, captured
requests and completion artifacts were independently revalidated. This outer
session status is retained in the execution audit.

This is one book and one pass over 48 fact questions. Chunk size, overlap,
chunk count and preserved heading context differ; the outcome cannot isolate
the selector's contribution. The adaptive plan was selected before these
queries and was not retuned. This original result applies to the raw-chunk
integration; the corrected follow-up above adds breadcrumbs and uses eight calls.

Cognee falls back to its TikToken tokenizer for the Ollama-style model name;
the adaptive boundaries were selected with `o200k_base`. Actual embedding
input fit was checked by Ollama with truncation disabled. The serving instance
was shared rather than reserved, so elapsed times are observations under this
setup. Reported recall latency includes retrieval and generation; TTFT and
inter-token latency were not measured.

## Saved artifacts

The [original metrics and reviewed labels](experiments/adaptive-chunking/cognee/results.json)
contain the first two arms, and [corrected metrics](experiments/adaptive-chunking/cognee/context-results.json)
contain the follow-up. The [runner and protocol](experiments/adaptive-chunking/cognee/README.md)
describe reproduction. Both original full-book runs completed; the corrected
run completed after VM recovery, with seventeen local checks and the full
source, model-input, embedding, and reader-context audits passing.

Private source text, vectors, graph dumps, prompts, answers, review notes,
checkpoint history and the focused replay remain under ignored
`experiments/adaptive-chunking/raw/pabtso/cognee-20261120/`.
