# Phandelver in Cognee: adaptive chunking and sequential retrieval

Initial experiment completed November 21, 2026; native thinking follow-up
completed November 22, 2026 (Pacific).

Adaptive chunking made short factual queries faster in this experiment, but
answering a question across the adventure required more than a fast first
retrieval. On the same adaptive index, Qwen found the main connection and added
unsupported claims; Haiku stopped with an incomplete answer; Sonnet found the
relevant transition and produced the strongest source-supported answer of the
three. Sonnet took 53 seconds and approximately $0.16 in API usage. These are
observations from one narrative question, alongside a separate factual benchmark.
A subsequent Inkling 2 Mini trial ended at the agent action-schema guard and
produced no final answer. An opt-in schema-guided follow-up then completed
the research loop and recovered the main handoff, with grounding caveats.
Replaying the original 48 factual questions with Inkling then took 2.59 times
Qwen's median query time and showed no factual accuracy gain.
A later Hibana fast replay reduced that median to 3.41 seconds, with 44/47
complete factual answers. Its module-question agent finished in 84.51 seconds
but retained an unsupported transition mechanism.

A subsequent [Unsloth GGUF/DSpark test](experiments/inkling/dspark/COGNEE.md)
used verified no-thinking controls after finding a runtime effort-setting bug.
It completed all 48 factual questions at 10.35 s p50 with the same 44/47 strict
score as native Inkling. Its module agent took 221.57 seconds but missed the
actual Chapter 4 ending and return hook, so the shorter time is not a
quality-matched improvement.

A [paired native thinking test](experiments/inkling/native-thinking/RESULTS.md)
then held the checkpoint, process and factual reader inputs constant. Blocking
thinking cut factual p50 **10.38 → 4.51 s** and the 48-query sum **564.26 →
222.25 s**. Both arms scored 44/47 complete; fully grounded complete answers
fell 44/47 → 43/47. The faster agent chose fewer searches and both fresh
agent answers retained source errors.

The selected larger context was then tested in a separate
[411K FP8 replay](experiments/inkling/fp8-cognee/RESULTS.md). Factual p50/p95
were **6.03/8.41 s**, with **44/47 complete and fully grounded** and a
**300.32 s** query sum. This is slower than the historical Qwen and native
64K BF16 timings. Its six-search agent took **180.03 s** and recovered the
actual Chapter 4 ending and Chapter 5 opening, while retaining a mistaken
belief attribution. Services and stored graph content stayed unchanged.

A [larger BF16 follow-up](experiments/inkling/bf16-cognee/RESULTS.md) then
measured **6.10/8.71 s p50/p95**, **301.10 s** across 48 questions and
**44/47 complete, 43/47 grounded**, essentially matching FP8's timing.
Every factual answer and the full agent trace reproduced the prior 64K BF16
off run. The 182K window also passed a 181,116-token extraction. The selected
411K FP8 service was restored afterward and Clef stayed unchanged.

A [stock-index Inkling comparison](experiments/inkling/stock-cognee/RESULTS.md)
then kept the selected 411K FP8 serving configuration and test controls fixed.
Stock was slower (**10.87 s median versus 6.03 s**) but recovered two more
complete factual answers (**46/47 versus 44/47**). Both had **44/47 fully
grounded complete**. Stock's six-search module agent took **263.71 s** versus
180.03 s, recovered the actual ending earlier, and avoided the adaptive
answer's belief-attribution error, with minor wording caveats.

A separate [LLM Wiki/CROW comparison with existing Spark Clef](experiments/llm-wiki/RESULTS.md)
then tested the full module with native Inkling and the same frozen questions.
Wiki scored **21/47 complete/grounded at 21.74 s median**; disabling reranking
scored **22/47 at 19.33 s**. Both represented the required chapter in 47/48
selections, while generated notes had omitted many requested details. Its
four-search module agent took **300.62 s of research time**, covered the broad
transition and omitted the Nundro escort hook and early foreshadowing. Neither
Inkling nor Clef was restarted. This new wiki was Inkling-ingested, while the
saved Cognee graphs were Qwen-ingested; the result does not isolate architecture.

The work began as a trial of
[Ekimetrics adaptive-chunking](https://github.com/ekimetrics/adaptive-chunking)
on *Phandelver and Below: The Shattered Obelisk*. The
[initial retrieval report](phandelver-and-below-chunking-results.md) covers the
vector-only comparison. The later
[full Cognee report](phandelver-cognee-chunking-results.md) records graph ingestion,
the context defect and repair, recovery, and every subsequent narrative probe.

The full module was expanded into 13 source sections. Stock Cognee produced
78 chunks; the frozen adaptive plan produced 759. Local Qwen on a DGX Spark
performed graph extraction and summarization. Embeddings came from Ollama's
`qwen3-embedding:0.6b`, with 1,024 dimensions and truncation disabled. The
Hibana/Cognee harness captured requests, while vLLM served Qwen. Hibana itself
was not the inference engine for these tests.

The first adaptive integration lost heading context. A fragment describing one
magic item's effect could therefore be attributed to the adjacent item. Passing
the existing heading breadcrumb into extraction, summarization and embeddings
repaired the observed effect-owner error. It did not establish complete graph
correctness: one separate item-rarity description remained wrong.

The original Qwen factual comparison used 48 frozen questions, with one predeclared ambiguous
question excluded from scoring:

| Configuration | Complete correct answers | Recall p50 | Recall p95 |
|---|---:|---:|---:|
| Stock Cognee | 46/47 | 9.87 s | 14.04 s |
| Adaptive, raw chunks | 43/47 | 4.84 s | 6.65 s |
| Adaptive, headings preserved | 45/47 | 4.68 s | 6.16 s |

An Inkling reader follow-up reused the corrected adaptive index and all 48
original questions, with one recall per question and no agent. All reader
messages and raw source sets matched Qwen's exactly. Inkling's p50/p95 were
12.10/18.60 seconds, versus Qwen's 4.68/6.16 seconds. Its query sum was 636.25
seconds versus 226.79 seconds. Strict source review scored 44/47; accepting one
misleading sentence about Ruxithid's beliefs as awkward wording gives 45/47.
The two existing retrieval misses remained, and all six item questions passed.
The [question-test receipt](experiments/adaptive-chunking/cognee/inkling-question-results.json)
records both scoring interpretations and per-question measurements. Inkling
reported 11,497 reasoning tokens despite the lowest requested effort. These
compare the deployed vLLM configurations: Qwen used MTP-2 and prefix caching,
while Inkling used eager execution without speculation beside Clef.

A [retrospective audit](experiments/inkling/dspark/native-reasoning-audit-20261122.json)
confirmed that native Inkling spent **78.35% of factual output tokens** and
**60.52% of its completed agent output tokens** on reasoning. Its exact pinned
vLLM renderer correctly turns effort `none` into zero, unlike the later
Unsloth mapping bug, but zero still permits generated thinking. The earlier
latency comparison therefore includes this extra reasoning workload. The
[subsequent paired native test](experiments/inkling/native-thinking/RESULTS.md)
measured 11,497 → zero reasoning tokens with identical factual reader input,
and a 2.30× median speedup. Historical rows below remain unchanged.

With Qwen, corrected adaptive recall was approximately 2.28 times faster at p95, while
stock retained the higher answer score. Adaptive also incurred substantially
more ingestion work. The original stock and raw-adaptive runs took 64.85 and
230.28 minutes respectively. The corrected run's 244.52 minutes sums successful
document attempts after a VM crash and recovery; it is not a continuous ingestion
benchmark. Concurrency also changed from four to eight. The
[original metrics](experiments/adaptive-chunking/cognee/results.json) and
[corrected metrics](experiments/adaptive-chunking/cognee/context-results.json)
retain those distinctions.

The [Hibana fast factual replay](experiments/adaptive-chunking/cognee/hibana-fast-question-results.json)
used the same 48 reader messages and counted the same input tokens as Qwen
vLLM. It reduced p50/p95 to 3.41/4.32 seconds and the query sum to 168.99
seconds, versus 4.68/6.16 and 226.79 seconds. Complete answers fell from 45/47
to 44/47 because of an added contradiction about Agatha's knowledge. Two
otherwise correct answers added unsupported qualifications, leaving 42/47
fully grounded complete answers under strict review of those extra claims.
Hibana fast changes calibrated weights and speculative acceptance along with
the serving engine, so this does not isolate an engine-only gain.

| Reader on the corrected adaptive index | Complete correct | Query p50 | Query p95 | All 48 queries |
|---|---:|---:|---:|---:|
| Qwen, Hibana fast | 44/47 | 3.41 s | 4.32 s | 168.99 s |
| Qwen, pinned vLLM | 45/47 | 4.68 s | 6.16 s | 226.79 s |
| Inkling, vLLM | 44/47 strict; 45/47 lenient | 12.10 s | 18.60 s | 636.25 s |
| Inkling, Unsloth DSpark 3, no thinking | 44/47 strict; 45/47 lenient | 10.35 s | 15.13 s | 515.36 s |
| Inkling, paired native control, thinking permitted | 44/47 strict | 10.38 s | 18.78 s | 564.26 s |
| Inkling, paired native, thinking blocked | 44/47 strict | **4.51 s** | **6.42 s** | **222.25 s** |
| Inkling, native 411K FP8 KV, thinking blocked | 44/47 strict | 6.03 s | 8.41 s | 300.32 s |
| Inkling, native 182K BF16 KV, thinking blocked | 44/47 strict | 6.10 s | 8.71 s | 301.10 s |

The paired native rows use matched 4,096-token caps and separate cache
namespaces on the same running vLLM process. None of the 96 answers truncated.
The off arm abstained on q22 rather than giving an unsupported partial answer,
but added an unsupported causal claim on q30. Its fully grounded complete
score is therefore **43/47**, versus the control's **44/47**. This fresh off
result is close to historical Qwen timing; Qwen was not rerun alongside it.

The FP8 replay's actual factual reader bodies matched the paired BF16 off
control except for the cache namespace. Both counted 204,946 input tokens;
outputs were 3,069 versus 3,071 tokens with zero reasoning. Upstream request
time rose 157.44 → 237.53 s, while non-upstream query time stayed close at
64.81 versus 62.79 s. The FP8 configuration retained 44/47 complete answers
and avoided the prior q30 extra causal claim, yielding 44/47 fully grounded.
One run per configuration, different device activity, and changes to cache
allocation/context as well as precision prevent attributing the increase to
FP8 alone. The [public metrics](experiments/inkling/fp8-cognee/results-20261122.json)
preserve all source/request checks.

The 182K BF16 replay retained the same 13 GiB fixed budget and ordinary cache
allocator as the 411K FP8 service, with automatic fitting producing the smaller
BF16 window. Its request bodies and answer texts matched old 64K BF16 exactly,
including the q30 unsupported extra claim. Input/output totals stayed at
204,946/3,071 tokens; upstream time was 239.21 s. Similar timing in both larger
configurations leaves the allocator, budget, nominal context and shared-host
conditions unresolved rather than identifying FP8 alone as the cause.

The subsequent Inkling stock-index replay gives a direct chunking comparison
under the selected 411K FP8 configuration, with generated thinking blocked:

| Index with native Inkling | Complete / fully grounded complete | Query p50 | Query p95 | All 48 queries |
|---|---:|---:|---:|---:|
| Stock Cognee, 78 chunks | **46/47 / 44/47** | 10.87 s | 14.72 s | 514.61 s |
| Corrected adaptive, 759 chunks | 44/47 / 44/47 | **6.03 s** | **8.41 s** | **300.32 s** |

Stock supplied 844,840 input tokens versus 204,946, a **4.12×** difference.
It recovered Ruxithid's belief and clairvoyance answers; the flood timing
remained a partial answer. Extra unsupported hovering-device and treat-demand
claims reduced its fully grounded complete count to the same 44/47 as adaptive.
All 48 stock reader messages matched the original Qwen stock run, while model,
generation controls and question hashes matched the adaptive FP8 replay.
The learned graphs/summaries differ between indexes. Runs occurred at different
times, with an intervening BF16 test and FP8 restart; no service was restarted
for the stock replay. [Protocol, source review and metrics](experiments/inkling/stock-cognee/RESULTS.md)
retain these limits.

We then asked a question outside the frozen factual benchmark:

> How does the first part of Phandelver and Below: The Shattered Obelisk,
> culminating in Wave Echo Cave, feed into the second part of the adventure?

A one-pass adaptive answer at `top_k=15` took 27.76 seconds but invented a
Forge of Spells/Obelisk connection. Its seven raw source passages contained
no introduction or Chapter 5 material. Increasing the budget to `top_k=60`
took 36.90 seconds and recovered much of the real transition, while still
leaving Chapter 5 gaps. This supported the user's challenge that the harness
had stopped searching too early. The observed prompt contained 4,215 tokens;
there was no fixed 4,200-token context limit. Here, `top_k` limits graph
relationships rather than tokens or equal volumes of source text.

An earlier continuation wrapper expanded the raw-source union and improved
the answer, but planned multiple searches together and eventually used a
reviewer-directed validation query. Its adaptive run took 182.00 seconds and
101,702 model tokens including the seed query. Those measurements describe
that procedure, rather than an agent independently choosing each next question.
The [narrative probe receipt](experiments/adaptive-chunking/cognee/narrative-probe-results.json)
preserves the one-pass, wider-budget and continuation results.

Hibana fast's later [one-pass answer](experiments/adaptive-chunking/cognee/hibana-fast-onepass-results.json)
took 14.87 seconds with identical initial reader messages, but still invented
the transition through Nezznar, the Forge and deeper Drow Vault content. Its
[five-search agent](experiments/adaptive-chunking/cognee/hibana-fast-sequential-results.json)
finished in 84.51 seconds and 60,389 tokens. It recovered the chapter structure
and broad psionic-goblin continuation, but treated the cave's lack of shards
as a discovery prompting the party's next move. The full retrieved passage
explicitly restricted what characters could learn; the agent lost that
GM-versus-player distinction and never retrieved the actual Chapter 4 ending.
It completed faster than Qwen while retaining grounding failures.

The next experiment implemented the user's proposed sequence: ask one question,
read the response, identify remaining gaps, and choose exactly one next question.
Both the Cognee reader and the research agent used the selected model. Each
decision received earlier observations and the accumulated raw source passages;
generated tool answers were explicitly marked unverified. The agent could
search again or finish with citations and stated uncertainty.

Qwen, Haiku 5.5 and Sonnet 5.5 started fresh on isolated copies of the same
corrected adaptive index. The original question, research policy, `top_k=15`,
eight-search maximum and session isolation were unchanged. Their first reader
messages and seven initial raw chunks were identical. Follow-up questions were
chosen independently, without prior probe answers or reviewer intervention.
The stored graph and summaries remained from Qwen ingestion in all three runs.

| Model | Searches | Model calls | Client time | Total model tokens | Estimated API charge |
|---|---:|---:|---:|---:|---:|
| Qwen, local DGX | 5 | 10 | 143.41 s | 61,432 | Local compute, unpriced |
| Haiku 5.5 | 2 | 5 | 28.72 s | 32,051 | $0.0045723 |
| Sonnet 5.5 | 3 | 6 | 52.98 s | 61,632 | $0.161656 |
| Qwen, Hibana fast follow-up | 5 | 10 | 84.51 s | 60,389 | Local compute, unpriced |
| Inkling, schema-guided follow-up | 7 | 14 | 499.32 s | 104,978 | Local compute, unpriced |
| Inkling, Unsloth DSpark 3, no thinking | 7 | 15 | 221.57 s | 59,598 | Local compute, unpriced |
| Inkling, paired native control, thinking permitted | 8 | 18 | 579.03 s | 151,965 | Local compute, unpriced |
| Inkling, paired native, thinking blocked | 5 | 11 | 164.69 s | 57,729 | Local compute, unpriced |
| Inkling, native 411K FP8 KV, thinking blocked | 6 | 12 | 180.03 s | 82,682 | Local compute, unpriced |
| Inkling, native 182K BF16 KV, thinking blocked | 5 | 11 | 170.57 s | 57,729 | Local compute, unpriced |
| Inkling, native 411K FP8 KV, **stock Cognee**, thinking blocked | 6 | 13 | 263.71 s | 294,447 | Local compute, unpriced |

Client times include Cognee worker startup, retrieval, reader generation,
agent decisions and protocol corrections. They exclude prior ingestion,
storage cloning, earlier failed attempts and subsequent source review.
These are individual observations, without repeated-run latency statistics.
The DGX server was shared, and runs occurred at different times.

The paired native agents supported the broad level-five Phandalin/goblin
bridge, but neither retrieved the actual mine-profit/Nundro-homecoming ending
or Chapter 5's first paragraph. The thinking-permitted answer reversed an
optional campaign-ending condition; the off answer denied first-half Obelisk
foreshadowing despite a retrieved cave goblin scouting for shards. The off
agent rewrote a finish after protocol feedback without searching again, leaving
three searches unused. Its 3.52× shorter client time also reflects less search
work, and is not a quality-matched speed gain. [Paired metrics and source review](experiments/inkling/native-thinking/results-20261122.json)
preserve the separate assessments. DSpark was restored and Clef stayed running.

The later 411K FP8 agent retrieved Chapter 5's actual opening on search 4 and
Chapter 4's mine-profit/Nundro-homecoming handoff on search 6, then finished
without protocol corrections. It avoided the Forge/Obelisk causal invention
and the corpse-note claim, but assigned the gods belief to the mind flayer
fanatics when the cited passage assigns it to Ruxithid. Its main handoff is
more complete than the prior native off run, with a remaining source error.
The 180.03 s client time includes more search work than either that run or
historical Qwen and is not a quality-matched decoder comparison.

The 182K BF16 agent reproduced the 64K BF16 off run's full request/output
trace, final answer and 17-source union, except for cache salts. Its 170.57 s
client time was 3.6% higher for identical work. The same incomplete exact
handoff, unsupported denial of first-half Obelisk foreshadowing, and finish-gap
retry remain. The measured answer was preserved; the source review was reused
after exact matching. The separate near-limit extraction passed after the
measured benchmark and is not included in that time.

The stock-index Inkling agent found Chapter 5's opening on search 2 and the
actual Chapter 4 ending on search 3, then continued checking the goblin/shard
connection and later transition. Its final answer covered the mine-profit
award, Nundro homecoming, optional stopping point and genuine first-half
foreshadowing, without the adaptive answer's mistaken belief attribution.
Substantive claims were supported by the raw-source union under Codex review;
the scouting paraphrase and adjacent citation specificity retain minor
caveats. It used 3.56× as many total model tokens and took 46.5% longer than
adaptive, with six searches in both runs and one duplicate-query correction.
This is a different research trajectory and quality outcome, not an equal-work
decoder comparison. [Stock replay details](experiments/inkling/stock-cognee/RESULTS.md).

The Unsloth follow-up applied nested effort zero and blocked the checkpoint's
thinking-channel token; all 63 factual/agent responses contained zero thinking
text. It retained the same questions, index, research policy and action schema,
with explicit 4,096-token reader caps that did not truncate any answers. The
agent repeatedly reformulated transition queries and retrieved only 12 raw
passages, versus native Inkling's 31. After a finish guard rejected a declared
gap, it cleared the gap list and returned the same answer without another
search. Its supported broad chapter overview omitted the actual mine-profit
and Nundro-homecoming handoff. The [separate report](experiments/inkling/dspark/COGNEE.md)
and [audited metrics](experiments/inkling/dspark/cognee-results-20261122.json)
preserve the incomplete answer assessment, compatibility diagnosis and earlier
stalled attempts. Runtime, quantization and generation policy all changed.

Qwen recovered the return to Phandalin and Chapter 5 investigation, but its
final answer invented a note on the goblin corpse explaining the shard hunt.
The actual passage is GM narration. It also claimed that the earlier bandits
were pawns or distractions in the mind flayer conspiracy, a relationship the
sources did not establish. The agent declared no remaining gaps despite those
errors. Its [receipt](experiments/adaptive-chunking/cognee/sequential-results.json)
includes the decisions and post-run source review.

Haiku avoided those inventions but stopped after two searches with six still
available. Its only follow-up added the introduction's two-part summary. It
correctly described the chapter and level structure, then admitted it had not
verified the narrative handoff. A protocol correction removed its contradictory
gap list without causing another search. Its speed therefore describes a
cautious incomplete answer. The
[Haiku receipt](experiments/adaptive-chunking/cognee/haiku-sequential-results.json)
records that stopping failure and the missing evidence.

Sonnet's second search retrieved the actual Chapter 4 ending: the mine-profit
share, possible escort of Nundro home, strange-goblin clues, and return to
Phandalin amid new crimes. Its third search retrieved the Chapter 5 investigation,
Sawplee background, return-to-town scene and later shard hunt. The final answer
connected those passages and treated the Spider/mine arc as a self-contained
story with an optional continuation. It avoided the invented note, villain
hierarchy and Forge/Obelisk mechanism. No protocol correction was needed. The
[Sonnet receipt](experiments/adaptive-chunking/cognee/sonnet-sequential-results.json)
contains its queries, usage and source assessment.

Inkling 2 Mini was then installed locally from the original mixed-precision
checkpoint in a pinned native vLLM container on Spark 2. Basic text and JSON
requests passed. The same sequential policy completed two searches but failed
at its second decision: the model placed its proposed next query inside an
`args` array and omitted required top-level control fields. Three bounded
attempts failed this contract, so the run ended without a final answer after
137.94 seconds and six model calls. It consumed 30,429 tokens, including
1,946 reasoning tokens despite requesting the lowest effort. This evaluates
compliance with the harness's JSON action schema; final narrative quality and
native tool calling remain unassessed. Its elapsed time is cost until failure,
not successful-answer latency. The
[Inkling receipt](experiments/adaptive-chunking/cognee/inkling-sequential-results.json)
and [installation/client guide](experiments/inkling/README.md) record the result
and endpoint. The model remains available; existing Qwen and Clef services
were healthy after the test.

A schema-guided follow-up then resolved Inkling's action-format failure. The
endpoint was given the required JSON schema through `response_format`, with
the same research instructions and validator. Replaying the exact failed
decision prompt passed after changing only that field. A fresh full run
finished autonomously after seven searches and fourteen calls, taking 499.32
seconds and 104,978 tokens with zero protocol corrections. It recovered the
actual Chapter 4 ending and the main Chapter 5 bridge. Source review retained
caveats about broad geography and ambiguous GM-note wording. The
[schema-guided receipt](experiments/adaptive-chunking/cognee/inkling-schema-sequential-results.json)
records the successful format workaround separately from the original failure;
the [client guide](experiments/inkling/README.md#schema-guided-action-workaround)
shows the opt-in flag. It does not establish a speed improvement at equal work
or remove the need to review factual grounding.

The supported narrative is that the first half builds the party's relationship
with Phandalin and resolves the lost-mine plot. Early psionic-goblin clues
foreshadow another threat. Returning to town brings the party into the goblin
crimes and investigation of Chapter 5, which leads through Ruxithid to the
mind flayer fanatics and shard hunt. This is a narrative handoff between plots;
the sources do not establish that Nezznar's defeat causes the supernatural plot
or that the Forge powers the obelisk.

Sonnet's answer still has limits. It does not explicitly state the restriction
on what characters can learn from the Wave Echo Cave corpse; that sentence was
outside its retrieved fragments. It also lacked the exact Chapter 5 sentence
establishing background activity. Its answer did not turn that GM information
into a discoverable note or claim the plots had the unsupported causal link.
The quality judgments here are Codex's post-run source review, rather than
blind human GM adjudication.

The Qwen agent also ran against the stock index. It repeated the same follow-up
despite protocol corrections and stopped after two searches without a final
answer. Its 65.70 seconds and 146,757 tokens are recorded as cost until stall
in the [paired-agent receipt](experiments/adaptive-chunking/cognee/sequential-results.json).
That run does not supply a successful stock-agent latency comparison.

The cost table reports measured completed runs. Two preserved Haiku compatibility
attempts consumed another 63,287 tokens and approximately $0.0086615: one exposed
Markdown-fenced JSON handling, and another exposed a parse-error path that
bypassed bounded retries. After those repairs, a fresh run used the unchanged
research policy. All Haiku attempts together cost an estimated $0.0132338.
Sonnet completed on its first attempt. API estimates use captured native usage
and [Anthropic's published rates](https://platform.claude.com/docs/en/about-claude/pricing);
they exclude local retrieval and embeddings and are not billing statements.
Qwen's electricity and hardware costs were not priced.

Provider settings also differ. Qwen used temperature 0, seed 604 and disabled
thinking. Haiku used provider-default sampling with disabled thinking. Sonnet
used its supported `between_tools` setting to suppress up-front thinking; all
six responses reported zero thinking tokens. No prompt-cache tokens were
recorded for either cloud model. Tokenizers differ, so token counts are not
equal units of text or compute. These controls are documented for
[Haiku](https://platform.claude.com/docs/en/models/haiku-5-5/overview) and
[Sonnet](https://platform.claude.com/docs/en/models/sonnet-5-5/overview).
Cloud readers had a 4,096-token output ceiling; agent decisions used 2,800.
Qwen reader requests used server defaults. No measured response was truncated.

The audits checked actual model requests against retrieved source text, rather
than relying only on citation IDs. Captures verify that each follow-up followed
the preceding decision and that each decision received the latest answer and
source union. Original artifacts were preserved. These checks establish what
the models saw; semantic accuracy still required source review.

The results support three distinct findings: corrected adaptive chunks provided
faster short factual recall with a slightly lower score than stock; wider or
continued retrieval could recover evidence missed by the initial narrative
query; and model-dependent search and stopping behavior changed whether that
evidence produced a useful answer. Sonnet was the strongest result on this
question. The experiment does not isolate chunking from chunk size and context
volume, or reader quality from agent policy, because both generation roles
changed together. The narrative probes do not alter the frozen factual scores.

Reproduction instructions are in the
[Cognee runbook](experiments/adaptive-chunking/cognee/README.md). The
[sequential runner](experiments/adaptive-chunking/cognee/sequential.py),
[Anthropic capture adapter](experiments/adaptive-chunking/cognee/anthropic_capture.py)
and [source-context audit](experiments/adaptive-chunking/cognee/audit_context.py)
record the protocol and request provenance. Full answers, source text, prompts
and traces remain in ignored local `raw/` storage; the linked JSON receipts
contain the shareable metrics and assessments. The cloud requests included
retrieved passages after the user's explicit approval. API key contents were
kept out of saved artifacts.
