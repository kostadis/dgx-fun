# Phandelver in Cognee: adaptive chunking and sequential retrieval

Experiment completed November 21, 2026.

Adaptive chunking made short factual queries faster in this experiment, but
answering a question across the adventure required more than a fast first
retrieval. On the same adaptive index, Qwen found the main connection and added
unsupported claims; Haiku stopped with an incomplete answer; Sonnet found the
relevant transition and produced the strongest source-supported answer of the
three. Sonnet took 53 seconds and approximately $0.16 in API usage. These are
observations from one narrative question, alongside a separate factual benchmark.

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

The factual comparison used 48 frozen questions, with one predeclared ambiguous
question excluded from scoring:

| Configuration | Complete correct answers | Recall p50 | Recall p95 |
|---|---:|---:|---:|
| Stock Cognee | 46/47 | 9.87 s | 14.04 s |
| Adaptive, raw chunks | 43/47 | 4.84 s | 6.65 s |
| Adaptive, headings preserved | 45/47 | 4.68 s | 6.16 s |

Corrected adaptive recall was approximately 2.28 times faster at p95, while
stock retained the higher answer score. Adaptive also incurred substantially
more ingestion work. The original stock and raw-adaptive runs took 64.85 and
230.28 minutes respectively. The corrected run's 244.52 minutes sums successful
document attempts after a VM crash and recovery; it is not a continuous ingestion
benchmark. Concurrency also changed from four to eight. The
[original metrics](experiments/adaptive-chunking/cognee/results.json) and
[corrected metrics](experiments/adaptive-chunking/cognee/context-results.json)
retain those distinctions.

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

Client times include Cognee worker startup, retrieval, reader generation,
agent decisions and protocol corrections. They exclude prior ingestion,
storage cloning, earlier failed attempts and subsequent source review.
These are individual observations, without repeated-run latency statistics.
The DGX server was shared, and runs occurred at different times.

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
