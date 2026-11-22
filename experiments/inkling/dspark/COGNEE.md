# Cognee with Unsloth Inkling Mini and DSpark

Factual replay completed November 22, 2026 UTC. The no-thinking configuration
completed all 48 questions with no truncations and scored **44/47** under the
same strict source review as the earlier trials. Query latency improved over
the original native Inkling deployment, while Qwen and Hibana fast remain
faster in the historical measurements. The module agent also finished faster,
but failed to recover the actual Chapter 4-to-5 handoff that native Inkling had
found. The [audited metrics](cognee-results-20261122.json) retain both outcomes.

| Reader | Complete correct | Query p50 | Query p95 | Sum of 48 query times |
|---|---:|---:|---:|---:|
| Qwen, Hibana fast (historical) | 44/47 | 3.41 s | 4.32 s | 168.99 s |
| Qwen, pinned vLLM (historical) | 45/47 | 4.68 s | 6.16 s | 226.79 s |
| Inkling, native vLLM (historical) | 44/47 strict | 12.10 s | 18.60 s | 636.25 s |
| **Inkling, Unsloth GGUF, DSpark 3, no thinking** | **44/47 strict** | **10.35 s** | **15.13 s** | **515.36 s** |

The new median is **14.5% lower**, and the query-time sum **19.0% lower**, than
native Inkling. These compare deployed configurations, not an isolated change
to the decoder or thinking policy: runtime, quantization, prefix caching,
device load and dates also differ. Each factual question ran once. This is
complete-answer latency, including retrieval and prompt processing, rather
than the synthetic decode benchmark's token rate.

Across the factual calls, the server reported **360.98 seconds of prompt
processing** and **76.98 seconds of generation**. Output dropped to 3,015 tokens
from native Inkling's 14,674, while input stayed almost unchanged at 204,970
tokens. Prompt processing is now the main measured model-side cost. Removing
generated thinking does not remove that cost.

## What changed

The endpoint remains `http://192.168.1.121:8004/v1`, model `inkling-2-mini`,
using the pinned Unsloth UD-Q4_K_XL target and Q8_0 DSpark drafter at draft
length three. Context is 65,536 tokens, with one active request and prompt
caching disabled. Clef shares Spark 2. The service was not restarted for this
test; the new controls are applied per request.

The first attempt revealed a compatibility bug. Top-level
`reasoning_effort="none"` makes this runtime erase Inkling's effort argument;
the template then renders effort **0.9**. One factual question generated more
than **14,000 tokens** without completing before the worker was stopped.
A second bounded attempt was preserved when the template diagnostic confirmed
the cause. Neither is counted as a completed benchmark.

The successful configuration supplies effort **0.0 inside
`chat_template_kwargs`**, omits top-level effort, and bans the checkpoint's
`<|content_thinking|>` token, **200008**, through llama.cpp's native logit-bias
extension. Correct effort zero alone still emits thinking. A zero reasoning
budget instead produced repeated empty thinking blocks, so it was not used.
[Verified controls, diagnostics and client example](THINKING.md).

The factual reader now has an explicit 4,096-token output cap. Capped outputs
would be retained as failed generations; all 48 finished naturally. No response
contained thinking text. This suppresses the generated thinking channel;
Inkling's internal fixed layer loops still execute.

## Retrieval and quality

The test uses fresh copies of the same corrected adaptive Cognee index:
**759 chunks, 4,335 graph nodes and 19,700 edges**. Ingestion and stored summaries
remain from Qwen. Embeddings remain Ollama `qwen3-embedding:0.6b` at
`192.168.56.1:11434`, with 1,024 dimensions and truncation disabled. No ingestion
or reindexing was performed.

All **48 actual reader messages matched the Qwen baseline**. The source-input
audit verified **329 contextual chunk appearances**. Each question used a fresh
session, automatic feedback stayed disabled, and the cloned graph counts did
not change.

The same three questions prevent a higher strict score:

- **q04:** the flood mechanism/timing remains a retrieval gap; the answer adds
  an incorrect release origin and omits the actual sequence.
- **q20:** the answer reverses the direction of Ruxithid's mistaken belief.
  Its Insight DC is correct. The same lenient reading used for native Inkling
  would raise either result to 45/47; strict scoring remains 44/47.
- **q22:** the retrieved material does not establish Ruxithid's clairvoyance;
  this run abstains.

The predeclared source-ambiguous q37 remains excluded. The existing q38 ruling
scores the trigger and combat round without treating the purple worm as an
ally. Review was performed by Codex against the frozen source and captured
context, not by blinded human adjudicators.

## Earlier native Inkling was also thinking

A retrospective audit of the saved native vLLM calls confirms substantial
generated reasoning despite `reasoning_effort="none"`:

| Earlier native test | Reasoning output tokens | All output tokens | Reasoning share |
|---|---:|---:|---:|
| 48 factual questions | 11,497 | 14,674 | **78.35%** |
| Completed schema-guided agent | 8,589 | 14,191 | **60.52%** |
| Initial agent failure | 1,946 | 3,447 | 56.45% |

The [audit receipt](native-reasoning-audit-20261122.json) recomputes these counts
from all 68 captured calls. All requests used top-level effort `none` and
requested `enable_thinking:false`. A CPU-only offline check of the exact
retained vLLM image and original checkpoint template confirms that native
vLLM preserves `none` and renders effort **0**. It did **not** have the later
Unsloth mapping bug that renders effort 0.9. Native effort zero still permitted
reasoning generation.

The earlier result should therefore be read as latency for that configured
thinking-enabled deployment. It does not establish the model's no-thinking
latency or isolate intrinsic model speed. Token percentages are not measured
percentages of wall time. A subsequent [paired native replay](../native-thinking/RESULTS.md)
now measures that cost on the same checkpoint and process: factual p50 fell
**10.38 → 4.51 s**, and the 48-query sum **564.26 → 222.25 s**. Complete
answers stayed 44/47, while fully grounded complete answers fell 44/47 → 43/47.
Those new native results remain separate from this GGUF experiment. Both
fresh native agent answers missed the exact handoff and retained source errors.

## Agentic module question

The fresh schema-guided agent used the original Wave
Echo Cave transition question, the same research policy, `top_k=15`, a maximum
of eight searches and a new decision after every observed answer. Both reader
and agent receive the no-thinking controls. Reader outputs are capped at 4,096
tokens; agent actions retain the previous 2,800-token cap. Final timing and
source review are included in the [metrics](cognee-results-20261122.json).

| Inkling configuration | Searches | Model calls | Raw source union | Client time | Model tokens |
|---|---:|---:|---:|---:|---:|
| Native vLLM, schema-guided | 7 | 14 | 31 | 499.32 s | 104,978 |
| Unsloth DSpark 3, no thinking | 7 | 15 | 12 | 221.57 s | 59,598 |

The new research loop took 216.57 seconds, or **3 min 42 sec including worker
startup**. The first reader messages matched native Inkling's exactly. After
that, the agent chose its own queries without reviewer intervention or prior
answers. Source auditing verified all seven reader observations and eight
decision inputs, including the protocol retry. All **63 calls** across the
factual and agent tests finished naturally with zero thinking text.

The agent repeatedly reformulated near-identical Chapter 5 transition queries.
Three consecutive searches added no new raw passages. It eventually recovered
the adventure's two-part overview, part of the Chapter 5 investigation setup,
and Chapter 5's ending. It correctly described the level progression and the
later obelisk-shard race, without inventing a Forge/Obelisk causal link or a
corpse note.

It did **not** retrieve Chapter 4's ending: the mine-profit award, escorting
Nundro home, and the return to Phandalin where the strange goblins' crimes
become the next hook. It also missed Chapter 5's first paragraph, which
explicitly continues from the heroes' Wave Echo Cave adventures. Its final
answer acknowledged that its sources lacked the specific transition and
characterized the handoff as structural/mechanical. That is an incomplete
answer, rather than a recovery of the book's actual transition.

The first finish action still declared this gap. Protocol feedback rejected
that action. The agent then cleared `remaining_gaps` and returned the **same
answer**, without another search, leaving one search unused. Schema compliance
and an empty gap list did not establish completeness.

Native Inkling had recovered the actual ending and return hook, with other
grounding caveats. The new elapsed time is **55.6% lower**, but the answers are
not equal in quality. The experiment does not isolate thinking suppression
from the different quantization and runtime.

## Operations and saved evidence

Clef retained the same container, image and start time throughout the successful
run, and the Inkling server process was not restarted. Both services passed
final health checks; the memory guard did not trip. Hibana remains stopped,
and Spark 1's Qwen service remains healthy. The client worker was bounded to
12 GiB of VM memory.

All **20,174 protected historical/input files** passed their hash check. Full
answers, requests, observations, decisions and source reviews are saved under
ignored `experiments/adaptive-chunking/raw/pabtso/cognee-20261120/`:

- `inkling-dspark-no-thinking-20261122/`: completed factual and agent tests.
- `inkling-thinking-controls-20261122/`: template, budget and token-ban diagnostics.
- `inkling-dspark-20261122/`: preserved unbounded stall.
- `inkling-dspark-bounded-20261122/`: preserved attempt superseded by the effort fix.

Regenerate the public metrics with `cognee_analyze.py --root` pointing to the
completed private run and `--output` to the public JSON path. Review labels
must be present; the exporter verifies protocol, provenance and source-union
checks before writing metrics. Historical comparison receipts remain linked
in the [combined Cognee report](../../../phandelver-cognee-agent-results.md).

AtlasKV results and the original synthetic decode measurements remain separate.
