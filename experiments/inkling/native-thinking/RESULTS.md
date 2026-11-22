# Native Inkling: the cost of generated thinking

Completed November 22, 2026 (Pacific). Removing generated thinking from the
same native Inkling checkpoint and running vLLM process reduced median Cognee
factual-query latency from **10.38 to 4.51 seconds**, a **2.30× speedup**.
Complete answers stayed at **44/47**, but fully grounded complete answers fell
from **44/47 to 43/47** because one otherwise correct answer added an
unsupported causal claim. The [audited metrics](results-20261122.json) contain
per-question timings, review labels, request checks and restoration evidence.

## Original 48 factual questions

| Measurement | Minimum effort, thinking permitted | Thinking channel blocked |
|---|---:|---:|
| Query p50 | 10.38 s | **4.51 s** |
| Query p95 | 18.78 s | **6.42 s** |
| Sum of 48 query times | 564.26 s (9 min 24 s) | **222.25 s (3 min 42 s)** |
| Captured upstream model time | 487.56 s | 157.44 s |
| Input tokens | 204,946 | 204,946 |
| All output tokens | 14,674 | 3,071 |
| Reasoning output tokens | 11,497 | **0** |
| Natural completions / truncations | 48 / 0 | 48 / 0 |
| Complete correct answers | 44/47 | 44/47 |
| Fully grounded complete answers | 44/47 | 43/47 |

The summed query time fell **60.6%**, saving **5 min 42 s** over 48 questions.
Median latency fell **56.6%**. Of the 342.01 seconds saved, 330.12 seconds were
inside captured upstream model calls. The actual reader messages and input
token counts matched for every pair; the request bodies differed only in the
thinking-token ban and the cache namespace. This directly supports generated
thinking as a substantial contributor to native Inkling's earlier latency.
These are complete-answer query times, including retrieval and prompt
processing, rather than decode tokens per second.

Both arms retained the known flood retrieval gap (q04) and reversed belief
direction (q20). On q22, blocking thinking changed an unsupported partial
answer into an abstention. On q30, the faster answer supplied the requested
facts but incorrectly linked thought reading as the cause of a severed head's
demand for treats; the source lists independent behaviors. That extra claim
accounts for the lower grounding score. The predeclared ambiguous q37 remains
excluded, and q38 retains the original trigger-and-round rubric.

All 48 thinking-permitted answers and reader messages exactly matched the
previously reviewed native run, so those source-review labels were reused.
Previously reviewed answers were reused for the off arm only when both answer
and reader messages matched exactly; novel answers were reviewed against the
frozen gold and supplied source context. Review was by Codex, without blinded
human adjudication.

## Agentic module question

Each agent started with the original question about Wave Echo Cave's handoff
into the adventure's second half, read each result, and chose the next search.
No previous answer or reviewer-directed query was supplied during the runs.

| Measurement | Thinking permitted | Thinking blocked |
|---|---:|---:|
| Client time, including worker startup | 579.03 s (9 min 39 s) | **164.69 s (2 min 45 s)** |
| Research loop | 574.65 s | 159.42 s |
| Searches / model calls | 8 / 18 | 5 / 11 |
| Accumulated raw passages | 24 | 17 |
| Protocol corrections | 2 | 1 |
| Total model tokens | 151,965 | 57,729 |
| Reasoning output tokens | 9,505 | 0 |
| Fully grounded complete handoff answer | No | No |

The faster agent used fewer searches and model calls. Its 3.52× shorter client
time is therefore not a quality-matched measurement of decoder speed.
Both answers supported the broad level-five return to Phandalin and new
goblin-crime investigation, but neither retrieved the actual Chapter 4 ending
with the mine-profit award and Nundro homecoming, or Chapter 5's first
paragraph. Neither invented the earlier Forge/Obelisk causal link or corpse note.

The thinking-permitted answer reversed the book's optional campaign-ending
condition: replacing the psionic goblins applies when stopping after Chapter
4, not when continuing. The off answer categorically denied first-half
Obelisk setup despite retrieving the cave goblin scouting for shards, and
spent much of its specific detail on Chapter 5-to-6 rather than Chapter 4-to-5.
Both therefore retained substantive source errors.

The off agent initially declared a gap about a direct Wave Echo Cave-to-Chapter
6 scene. After protocol feedback rejected that finish, it cleared the gap and
rewrote its answer without another search, leaving three searches unused.
The on agent also cleared a finish gap without another search, after exhausting
its eight-search allowance; its other correction was a duplicate query.
An empty gap list still did not establish a complete answer.

The historical native agent had recovered the actual ending in 499.32 seconds,
with separate grounding caveats. That remains a historical observation; the
fresh thinking-permitted agent above is the control for this experiment.

## Controls and operations

Both arms used the same retained native container and process, checkpoint
`b85c3cfa615fc413cc9af3379f491b963f3b586a`, image
`sha256:f95362fc89f98d28a429d7047caa9090ccab881e1c4ffd0385f643627211c6de`,
65,536-token context, one active request, eager execution, 0.26 GPU-memory
utilization, prefix caching and no speculative decoding. Both requested effort
`none`, which this native template correctly renders as zero, and
`enable_thinking:false`. Only the off arm added
`bad_words:["<|content_thinking|>"]`, a verified hard ban on token 200008.
The Unsloth effort-mapping bug does not apply to this native renderer.

Separate `cache_salt` values isolated each arm and phase, with the same four
synthetic warmups before each factual test. Prefix caching remained enabled
within phases. Cached-token counts were not reported by the native API and
are recorded as null. Temperature was zero and seed 604. Reader and agent
caps were matched at 4,096 and 2,800 tokens; no response hit a cap.

All phases used fresh copies of the corrected adaptive index: **759 chunks,
4,335 nodes and 19,700 edges**, with Qwen-ingested graph/summaries unchanged.
Embeddings used Ollama `qwen3-embedding:0.6b` at `192.168.56.1:11434`,
1,024 dimensions and truncation disabled. No ingestion was performed. Fresh
session IDs and disabled automatic feedback prevented conversational carryover.
The factual request/context audits and agent observation/source-union audits
passed, as did hashes for **20,504 protected historical/input files**.

The successful native startup took 143.04 seconds, outside the measurements.
A host guard would stop only Inkling below 12 GiB available memory; the minimum
sample was **15.38 GiB** and the guard did not trip. The Cognee worker was
limited to 12 GiB VM memory. An earlier cache-reset-discovery preflight stopped
before measured generation and restored DSpark; it is excluded and preserved.

**DSpark draft three was restored with its original launch command.** The
native container was stopped and retained at the end of the experiment.
Clef's container identity, image and start time stayed unchanged. Final model-discovery checks at **08:52
Pacific** passed for Inkling on port 8004, Clef on 8002 and Spark 1 Qwen on
8001; both Hibana shadow units were inactive. That restored DSpark endpoint
needed its different [per-request no-thinking controls](../dspark/THINKING.md).
The user subsequently selected native vLLM as the running service; its
[current configuration](../NATIVE.md) is separate from these experiment receipts.

There was one observation per factual question and one agent run per arm,
in fixed on-then-off order on a device shared with Clef. There are no
repeated-run confidence intervals. The new native off result is close to the
historical Qwen factual timing, but Qwen was not rerun as a simultaneous control.
The historical native, GGUF/DSpark and AtlasKV results remain separate.

## Saved evidence and reproduction

[Protocol and native client controls](README.md), [controller](run.py) and
[exporter](analyze.py) accompany the public metrics. Full passages, answers,
requests, decisions, reviews and operational receipts remain under ignored
`experiments/adaptive-chunking/raw/pabtso/cognee-20261120/inkling-native-thinking-paired-20261122/`.
The excluded preflight remains in the sibling `inkling-native-thinking-20261122/`.

Regenerate metrics from the reviewed private artifacts:

```bash
python3 experiments/inkling/native-thinking/analyze.py \
  --root experiments/adaptive-chunking/raw/pabtso/cognee-20261120/inkling-native-thinking-paired-20261122 \
  --output experiments/inkling/native-thinking/results-20261122.json
```
