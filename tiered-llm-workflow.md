# Tiered LLM workflow — when to use which model, and how they compose

**Operational model** for running AI work across three LLM tiers without
paying for any one of them more than necessary. The pattern is:
**generate-once with the best model, index-and-query-forever with cheap
local ones.**

**Authors:** Kostadis + Claude.

## The three tiers

| tier | model class | location | cost per call | strength | weakness |
| --- | --- | --- | --- | --- | --- |
| **1** | Frontier API (Opus / Sonnet / GPT-4) | Anthropic / OpenAI | $1–20 per task | novel synthesis, multi-step reasoning, structured prose | sends data off-network, slow per call, billed per token |
| **2** | Spark vLLM, mid-size (Qwen2.5-14B-AWQ, qwen2.5-32B, llama3.3-70B) | local Spark | ~$0 (GPU time) | high-quality batch work, large context | slow throughput, locks the GPU |
| **3** | Spark vLLM, small fast (Gemma-2/3 in 2–9B range) | local Spark | ~$0 (GPU time, low) | fast lookup, iteration, interactive QA | weaker semantic reasoning, shorter effective context |

The key insight: **these tiers don't compete; they compose.** Each one is
the right answer for a different question, and the output of an
expensive tier-1 call becomes cheap forever once captured into the
local knowledge base.

## The mempalace bridge

The thing that makes the three tiers work together is mempalace as a
**persistence layer**:

```
┌────────────────┐                                 ┌────────────────┐
│  Tier 1: Opus  │  ← one-shot, ~$5-20             │  Tier 3: Gemma │
│                │                                  │                │
│  Architecture  │                                  │  Indexing,     │
│  spec          │                                  │  lookup,       │
│  Design doc    │     ┌─────────────────┐         │  query-time    │
│  Code review   │ ──► │   mempalace     │ ◄──     │  reasoning     │
│  Long prose    │     │   (drawers +    │         │                │
│                │     │   LLM closets)  │         │                │
└────────────────┘     └─────────────────┘         └────────────────┘
                              ▲
                              │
                       ┌──────────────┐
                       │ Tier 2: Qwen │
                       │              │
                       │ Batch index  │
                       │ regen, fine- │
                       │ tune source  │
                       │ data prep    │
                       └──────────────┘
```

A single $10 Opus call to generate the canonical architecture doc funds
*essentially unlimited* lookups against that doc thereafter, because
the costly synthesis happens once and the cheap recall happens forever.

## The economic argument in one example

You want a thorough architecture spec for the mempalace codebase, then
you want to be able to ask questions about it interactively for the
next year.

**Bad workflow** — use Opus for everything:
- 1 Opus call to generate the spec: ~$10.
- 200 Opus follow-up calls over the year ("how does parallelism work
  in convo_miner?"): $1–3 each = $200–600.
- **Total: $210–610.** Plus every question sends some content off-network.

**Good workflow** — Opus authors, mempalace indexes, Gemma queries:
- 1 Opus call to generate the spec: ~$10.
- mempalace mine + Gemma closet regen: ~$0 (Spark GPU time, ~10 min).
- 200 Gemma queries over the year: ~$0.
- **Total: $10.** Nothing leaves the network past the initial authoring.

The factor-of-50-ish cost reduction is real, and the speed difference
on the query side is actually *in your favor* with the local setup —
Opus has ~1-3s latency per call; Gemma on Spark is sub-second.

## The decision framework

For any LLM task you're considering, ask: **what kind of work is this?**

### Synthesis-class tasks → Tier 1

A synthesis-class task is one where the LLM has to produce something
*novel*, with structure that didn't exist in the inputs. The output
contains insights / decisions / framings that weren't in any single
source.

Examples:
- Writing a multi-page design doc from a verbal description.
- Generating an exhaustive architecture spec for a codebase.
- Doing a complex code review where you need an opinion on whether the
  design is right.
- Producing structured plans (like every spec doc in `~/src/dgx/`).

**Tell:** you cannot easily verify the answer mechanically. You need an
opinionated synthesis from the model, not a fact lookup. **Use Opus.**
Capture the output into mempalace or a doc tree.

### Lookup-class tasks → Tier 3

A lookup-class task is one where the LLM is retrieving or summarising
content that *already exists somewhere in the knowledge base*. The
model isn't producing novel insight — it's just helping you find or
phrase something.

Examples:
- "What does the closet_llm function do?" (Answer is in the code +
  docstrings; the model just retrieves and phrases.)
- "Where is the prefix-cache optimisation discussed?" (Answer is in
  `~/src/dgx/closet-llm-prefix-cache.md`; the model just retrieves it.)
- "Summarise the four PRs we shipped in this session." (Answer is in
  git log; the model formats it.)
- Generating per-source closet sentences from existing drawer content
  (literally just paraphrasing what's already in the file).

**Tell:** the answer is mechanically verifiable against the knowledge
base. **Use Gemma.** Speed and zero-cost let you query freely.

### Batch-class tasks → Tier 2

A batch-class task is one where you're doing a *quality-sensitive*
transformation across many inputs, where each individual call is
synthesis-shaped but the volume rules out tier 1 spend.

Examples:
- Generating closets for 900 source files (synthesis per file, but
  $900-9000 of Opus calls is way out of proportion to the value).
- Fine-tune training-data generation (e.g. having qwen2.5-32B generate
  synthetic examples for training a 7B model).
- Translating a codebase's comments to another language.
- Producing summaries for a year's worth of session transcripts.

**Tell:** each individual call is "you'd use Tier 1 if it were one-shot,
but you have 1000 of them." **Use Qwen2.5-14B-AWQ** or whatever Spark
LLM gives the right quality/wallclock tradeoff.

### Boundary case: fast iteration on tier-2 work

A common pattern: you're iterating on a prompt or format that will
*eventually* go to a tier-2 batch run, but right now you're testing
variants. The tier-2 model's slow per-call latency makes iteration
painful.

**Pattern:** use Tier 3 (Gemma) for iteration, switch to Tier 2 (Qwen)
for the final production run. Same prompt, different endpoint. Don't
optimise on the slow model.

Concrete example from this session: the closet_llm prompt-format
change (PR #14). We could have tested 4-5 sentence-format variants in
20 minutes on Gemma, converged on the best one, *then* spent 87
minutes on the Qwen production regen. Instead we spent 87 minutes on
the first guess, found it suboptimal, and spent another 87 minutes on
the fix. Net: ~3 hours instead of ~2.

## Spark setup that supports the tiered model

`current-setup.md` already documents two always-on vLLM containers
(vllm-embed on 8000, vllm-chat on 8001 with Qwen). To add tier 3
cleanly:

1. **vllm-gemma on port 8002** — always running, low GPU-utilization
   cap. Fast lookups, interactive QA, fast iteration.

2. **Keep vllm-chat (Qwen) on 8001** — used for batch work that
   demands quality (closet regen, fine-tune data prep).

3. **Anthropic API** — pulled in for one-shot synthesis, never running
   locally. Use the Claude API via `mempalace.llm_client.AnthropicProvider`
   for programmatic access, or via Claude Code for interactive work.

VRAM budget on the Spark (128 GB unified):
- vllm-embed: ~6 GB cap (already configured at 0.05)
- vllm-chat: ~64 GB cap (already configured at 0.5)
- vllm-gemma: ~6-15 GB cap (suggest 0.10 for a 9B Gemma)
- OS + caches: ~25-50 GB free

Plenty of room. All three can run concurrently.

## What this means for daily work

Three things change once the tiered pattern is in place:

**1. Generation budget thinking flips.**

You stop optimising "did I write this prompt well enough to make Opus
do it on the first try?" — that's the wrong frame because Opus calls
should be deliberate one-shots. You start optimising "how do I capture
this Opus output so I never have to re-ask?"

**2. Knowledge accumulates instead of evaporating.**

Every Opus call's output goes through mempalace ingest →
LLM-regenerated closets → search-indexed. The next time you need
something from that doc, Gemma fetches it. The expensive call becomes
durable.

**3. Iteration loops get faster.**

The bottleneck in most AI work is "how fast can I try the next
variation." Tier 3 cuts that from minutes to seconds. You converge on
better prompts/formats/structures because failure is cheap.

## Concrete workflow recipes

### Recipe: write a new architecture spec, then query it

```bash
# Step 1 (one-shot, ~$5-20): Opus generates the spec.
# In Claude Code: "write a thorough spec at ~/docs/architecture-v1.md
#                  covering [the codebase], based on the source trees."
# Output: ~/docs/architecture-v1.md — 8-12 pages of structured prose.

# Step 2 (~10 min, $0): mempalace ingests + indexes via Gemma.
mempalace mine ~/docs/ --workers 8 --wing docs
MEMPALACE_LLM_PROVIDER=openai-compat \
MEMPALACE_LLM_MODEL=google/gemma-2-9b-it \
MEMPALACE_LLM_ENDPOINT=http://192.168.1.147:8002/v1 \
python -m mempalace.closet_llm --palace ~/.mempalace/palaces/chat

# Step 3 (sub-second per query, $0): query freely.
mempalace search "how does authentication flow work"
mempalace search "where is the retry logic"
mempalace search "what assumptions does the parallel pipeline make"
```

### Recipe: iterate on a closet_llm prompt change

```bash
# Iterate on a 16-source bench against Gemma — ~30s per regen.
MEMPALACE_WORKERS=8 \
MEMPALACE_LLM_PROVIDER=openai-compat \
MEMPALACE_LLM_MODEL=google/gemma-2-9b-it \
MEMPALACE_LLM_ENDPOINT=http://192.168.1.147:8002/v1 \
python -m mempalace.closet_llm --palace ~/.mempalace/palaces/test --sample 16
# inspect closet content
# tweak prompt
# repeat

# Once happy with the prompt, swap endpoint to Qwen for the production regen.
MEMPALACE_LLM_MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ \
MEMPALACE_LLM_ENDPOINT=http://192.168.1.147:8001/v1 \
python -m mempalace.closet_llm --palace ~/.mempalace/palaces/campaign-dev
```

### Recipe: complex code review

```bash
# Tier 1: Opus reviews the change. Captures the review as a doc.
# In Claude Code: "review this PR diff. Output a review doc to
#                  /tmp/review-pr14.md covering correctness, safety,
#                  and architectural concerns."

# Tier 3: future lookups via Gemma.
mempalace mine /tmp/review-pr14.md --wing reviews
# Now searchable: "what did Opus think about the closet line format?"
```

### Recipe: fine-tune on D&D session summaries

(Covered in detail in `~/src/dgx/finetune-qwen-on-dnd-plan.md`.) The
tier mapping there is:

- Tier 2 (Qwen2.5-14B or 32B) **generates synthetic training data** if
  needed, since you have only 30-100 real handwritten summaries and
  more would help.
- Tier 2 hardware **runs the actual fine-tune** (QLoRA on Qwen2.5-14B
  base).
- Tier 3 (Gemma or the fine-tuned model) **serves the result** for
  interactive use after training.
- Tier 1 (Opus) is **only** used during the *evaluation* phase if you
  want a "judge" model to compare fine-tune output against gold
  summaries. Optional.

## What about privacy and the local-first ethos

The whole mempalace CLAUDE.md design principle is "your data never
leaves your machine by default." Tier 1 breaks that — Opus calls
necessarily send content to Anthropic.

The tiered pattern *honors* that principle by being deliberate:

- **Source material that's already public** (open-source code, public
  documentation, anything you wrote with the expectation of sharing) →
  fine to send to Tier 1.
- **Source material that's private** (personal journals, business
  internals, anything subject to confidentiality) → never goes to
  Tier 1. Use Tier 2/3 exclusively.
- **Generated material** (the architecture spec Opus writes, the
  Opus review) → fine to keep locally even though its origin was
  Tier 1, because the input you sent was non-sensitive in the first
  place.

The local indexing layer (Tier 3 + mempalace) means you can *think
out loud* against private content all day without sending anything
anywhere. The Tier 1 calls become deliberate, occasional, scoped to
content you've decided is OK to share.

## Open questions

1. **Where does the Anthropic SDK call live in code?** Mempalace already
   has `mempalace.llm_client.AnthropicProvider`. The pattern would be
   to add a small CLI / helper that drives "Opus generates a doc;
   mempalace ingests it" as a one-liner. Out of scope for now.

2. **When is fine-tuning the right answer vs. tiered retrieval?** They
   solve different problems. Fine-tuning learns *style* (cheap to
   query, can't change content without retraining). Retrieval grounds
   in *content* (queries are slower than fine-tuned generation but
   content can change without retraining). For "answer questions
   about the architecture" → retrieval. For "write in my voice" →
   fine-tuning. Often you want both.

3. **What's the right Gemma variant for tier 3?** Open per the
   gemma-vs-qwen-ab experiment in `~/src/dgx/gemma-vs-qwen-ab.md`. The
   answer depends on whether 4B or 9B Gemma is good enough for your
   actual queries.

4. **Is there a Tier 0 — fine-tuned-on-your-style-Gemma?** Maybe. Once
   the fine-tune workflow from `finetune-qwen-on-dnd-plan.md` is
   working for one domain, applying it to a tier-3 model gives you a
   model that's *both* fast *and* personalized. The interesting
   question is whether a fine-tuned 4B Gemma on Kostadis-style D&D
   prose beats Qwen2.5-14B-stock on the same task. Probably yes for
   stylistic transfer, probably no for novel-synthesis tasks. Worth
   benchmarking.

## How to pick this up

This isn't a single-task spec like the others. It's an operational
framework. The natural starting point is:

> "We have a tiered-LLM workflow doc at `~/src/dgx/tiered-llm-workflow.md`.
> Set up vllm-gemma on port 8002 per the doc's hardware section, and
> verify all three tiers (Anthropic API, vllm-chat on Qwen, vllm-gemma)
> are usable from mempalace via the existing `MEMPALACE_LLM_*` config."

That's the prerequisite for adopting the rest of the framework. After
that, the practice is incremental: catch yourself reaching for the
wrong tier, switch.
