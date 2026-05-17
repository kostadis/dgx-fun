# A/B experiment: small fast Gemma vs Qwen2.5-14B for closet_llm

**Goal:** Empirically decide whether a "super fast" Gemma model
(typically 2–9B parameters) produces closets that work as well as
Qwen2.5-14B-AWQ on the four validation queries we care about. Trade
4–6× wallclock for some unknown amount of quality loss; the experiment
tells us how much.

**Status:** not run. This is the spec.

**Authors:** Kostadis + Claude.

## What we already know

The closet_llm pipeline is end-to-end parallel against vLLM chat and
produces prose closets per PR #14. The mempalace search path now has
the right kind of content in the closet collection to compete with
drawer-content cosine matches.

A full-palace regen against Qwen2.5-14B-Instruct-AWQ at workers=8 takes
**~87 min**. Aggregate decode throughput ~50–80 tok/s. Spark's GPU runs
at ~14–16% KV cache, plenty of headroom.

A 4B Gemma should hit **~50–80 tok/s solo decode** and ~400–600 tok/s
aggregate at workers=16. Predicted regen: **15–25 min**. That's the
optimistic case. The question is whether the closets it writes are good
enough to make that speed worth the swap.

## The experiment

### 1. Stand up Gemma on the Spark

A second vllm-chat container on port **8002**. Pick a Gemma variant
(see Model choice below) and adjust `--max-model-len` to whatever the
chosen model supports (Gemma 2 maxes at 8192; Gemma 3 / 4 may differ).

```bash
docker run -d --runtime nvidia --gpus all \
  --name vllm-gemma \
  -p 8002:8002 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  google/gemma-2-9b-it \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.25 \
  --host 0.0.0.0 --port 8002
```

`--gpu-memory-utilization 0.25` because the production vllm-chat
container is already capped at 0.5 and vllm-embed at 0.05; we have
~45% of GPU memory free, which is plenty for a 9B-or-smaller Gemma.

Wait for `Application startup complete.` in `docker logs -f vllm-gemma`,
then smoke-test:

```bash
curl -sS http://192.168.1.147:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-2-9b-it","messages":[{"role":"user","content":"Say only OK"}],"max_tokens":10}'
```

### 2. Build a parallel test palace

Don't pollute the main `campaign-dev` palace. Mine to a fresh palace
and regen its closets via Gemma.

```bash
# fresh palace dir
mkdir -p ~/.mempalace/palaces/test-gemma

# mine the same three trees we have in campaign-dev
MEMPALACE_PALACE_PATH=~/.mempalace/palaces/test-gemma \
  python -m mempalace mine ~/src/CampaignGenerator --workers 8 \
  --wing campaigngenerator
MEMPALACE_PALACE_PATH=~/.mempalace/palaces/test-gemma \
  python -m mempalace mine ~/src/mytools --workers 8 \
  --wing mytools
MEMPALACE_PALACE_PATH=~/.mempalace/palaces/test-gemma \
  python -m mempalace mine ~/src/mempalace --workers 8 \
  --wing mempalace
```

This is the slow part — embeddings on nomic-embed, same ~24 min as
campaign-dev's initial mine. Skippable if you have a snapshot of
campaign-dev sitting around: just `cp -r ~/.mempalace/palaces/campaign-dev
~/.mempalace/palaces/test-gemma`, but be careful, the chromadb files
need to be quiesced first (`docker stop` anything that has the palace
open).

### 3. Regen closets via Gemma

```bash
time MEMPALACE_WORKERS=16 python -m mempalace.closet_llm \
  --palace ~/.mempalace/palaces/test-gemma \
  --endpoint http://192.168.1.147:8002/v1 \
  --model google/gemma-2-9b-it 2>&1 | tee /tmp/gemma_regen.log
```

Workers=16 because Gemma's KV cache footprint is much smaller per
stream than Qwen's. If vLLM logs show `Waiting > 0` more than briefly,
back off to workers=8.

Expected wallclock: **15–25 min** for the 906-source palace if
prediction holds.

### 4. Run the four validation queries on BOTH palaces

The same queries that didn't work at all in the original tag-soup
format and should work post-PR #14 against Qwen:

```bash
for query in \
  "truth of session" \
  "authoritative source" \
  "tiered retrieval pipeline" \
  "black-box integration of external service"
do
  echo "=== $query ==="
  echo "--- Qwen palace (campaign-dev) ---"
  MEMPALACE_PALACE_PATH=~/.mempalace/palaces/campaign-dev \
    python -m mempalace search "$query" 2>&1 | grep -E 'Source:|cosine=' | head -10
  echo "--- Gemma palace (test-gemma) ---"
  MEMPALACE_PALACE_PATH=~/.mempalace/palaces/test-gemma \
    python -m mempalace search "$query" 2>&1 | grep -E 'Source:|cosine=' | head -10
  echo ""
done
```

### 5. Compare

Two simple measures.

**Speed:** wallclock of step 3 vs the Qwen baseline (87 min).

**Quality:** for each of the four queries, does the LLM-tagged target
file appear in top-5? Tabulate:

| query | target file | Qwen rank | Gemma rank |
| --- | --- | --- | --- |
| truth of session | GMASSISTANT_PIPELINE.md | ? | ? |
| authoritative source | GMASSISTANT_PIPELINE.md | ? | ? |
| tiered retrieval pipeline | rpg_retriever.py | ? | ? |
| black-box integration | mempalace_client.py | ? | ? |

If both columns are roughly tied (say, Qwen wins 2/4 and Gemma wins
2/4), the Gemma swap is essentially free — keep the speed. If Qwen
sweeps 4/0, the quality gap is real and you have to pick whether 87
min of regen is worth the search-result fidelity.

Optional eyeball test: pick a single file (e.g. `quote_ledger.py`),
inspect the closet rows from both palaces in chromadb, and judge which
set of sentences "reads like" a useful index entry.

## Model choice

The Spark already has these in Ollama (per `current-setup.md`):
`qwen2.5:14b`, `qwen2.5:32b`, `llama3.3:70b`. **None of them is a small
fast Gemma.** Which one to pull from HF depends on what "super fast"
means for you:

| model | params | est. throughput vs Qwen2.5-14B-AWQ | est. quality |
| --- | ---: | ---: | --- |
| google/gemma-2-2b-it | 2B | ~10× faster | clearly worse on semantics |
| google/gemma-2-9b-it | 9B | ~3× faster | competitive on many tasks |
| google/gemma-3-4b-it | 4B | ~5× faster | (new — verify quality) |
| google/gemma-3-12b-it | 12B | ~1.5× faster | competitive but not the "super fast" play |

Recommended starting point: **google/gemma-2-9b-it**. It's the
sweet-spot for this experiment — fast enough to feel obviously faster
than Qwen, big enough that quality regression isn't a foregone
conclusion. If it loses 0/4 on validation, drop to gemma-2-2b for a
truer "speed" test.

(If "gemma4:e4b" in your existing config refers to a specific model
you've already evaluated and want to use, swap in that HF identifier
and adjust `--max-model-len` accordingly.)

## What the result actually tells you

- **Gemma wins 4/4 + 4–6× faster** → migrate. The closet pipeline pays
  87 min on a regen today; cutting that to 20 min unblocks more
  iteration. Plus you free up Spark VRAM for other experiments.
- **Gemma 3/4 + much faster** → the speed-quality tradeoff favors Gemma
  for development iteration (palace regen during prompt tuning), Qwen
  for the final production regen.
- **Gemma 0–2/4** → the quality gap is real. Keep Qwen as the closet
  generator. The speed advantage is moot if the output isn't useful.

There's no a priori-right answer. Each model is a different point on
the speed-quality curve and your corpus is the only authority on which
point matters.

## What this experiment doesn't cover (deliberately)

* **Embedding model choice.** Different question — the closet sentences
  get embedded via nomic-embed regardless of which LLM wrote them. If
  embeddings are the bottleneck for search quality, this experiment
  won't surface that.
* **Prompt engineering for Gemma specifically.** Gemma might respond
  differently to the prompt we tuned for Qwen. A serious eval would
  iterate prompts per model. We're testing the *same prompt* on both
  to keep the comparison fair, even though that arguably underrates
  Gemma.
* **Inference-cost dollars.** Both run locally on the Spark, so this
  is purely about engineering time / wallclock.

## How to pick this up

Open a new Claude session and start with:

> "We have an A/B spec at `~/src/dgx/gemma-vs-qwen-ab.md`. Pick a Gemma
> model and run the experiment. Report which palace surfaces each of
> the four validation queries' target files in top-5."

Total time: ~30 min for Gemma container setup + first probe, ~25 min
for mining the test palace (parallel with everything else — skip if you
have a campaign-dev snapshot), ~20 min for Gemma closet regen, ~5 min
for the four queries. Sub-1.5-hour experiment end-to-end.
