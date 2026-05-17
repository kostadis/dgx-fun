# Fine-tune Qwen2.5-14B on the D&D campaign corpus — plan

**Status:** not started. Plan written 2026-05-10 to be picked up in a future
session. Single-Spark workflow; runs overnight.

**Authors:** Kostadis + Claude (Sonnet/Opus via Claude Code).

## The goal

Produce a fine-tuned Qwen2.5-14B-Instruct that writes D&D session summaries
**in Kostadis's voice** — the style of the handwritten summaries from earlier
in the campaign, not the dry "accurate but corporate" style of generic LLM
output.

Concretely, the success state: feed the model a Zoom VTT transcript or a
gmassistant.app structured recap of a session, and get back a session summary
that's indistinguishable from one Kostadis would have handwritten. Drops into
the existing `CampaignGenerator/session_doc.py` narration pipeline as a
swap-in replacement for the stock Qwen call.

This is **the first fine-tune** — proving the loop end-to-end. Later fine-tunes
build on the same infrastructure (per-character voice models, GM-prep model,
scene-extraction model — see "Future siblings" at the bottom).

## Why this corpus is unusually well-suited

The pairing already exists. Per `~/src/CampaignGenerator/GMASSISTANT_PIPELINE.md`,
each session has produced:

- A **raw Zoom .vtt** transcript (hours of dialogue, noisy).
- A **gmassistant.app structured recap** (Scenes, NPCs, Memorable Moments —
  accurate but dry).
- A **handwritten summary** — described in-tree as "the gold standard for
  voice and style, but too slow to write every week."

That's a textbook instruction-tuning dataset that nobody had to construct.
For most fine-tune projects, the data prep is paying annotators to create
input/output pairs. Here, years of weekly campaign play already produced them.

## Plan

### 1. Survey the corpus

Walk every campaign tree to inventory:
- Which sessions have handwritten summaries (the target).
- Which of those also have VTTs or recaps (the inputs).
- How many complete pairs exist.

Likely locations to check:
- `~/src/campaigns-test/` — phandalin, out-of-the-abyss, hillsfar, toee
- `~/src/CampaignGenerator/` — pipeline outputs, examples
- Anywhere `*recap*.md`, `*summary*.md`, `session_*.md` lives

Threshold for go/no-go: **30+ complete pairs** for LoRA stylistic transfer to
have a chance, **100+ pairs** for it to work well.

Output of this step: a manifest like
`finetune-data/manifest.json` with `{session_id, vtt_path, recap_path,
summary_path, campaign, year}` per row.

### 2. Format as chatml instruction-tuning data

Convert each pair into a chat-formatted example. Two viable shapes:

**Shape A — recap → summary** (recommended start):
```json
{
  "messages": [
    {"role": "system", "content": "You are writing a session summary for a Dungeons & Dragons campaign. The voice is reflective, focused on character moments, and treats the party as protagonists. Memorable scenes get more space than mechanical play."},
    {"role": "user", "content": "<gmassistant.app structured recap goes here>"},
    {"role": "assistant", "content": "<handwritten summary>"}
  ]
}
```

**Shape B — VTT → summary** — same shape, but the user message is the raw
VTT (or a cleaned VTT). Harder for the model (more noise, more tokens to
synthesise from), but trains a more general "raw input → polished output"
skill.

Start with Shape A. Shape B is a follow-on if Shape A works.

System prompt should include campaign-specific facts: who narrates, who the
party is, canonical names, established tone. Pull from the existing
CampaignGenerator system_prompt.md files.

Token budgeting: Qwen2.5-14B's context is 32K. A long recap can be ~3K
tokens, a long summary ~3K tokens. Comfortably fits with room to spare.
Truncate or filter any pairs that exceed ~28K combined.

### 3. Train/eval split

Hold out **10% — at least 5 examples**, ideally 10–15. Stratify by campaign
so the eval covers the styles you actually use. Pick sessions from across
the timeline, not just early or late ones (style drifts).

### 4. Training run

Use **[Unsloth](https://github.com/unslothai/unsloth)**. Purpose-built for
this scenario (single GPU, mid-size model, small corpus, QLoRA).

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    max_seq_length=8192,        # raise if pairs are longer
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,                        # LoRA rank — start here
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ... SFTTrainer with the dataset ...
# lr = 2e-4, epochs = 2–3, per_device_train_batch_size = 1,
# gradient_accumulation_steps = 8
```

Expected wallclock for 100 examples, 3 epochs on a Spark: **4–8 hours**.
Adapter output: ~100–500 MB.

### 5. Serve

vLLM supports LoRA adapters natively. Two paths:

**A — Add LoRA support to the existing vllm-chat container** (one container,
serves both models):
```bash
# stop and restart vllm-chat with LoRA flags:
docker run -d --runtime nvidia --gpus all --name vllm-chat \
  -p 8001:8001 --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/loras:/loras \
  vllm/vllm-openai:latest \
  Qwen/Qwen2.5-14B-Instruct-AWQ \
  --max-model-len 32768 --gpu-memory-utilization 0.5 \
  --enable-lora --max-loras 1 --max-lora-rank 32 \
  --lora-modules dnd=/loras/dnd-summaries \
  --host 0.0.0.0 --port 8001
```
Then request the LoRA by model name `dnd` instead of
`Qwen/Qwen2.5-14B-Instruct-AWQ`.

**B — Run a second container on port 8002** with the LoRA baked in. Useful
for A/B testing because both endpoints can serve simultaneously.

### 6. Evaluation

The hard part — fine-tuning is empirical. For each held-out eval input:

- Generate a summary with **stock** Qwen2.5-14B-Instruct-AWQ.
- Generate a summary with the **fine-tuned LoRA**.
- Compare both against the **handwritten gold** held-out summary.

Subjective measures (humans only):
- Does the LoRA output sound like Kostadis wrote it? Blind comparison —
  shuffle the labels.
- Does it correctly identify which character did what?
- Is the pacing right (memorable moments get more space than mechanics)?
- Does it slip into generic LLM voice anywhere?

Optional cheap automated measures:
- Perplexity of the gold summary under each model. Fine-tune should beat
  base on the eval set (it had better — that's what was optimized).
- Style similarity via embedding cosine to held-out summaries.

## Hardware plan (single Spark)

This entire workflow runs on the existing Spark (`gx10-46ea`, 192.168.1.147)
without buying anything new. Operational steps:

**The evening setup** (~10 min):
```bash
ssh kostadis@192.168.1.147
# Option A: stop vllm-chat for the night (frees the GPU completely)
docker stop vllm-chat
# Option B: keep vllm-chat running but at lower GPU cap so unsloth fits.
# Requires restart with --gpu-memory-utilization 0.35; ~3 min cost.

# Kick off fine-tune
cd ~/finetune-workspace
python train_dnd_lora.py 2>&1 | tee train.log &
```

**Overnight**: training runs. ~4–8 hours. Sleep. Mempalace hooks won't fire
in this window — they're fine to skip.

**The morning** (~5 min):
```bash
# Check it didn't crash
tail train.log
ls -lh outputs/lora-dnd/

# Restart vllm-chat with LoRA support
docker stop vllm-chat
docker run ... (with --enable-lora flags from above)

# A/B sanity check
curl -X POST http://192.168.1.147:8001/v1/chat/completions ... model=dnd ...
curl -X POST http://192.168.1.147:8001/v1/chat/completions ... model=Qwen/...
```

If keeping vllm-chat coexistent during training (Option B): SMs are shared,
both run ~50% as fast, but neither cares because nobody's using them.

## What can go wrong

1. **Corpus too small.** If the inventory in step 1 turns up <20 complete
   pairs, LoRA won't have enough signal to learn the style. The model will
   either memorise verbatim (mode collapse) or fail to shift voice
   meaningfully. **Mitigation:** include partial pairs (recaps without
   handwritten summaries get used as eval-only inputs, not training data);
   data-augment by splitting long summaries into multiple shorter examples
   keyed off scene boundaries.

2. **Mode collapse / verbatim memorisation.** Symptom: LoRA outputs are
   Frankenstein quilts of phrases from training summaries. Usually means
   too many epochs or learning rate too high. **Mitigation:** drop to 2
   epochs; halve learning rate; raise LoRA rank (more capacity to learn
   general voice rather than memorise specifics).

3. **Catastrophic forgetting.** LoRA doesn't usually hit this — the base
   weights stay frozen — but you can lose general instruction-following if
   the system prompts in training are very narrow. **Mitigation:** include
   ~10% non-D&D instruction-following examples (e.g., from the Alpaca
   dataset) in the training set so the model retains general capability.

4. **Voice transfer ≠ factual accuracy.** A fine-tuned model still
   hallucinates names, NPCs, what happened in a scene. The LoRA changes
   *style*, not *grounding*. **Mitigation:** keep the consistency-check
   pass in `session_doc.py` — it's doing fact-grounding work that voice
   alone can't replace.

5. **vLLM LoRA gotchas.** vLLM's LoRA support has historically been
   finicky with quantised base models (AWQ + LoRA together). If
   `--enable-lora --max-loras 1` fails to load against the AWQ base,
   options: (a) switch base to non-quantised FP16 (uses more VRAM, fits
   on Spark fine), (b) merge the LoRA into a new full-weight model and
   serve that, losing the swap-in flexibility.

## Success criteria (concrete)

After the fine-tune ships, both of these are true:

- **Subjective:** A blind comparison of LoRA-output vs handwritten-gold-output
  for held-out eval sessions is "hard to tell apart" on at least 7 of 10.
- **Practical:** Kostadis runs `session_doc.py` for the next session with
  `--model dnd` and the resulting summary is good enough to use as-is, or
  with minor edits. Stops handwriting weekly summaries.

If only the first is true but not the second, the model is *technically*
good but not yet drop-in. Iterate the system prompt.

If neither is true, the corpus probably wasn't large/clean enough.
Revisit step 1.

## Future siblings (once step 6 is green)

Each of these reuses the entire pipeline — survey, format, train, serve —
swapping only the training data:

- **Per-character voice models.** Mine each PC's dialogue from VTT
  transcripts across all sessions. Fine-tune a small LoRA per character
  (rank 16, even smaller). The narration pipeline picks the right adapter
  per scene. Result: first-person narration that actually sounds like that
  character.

- **GM-prep model.** Train on planning docs, beat structures, NPC reactions
  Kostadis has written over the years. Generates session prep that matches
  the GM style — feeds into `prep.py`.

- **Scene-extraction model.** Train on pairings of VTT → handwritten scene
  summaries. The current scene-extraction pass is the weakest link in the
  pipeline; replacing the generic LLM with a fine-tune trained for *this
  specific transformation* should sharpen it considerably.

- **Voice file generator.** Used by `session_doc.py` for per-character
  narration. The voice-file skill in Claude already produces these — could
  be replaced with a dedicated fine-tune.

After the first one works, each of these is ~4–8 hours of training and
maybe a half-day of data prep. Cheap.

## Open questions to resolve at start

These don't have to be decided now — they'll come up during the data-prep
walk-through. Listed here so future-Kostadis (or future-Claude) doesn't
hit them cold:

1. **Multi-campaign or campaign-specific?** Train one LoRA across all
   campaigns Kostadis has summaries for, or one per campaign? Cross-campaign
   trains general "Kostadis voice"; per-campaign trains style + lore.
   First instinct: train cross-campaign, system prompt provides the
   per-campaign context at inference time.

2. **What goes in the system prompt vs. the LoRA?** Style is in the LoRA;
   campaign facts (NPC names, established events) are in the system prompt.
   Decide where the line is — e.g., is "the party is named the Helmstone
   Pact" a system-prompt fact or a LoRA-learned fact? (Best practice: system
   prompt. LoRA learns style invariants, not corpus-specific facts.)

3. **Chunking strategy for long summaries.** If a single session summary
   is 8K tokens, do we train on the whole thing as one example or split it
   into scene-level chunks each with its own scene-level recap input?
   Affects whether the LoRA learns "summary-level voice" or "scene-level
   voice" — different but both useful.

4. **Eval-set leakage.** Make sure held-out sessions don't appear *anywhere*
   in training data including in the system prompt's "canonical facts"
   block. Easy to leak indirectly.

5. **Where does the workspace live?** Suggested: `~/src/finetune-dnd/` —
   keeps it adjacent to CampaignGenerator and mempalace without polluting
   either.

## How to pick this up

Open a new Claude session in `~/src/` (or wherever you put the workspace),
and start with:

> "We have a plan at `~/src/dgx/finetune-qwen-on-dnd-plan.md`. Let's start
> with step 1: survey the corpus."

The plan walks itself. The first concrete action is the inventory.
