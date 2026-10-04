# Clef / Clef-flash on spark2 — observations

Append-only experiment log. Deployment state lives in `current-setup.md`.

## 2026-10-02 — first bring-up

**What:** Cloudflare's open decision models (released 2026-10-01, Apache-2.0):
`Cloudflare/clef` (Qwen3.8-27B backbone, 55 GB BF16) and `Cloudflare/clef-flash`
(Qwen3.5-9B, 19 GB BF16). Not generative: one prefill pass, then a ~0.25 GB
"joint schema head" (2 evidence-routing layers + 4 decoder layers, width 1024)
scores every allowed option of every typed question. API = Jev/SystemOne.

**Why spark2:** both boxes ran an identical `qwen3.8-flash-next`, ~97 GB reserved and
~13 GB host available. Clef-flash alone (~19 GB + activations) would not fit
beside it, and lowering util is impossible (76.75 GiB weights). spark2 was the
redundant copy, so it was given up.

**Serving stack — and why not vLLM:** the HF cards show `vllm serve` / SGLang /
`docker model run` snippets. Those are HF's auto-generated "use this model" widgets
(`pipeline_tag: image-text-to-text`); they would load the Qwen backbone as a chat
model and silently drop `joint_head.safetensors`. The real path is the release's
`joint_schema_model.py` on transformers. `clef/server.py` is a thin FastAPI
wrapper around its `systemone()`; nothing is reimplemented.

**Image friction:**
- Base `vllm/vllm-openai:latest` purely for torch 2.11.0+cu130 (the release's tested
  torch, already proven on GB10). transformers 5.8.1 → 5.10.2.
- `flash-linear-attention` alone is now a shell package; the kernels are in
  `fla-core`. Installing only the former imports fine but has no kernels. Install both
  `--no-deps` so pip can't touch torch, and verify by importing
  `fla.ops.gated_delta_rule.chunk_gated_delta_rule`, not `fla`.
- `causal_conv1d` is not installed, so transformers logs "fast path is not available".
  Read the source: that flag only gates the warning; the gated delta rule still uses
  fla's Triton kernel and only the short conv falls back to torch. Open lever: build
  causal-conv1d for sm_121.

**Load:** clef-flash 105-142s, clef 410s (760 / ~1,000 tensors, slow start then fast).
CUDA allocated 17.8 GiB / 68.9 GiB. Host available after both loaded: ~41-45 GB.
First request per model: 11-22s (Triton JIT).

**Correctness (local only):**
- Model-card invoice example: `overdue` 0.97 (flash) / 0.99 (clef), `large` true.
  Flipping the state (paid, $40) flips both answers with ≥0.99 confidence.
- Deterministic: 3 identical repeats, and clef-flash's smoke answer was byte-identical
  across a container restart.
- **NOT DONE: equality vs Workers AI** (`@cf/cloudflare/clef[-flash]`). Needs a
  Cloudflare token. This is the check that would catch a silently wrong head load;
  the sanity checks above make that unlikely but do not rule it out.

**Latency / prefill** (warm, single stream, server-measured, GPU-serialized; filler-log
prompts with one urgent ticket at the end, single `noul` question):

| input tok | clef-flash ms | tok/s | clef ms | tok/s |
|---:|---:|---:|---:|---:|
| 383 | 131.6 | 2,910 | 410.1 | 934 |
| 908 | 255.8 | 3,550 | 820.6 | 1,107 |
| 3,158 | 997.5 | 3,166 | 2,846.0 | 1,110 |
| 6,158 | 1,861.0 | 3,309 | 5,369.5 | 1,147 |
| 12,158 | 3,736.2 | 3,254 | 11,077.8 | 1,098 |

3-question smoke (346 tok): clef-flash ~136 ms, clef ~390 ms. Cloudflare's H200
medians: 38.8 / 209.3 ms. **GB10 is 3.5× slower on flash but only 1.9× on the 27B** —
consistent with a compute-bound prefill workload where the small model is dominated by
fixed per-request overhead (head runs as Python loops per question/option, the torch conv
fallback, no batching) and the big model by raw FLOPs. ~1,100 tok/s × 27B ≈ 60 TFLOP/s
effective BF16.

**Open:** Workers AI equality check; causal-conv1d build; request batching
(`collate_records` supports it, server is one-at-a-time); image inputs untested.

## 2026-10-03 — first real workloads: verdict

Three CampaignGenerator experiments against GM-ruled OOTA records. Full numbers:
`vtt-spell-pass-local-design.md`; CampaignGenerator `docs/design/EntityLevelTyping_proposal.md`
(worktree `~/src/CampaignGenerator-entity-typing`, uncommitted). Scripts: `clef/spell-pass-eval/`,
`clef/ensemble-typing-eval/`.

- **Works — coarse semantic triage.** Spell-pass unknown-token pile (864 labelled tokens):
  clef 27B forwards 37–51 % with 0/84 real names missed; fast qwen missed 5–6, 5× slower.
- **Fails — spelling choice.** Canon-name pick: clef 66 %, 71 confident wrong names; flash 30 %;
  ignores word-level instructions (spoken "Misrim" → "Ilvara Mizzrym"). qwen with thinking: 94.5 %
  of answered, 7 wrong names.
- **Fails — conventions and consistency.** Ensemble entity typing ties a free majority vote (82/92);
  "errors" are GM filing conventions. Per-fact re-typing creates as many dossier splits as it fixes.
  Only useful role: second-opinion flag on the vote (8/10 errors caught, 7 false alarms) — and
  fast qwen / clef-flash do nearly as well.
- **Speed edge is small on GB10:** 0.15–0.6 s vs fast qwen's 2–4 s, which tied or beat it on most tasks.

**Lesson:** a decision model fits when the label is a coarse property readable from meaning; it
fails when the label is a convention, a string-level judgement, or a cross-item consistency
property — most CampaignGenerator decisions are the latter.

**Still open:** Workers AI equality check (needs a Cloudflare token) — the one result that could
overturn the negatives. Decision pending with the user: revert spark2 to `qwen3.8-flash-next`.

## 2026-10-03 (later) — vLLM Semantic Router "Decision 2.0" on spark2; Clef stopped

Same SystemOne API as Clef, served by semantic-router's `vllm-sr-runtime` (native torch engine;
never executes package code). Setup, GB10 memory patch and revert: `current-setup.md` LIVE banner,
`decision2/`, `spin-up-decision2.sh`.

| | Combat pairs (12) | Triage: forwarded at 0 misses / at ≤2 misses | Chatter recognised | Median latency, GB10 |
|---|---|---|---|---|
| clef 27B | 11 | 37–56 % / — | 136/152 | 400 ms |
| clef-flash | 11 | — / 66 % | 128/152 | ~140 ms |
| Decision-2.0 Lux-9B | 11 | 89 % / 54–71 % | 129/152 | 127 ms (card: 18.4 ms) |
| Decision-2.0 Nox-4B | 11 | 96 % / 36–72 % | 118/152 | 80 ms |
| Decision-2.0 Kai-0.6B | 5 | 98 % / 78 % | 7/152 | 16 ms (card: 4.9 ms) |

- **Lux-9B ≈ clef-flash** on our tasks, at the same speed on this box; **weaker than clef 27B** at triage.
  Kai is fast but barely reads the situation (combat 5/12 = FlexAI's blind dice).
- **The speed claims do not transfer to the GB10:** the runtime has no CUDA kernels (custom kernels are AMD
  gfx942 only) and the CUDA accelerator is upstream-unvalidated. Kai CUDA-vs-CPU parity checked here: ≤ 0.005.
- **GB10 friction found:** (1) placement uses `cuda.mem_get_info`, which ignores reclaimable page cache on
  unified memory → patched to MemAvailable; (2) Lux uses ~50 GB (33 GB RSS) for 16 GB of weights — co-hosted
  with Clef the box hit 1 GB available and Lux crashed once. Both are upstreamable findings.
- NPC prototype (`mytools` worktree `flexai-social-situational`) repointed to Lux: 357 ms per suggestion with a
  dossier (Clef 27B: ~1,040 ms), similar stance spread.
- **Nox-4B is the family's sweet spot:** same combat score as Clef/Lux, triage between Lux and clef 27B, 80 ms, ~19 GB RSS.
- **Open:** `batching`/`max_speed` profiles; registering fla Triton kernels for CUDA;
  why Lux's RSS is 2× its weights.

## 2026-10-03 (evening) — fine-tuning Decision-2.0-Nox-4B on the GM's rulings

Scripts: `clef/finetune/` (extract.py caches backbone features via `vllm_sr_runtime` exactly as
`serve` does; train_head.py = stage 1, head only; train_lora.py = stage 2, rank-16 LoRA on MLP +
attention projections + head, gradients through the native backbone). Loss: valid-k cross-entropy
(the release's own loss family) with an L2 pull toward the released head. Cached features reproduce
the server's probabilities (mean |dP| 0.0009). The released HF wrapper refuses `train()`; training
works on the runtime's plain nn.Modules. All splits by session (triage) or entity (typing).

**A. Spell-pass triage** (864 tokens, 9 sessions, 3 folds):

| | AUC | forwarded at 0 held-out misses | at ≤2 misses |
|---|---|---|---|
| released | 0.900 | 82 % | 34 % |
| head-only (lr 1e-4, L2-SP 0.1, 20 ep) | **0.963** | **67 %** | 34 % |
| LoRA r16 + head (2 ep) | 0.915 | 80 % | 53 % |

Weak regularisation overfits badly (train-chosen threshold: 19 held-out misses). The same three
names (`Heel Strike`, `Zoom`, `Pick / Shine`) are missed by every model and every tuning — label
limit, not capacity. Clef 27B untuned still the only zero-miss filter (44–51 % forwarded held-out).

**B. Entity typing** (2,538 facts, 147 entities, 5 folds):

| | per-fact accuracy | split (repair 92 / control 55) | facts to event/thread/date | entity-majority correct |
|---|---|---|---|---|
| released | 87.2 % | 45 / 12 | 70 | 137 / 147 |
| head-only | **91.6 %** | 35 / 11 | 1 | — |
| LoRA r16 | 80.6 % | **18 / 9** | 0 | **120 / 147** |

- Head-only consolidates entities whose text points one way (Glabbagool npc 146→180) but does **not**
  learn the GM's conventions: Zuggtmoy/Juiblex (GM: monster) moved *toward* npc; Yeenoghu (GM: npc)
  contradicts the rest. Sparse, inconsistent conventions are outvoted (1,630/2,538 facts are npc).
- LoRA cuts splits by **collapsing to the majority class**: all 63 monster-gold facts → npc, 140/468
  location → npc. Consistent and wrong is worse than split — a split is visible in review, a wrong
  dossier is not.
- **Finding:** fine-tuning on a few hundred to a few thousand rulings recalibrates the readout (A: AUC
  0.90→0.96) but cannot teach conventions the text does not carry, and backbone adaptation on this
  little data trades accuracy for consistency. Typing once per entity remains the fix.
