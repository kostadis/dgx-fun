# CampaignGenerator `enhance_summary` on the DGX Spark — calibration test

**Date:** 2026-06-02
**Author:** Kostadis (with Claude Code)
**Box under test:** `spark1` (`192.168.1.147:8001`), `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`, 128K context, TurboQuant KV (`turboquant_k8v4`), vLLM 0.22.0.
**Baseline:** Anthropic `claude-sonnet-4-6`.

A shareable writeup of running CampaignGenerator's Stage-1 recap-enrichment
pass (`enhance_summary.py`) against a local 80B model on the Spark, instead
of the Anthropic API. The goal is **calibration**, not optimization — the
Anthropic path is faster and denser (numbers below). What we wanted to know:
does the local model produce *trustworthy* output for this workload, and
where does the workload actually break the model?

> **TL;DR.** The Spark handled real sessions cleanly and its output passed
> the project's own canon consistency gate with **zero issues**. It is
> ~1.6× slower than Sonnet and ~20% less detail-dense, but factually
> faithful. It only "chokes" on a synthetic 10-session blob (~115K input
> tokens) — and even then it fails fast and cleanly with an HTTP 400, not
> garbage. A *single* real D&D session is only ~12–15K tokens, nowhere near
> the 128K wall.

---

## What `enhance_summary` does

Stage 1 of CampaignGenerator's post-session pipeline. It takes a lossy Zoom
VTT transcript + a human-authored `gm-assist.md` recap, stuffs the full VTT
into a cached system prefix, and asks the model to enrich the gm-assist's
section structure (Summary / Memorable Moments / Scenes / NPCs / Locations /
Items / Spells) with detail and verbatim moments the recap missed. Output is
human-reviewed before Stage 2.

It routes to the Spark via the built-in OpenAI-compat adapter in
`campaignlib.make_client()`:

```bash
export DGX_ENDPOINT=http://192.168.1.147:8001   # spark1
export DGX_MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
```

(Unset both to fall back to the Anthropic API.)

> **Note on `spark1`:** throughout this doc the box under test is `spark1`.
> Its `vllm-chat` HTTP endpoint is `192.168.1.147:8001`. The IP appears in
> the commands because `spark1` is an SSH alias, not a DNS name an HTTP
> client can resolve — so the env var needs the literal address.

---

## Test 1 — Quality: Spark vs Sonnet on a real session

Input: `out-of-the-abyss/summaries/20260601/` — a Candlekeep murder-mystery
session (crime scene → interviews). VTT 64,706 chars dialogue + gm-assist
23,006 chars → system prompt 66,375 chars, user 23,242 chars.

| Metric | Spark (Qwen3-Next-80B) | Sonnet 4.6 |
|---|---|---|
| Wall clock | 5:55 | **3:41** |
| Output size | 30,457 chars | 39,466 chars |
| Sections (h1–h4) | 46 | 47 |
| Scene bullets | 57 | **69** (~20% more) |
| Memorable-Moments quotes | **6** | 4 |
| Bold "moment" callouts | 5 | 5 |

**Findings — the differences are enrichment quality, not correctness:**

- Both held the gm-assist section structure 1:1. Neither dropped or
  hallucinated whole sections.
- Sonnet is faster *and* denser: ~20% more granular scene bullets, longer
  Summary paragraphs.
- Sonnet normalized the title (`Chapter 54: The Case of the Missing
  Tickles`); the Spark copied the gm-assist's raw `Chapter 54 The case of
  the missing tickles.` verbatim.
- Sonnet caught a 5th spell (`Cure Disease`) the Spark missed — its one
  extra section.
- Garbled VTT line handling: Sonnet rendered Daz's line as `"It would
  appear… body is dead."` (ellipsis, truer to the raw VTT, which cuts off);
  the Spark used the gm-assist's completed `"…his body is dead."`. Both
  faithful to *a* human-verified source.
- The Spark actually surfaced **more** verbatim quotes in Memorable Moments
  (6 vs 4).

**Quote-fidelity spot check (the failure mode we worried about most for a
small-active-param model):** 6 quotes checked against the raw VTT. 5 were
exactly verbatim with correct attribution (incl. correctly keeping
`Joe Beda (as Thorin)` as the speaker). The 6th — the gift tag's "your
loving rival" — looked invented at first (the VTT cuts off at "you're
loving."), but the word "rival" was **already in the human-authored
gm-assist**. The model faithfully carried the human text forward rather than
fabricating. **No fabrication detected.**

---

## Test 2 — Consistency: does the Spark output survive the canon gate?

Ran the project's own `check_consistency.py` on the **Spark** output against:
`campaign_state` + `world_state` (auto-loaded) + `docs/party.md` + the
session prep `notes/sessions/candlekeep_pickup_crime_scene_to_interviews.md`.
(Checker model: `claude-sonnet-4-20250514`, 3 context docs.)

> **Result: No issues found.** *"The recap accurately reflects the session
> content and is consistent with all provided context documents… strong
> continuity editing, maintaining consistency across a complex murder
> mystery with multiple suspects, clues, and red herrings."*

This is the result that matters for trusting local output in the pipeline:
the local 80B's enrichment passed the **same canon gate Anthropic output
would**.

---

## Test 3 — Stress: where does it actually choke?

| Input | Dialogue chars | Input tokens (vLLM count) | Result |
|---|---|---|---|
| 20260601 (OOTA) | 64,706 | ~9K | ✅ 5:55 |
| 20260526 (Phandalin, largest real session) | 90,046 | ~12K | ✅ 6:32 |
| **Synthetic: 10 real sessions concatenated** | 849,049 | **114,689** | ❌ HTTP 400 in **1.44s** |

The choke, verbatim:

```
openai.BadRequestError: Error code: 400 - This model's maximum context
length is 131072 tokens. However, you requested 16384 output tokens and
your prompt contains at least 114689 input tokens, for a total of at least
131073 tokens.
```

**Findings:**

1. **Real sessions never come close to the wall.** vLLM counted 114,689
   tokens for 849,049 dialogue chars → **~7.4 chars/token** for VTT content.
   So a single real session is only **~12–15K tokens**. You'd need to
   concatenate ~9 full sessions into one prompt to hit the 128K limit. The
   "stuff the whole VTT in the prefix" design is in no danger from
   single-session prep.
2. **The failure is well-behaved.** Hard 400 at request validation in 1.4s —
   no silent truncation, no wasted prefill compute. And because it's a 400
   (not 5xx/529), `stream_api`'s retry logic correctly did **not** burn its 4
   retries on it.
3. **But the tooling surfaces it as a raw traceback.** `enhance_summary.py`
   has no pre-flight token check, so an over-window input dies with an
   `openai.BadRequestError` stack trace rather than a friendly "input too
   large." Minor, but a real gap for anyone feeding it a multi-session blob.

---

## Two friction points (calibration, not bugs)

- **Throughput, not capacity, is the Spark's limit here.** ~6 min/session at
  ~21 tok/s effective output, vs Sonnet's 3.7 min. Fine for batch/offline
  prep; painful for anything interactive.
- **Prompt cache is silently dead on this path.** `enhance_summary` passes
  `cache_system=True`, but the OpenAI-compat adapter flattens the
  `cache_control` block to plain text and vLLM ignores it — every run pays
  full prefill. Irrelevant for one-shot Stage 1; it would hurt any pipeline
  reusing the VTT prefix across many calls.

---

## Reproduction

From a campaign workspace, with `ANTHROPIC_API_KEY` set and the Spark up.

```bash
# --- Spark run (Test 1, local; spark1) ---
export DGX_ENDPOINT=http://192.168.1.147:8001   # spark1
export DGX_MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
D=~/campaigns/out-of-the-abyss/summaries/20260601
python enhance_summary.py "$D/GMT20260602-005652_Recording.transcript.cleaned.vtt" \
  --gmassist "$D/gm-assist.md" --output "$D/session-summary.spark.md"

# --- Sonnet baseline (Test 1, Anthropic) ---
unset DGX_ENDPOINT DGX_MODEL
python enhance_summary.py "$D/GMT20260602-005652_Recording.transcript.cleaned.vtt" \
  --gmassist "$D/gm-assist.md" --output "$D/session-summary.sonnet.md"

# --- Consistency gate on the Spark output (Test 2) ---
cd ~/campaigns/out-of-the-abyss
python ~/src/CampaignGenerator/check_consistency.py \
  summaries/20260601/session-summary.spark.md \
  --context docs/party.md \
  --context notes/sessions/candlekeep_pickup_crime_scene_to_interviews.md \
  --output summaries/20260601/consistency_report.spark.md

# --- Over-limit stress (Test 3): concat ~10 raw VTTs, run on Spark ---
cat ~/campaigns/Phandalin/summaries/*/GMT*Recording.transcript.vtt > /tmp/overlimit.vtt   # exclude *.cleaned
export DGX_ENDPOINT=http://192.168.1.147:8001   # spark1
export DGX_MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
python enhance_summary.py /tmp/overlimit.vtt \
  --gmassist ~/campaigns/Phandalin/summaries/20260526/gm-assist.md \
  --output /tmp/overlimit.summary.md     # → HTTP 400, max context length, in ~1.4s
```

## Output artifacts (kept)

- `out-of-the-abyss/summaries/20260601/session-summary.spark.md`
- `out-of-the-abyss/summaries/20260601/session-summary.sonnet.md`
- `out-of-the-abyss/summaries/20260601/consistency_report.spark.md`
- `Phandalin/summaries/20260526/session-summary.spark.md`

The synthetic over-limit VTT was a throwaway and was deleted (it would
otherwise pollute "largest VTT" scans of the campaign tree).
