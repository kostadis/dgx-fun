# Nemotron 3 Nano 30B A3B — test plan

Drafted 2026-05-18. Mirrors the phase structure of
`gemma4-26b-moe-observations.md` so results compare 1:1 with the
Gemma 4 26B MoE baseline already in that file. Fill in measurements
as you go; promote to `nemotron3-nano-30b-observations.md` after
the workflow phase completes.

## Execution status as of 2026-05-18

| step | status | notes |
|---|---|---|
| 1. Gemma 4 baseline | ✅ done | TTFTs captured to `/tmp/gemma4-prefill.txt` + decode files; data noisy at endpoints, see Step 5 caveats |
| 2. Deploy Nemotron | ✅ done | ~18 min cold start; spin-up script bailed on a false-positive log match but container itself was healthy. Regex patched. |
| 3. KV pool size from logs | ✅ done | 5,769,184 tokens / 34.32 GiB. See `nemotron3-nano-30b-observations.md` §"KV cache budget". |
| 4. Tool-call gate | ✅ **PASS** | Open HF #3 bug does not reproduce on current vLLM image. Single-turn probe only — multi-turn validation still pending in Step 7b. |
| 5. Nemotron synthetic measurements | ⚠️ deferred | `bench-prefill.sh` ran but data too noisy on both Gemma 4 and Nemotron sides (cold-graph JIT effects, single-sample-per-length). Decode tok/s ~23 captured organically from vLLM stats logger (matches Gemma 4). Phase B real-workflow wallclocks will supersede this. |
| 6. Decision point | pending | Need Phase B data first. |
| 7a. CampaignGenerator | pending | Match the campaign/session/scene range Gemma 4 Phase B1 used. |
| 7b. opencode | pending | Step 4 cleared the gate. Add local `nemotron-3-nano-30b` entry to `opencode.json` (don't commit). |
| 8. Decide and update docs | partial | `current-setup.md`, `nemotron3-nano-30b-observations.md`, `model-comparisons.md` reflect Phase A state. Final decision (promote / swap-option / revert) waits on Phase B. |

**Findings so far that landed without running benchmarks:**

1. Nemotron picks **FlashInfer CUTLASS MoE + FlashAttention 2** on
   GB10 — the tuned paths. Gemma 4 was on TRITON fallback for both.
   This is the headline calibration result; quantitative work in
   Phase B will measure the magnitude.
2. KV per-token cost is **~6 KB** (12× cheaper than Gemma 4's
   ~73 KB), confirming the Mamba-hybrid long-context-is-cheap claim.
3. **Reasoning is verbose** even on trivial prompts — 232 reasoning
   tokens to say "OK". Per-tool-call wallclock under reasoning is
   ~5 s. The cost is real and will compound in opencode loops.
4. **`HF_TOKEN` plumbing**: added to all spin-up scripts. They now
   pick the token up from `~/.bashrc` and pass it to the container.
   Won't help the current deploy but will speed up every future cold
   start.

## What we're trying to learn

In priority order:

1. **Does NVIDIA's DGX Spark-tuned recipe materially beat Gemma 4's
   fallback-kernel prefill?** This is the most valuable calibration
   data point — it tells you how much performance has been left on the
   table for every model run on the sm_121 fallback path so far.
2. **Does the reasoning behavior help or hurt your read-heavy workflows
   in practice?** Synthetic decode tok/s won't answer this; only an
   actual CampaignGenerator + opencode session will.
3. **Is tool-call + reasoning broken** (per the open HF discussion #3)
   or has it been fixed in current vLLM? Gates whether opencode is
   viable.
4. **Does the Mamba-hybrid "long context is cheap" claim hold?** At
   256K, does it actually serve `max_num_seqs=8` without OOM, and does
   per-turn latency grow slower than Gemma 4's at long context?

## Decision criteria

After running the plan, the swap decision is one of three:

- **Promote to default chat model** (replace Gemma 4 in
  `current-setup.md` §3 + flip opencode default) if:
  - Step 5 prefill at 32K+ shows ≥1.5× improvement vs Gemma 4
    baseline, AND
  - Step 4 tool-call probe passes, AND
  - Step 7a CampaignGenerator output quality is at-or-above Gemma 4
- **Keep as a swap option** (add a spin-up entry to `current-setup.md`
  §8 but don't change defaults) if:
  - Prefill is better but tool calling is broken, OR
  - Tool calling works but real-workflow quality is worse
- **Revert and move on** if startup is unstable, OOMs, or the smoke
  test never produces clean content

## Time budget

~3-4 hours focused. You can break between Step 5 and Step 6 without
losing methodology integrity — Steps 1-5 must run as a single session
because the matched-pair prefill measurement assumes the same box
state and minimal time between back-to-back measurements.

## Files this plan uses

| file | purpose |
|---|---|
| `spin-up-vllm-nemotron3-nano-30b.sh` | deploy Nemotron to vllm-chat slot |
| `spin-up-vllm-gemma4-26b-moe-longctx.sh` | revert to Gemma 4 (current default) |
| `bench-prefill.sh` | TTFT probe at varying input lengths (Step 1, Step 5) |
| `bench-decode.sh` | tok/s probe single-stream + concurrent (Step 1, Step 5) |
| `test-gemma4-toolcall.sh` | tool-call gate, override `MODEL=...` for Nemotron |

---

## Step 1 — Gemma 4 baseline measurements (no swap, ~15 min)

Current `vllm-chat` is still Gemma 4. Capture the comparison numbers
now so the matched-pair measurement is back-to-back.

```bash
# Sanity: confirm the slot is actually Gemma 4
ssh kostadis@192.168.1.147 'curl -sS http://localhost:8001/v1/models' \
  | python3 -m json.tool

# Prefill — the calibration headline. Default lengths: 1K / 8K / 32K / 64K / 100K.
bash /home/kroussos/src/dgx/bench-prefill.sh google/gemma-4-26b-a4b-it \
  | tee /tmp/gemma4-prefill.txt

# Decode — single-stream
bash /home/kroussos/src/dgx/bench-decode.sh google/gemma-4-26b-a4b-it \
  | tee /tmp/gemma4-decode.txt

# Decode — 4-parallel aggregate
CONCURRENCY=4 bash /home/kroussos/src/dgx/bench-decode.sh \
  google/gemma-4-26b-a4b-it | tee /tmp/gemma4-decode-x4.txt
```

**Record:** TTFT at each input length, single-stream tok/s, aggregate
tok/s @ 4. These are the comparison baselines for Step 5.

---

## Step 2 — Deploy Nemotron (~15-30 min)

```bash
scp /home/kroussos/src/dgx/spin-up-vllm-nemotron3-nano-30b.sh \
    kostadis@192.168.1.147:~/
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-nemotron3-nano-30b.sh'
```

**Watch for:** the script's built-in smoke test must produce non-empty
`content`. It uses `max_tokens=2048` so the reasoning phase has room.

**Gate:** if startup errors out, the script prints common-cause hints.
Most likely failures and fixes:

- `reasoning-parser-plugin` not recognized → `docker pull
  vllm/vllm-openai:latest` to ensure recent enough image
- OOM during KV allocation → re-run with `GPU_UTIL=0.75 bash
  ~/spin-up-vllm-nemotron3-nano-30b.sh`
- `qwen3_coder` parser not recognized → same fix (newer image)

If unrecoverable, jump to Step 8 Path C.

---

## Step 3 — Capture actual KV pool size (~2 min)

This replaces the ~3-6M token estimate with the measured number,
which will land in `current-setup.md` §"VRAM budget" if you promote.

```bash
ssh kostadis@192.168.1.147 'docker logs vllm-chat 2>&1' \
  | grep -iE 'kv.cache|num_gpu_blocks|max.concurrency|mamba|attention.layer'
```

**Record:** exact GB / block-count / token-capacity line from vLLM's
startup output.

---

## Step 4 — Tool-call gate (~10 min)

```bash
ssh kostadis@192.168.1.147 \
  'MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 bash ~/test-gemma4-toolcall.sh'
```

**Pass:** `content` is null/empty, `tool_calls[]` populated, function
name `get_weather`, arguments parse as JSON. The response will also
have a `reasoning_content` field — that's expected.

**Fail (the known HF bug shape):** `tool_calls[]` empty, `content`
contains text describing the function call (parser didn't fire). If
you see this, opencode (Step 7b) is blocked but CampaignGenerator
(Step 7a) can still proceed.

---

## Step 5 — Nemotron synthetic measurements (~15 min)

Same probes as Step 1, pointed at Nemotron. The reason for
back-to-back: same box state, same time-of-day, same neighbour
processes — the only variable is the model.

```bash
# Prefill — note: bench-prefill.sh handles reasoning_content correctly,
# so TTFT here captures "time until first emitted token", which is
# what users feel.
bash /home/kroussos/src/dgx/bench-prefill.sh \
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  | tee /tmp/nemo-prefill.txt

# Decode — single-stream. MAX_TOKENS=2048 so reasoning has room and
# you also get content for inspection. tok/s counts TOTAL generated
# tokens (reasoning + content).
MAX_TOKENS=2048 bash /home/kroussos/src/dgx/bench-decode.sh \
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  | tee /tmp/nemo-decode.txt

# Decode — 4-parallel aggregate
CONCURRENCY=4 MAX_TOKENS=2048 bash /home/kroussos/src/dgx/bench-decode.sh \
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  | tee /tmp/nemo-decode-x4.txt
```

**This is the calibration headline.** Compare TTFT-per-input-length
side by side with `/tmp/gemma4-prefill.txt`. If Nemotron's curve grows
substantially flatter at 32K+, that's the NVIDIA-tuned-kernel +
Mamba-hybrid win quantified.

**Methodology notes:**

- `bench-decode.sh` prints per-request `content_chars` and
  `reasoning_chars` alongside `completion_tokens`. For Nemotron expect
  `reasoning_chars >> content_chars` even on short prompts — that's
  the think-overhead behavior, made visible.
- Default `LENGTHS` for prefill cap at 100K to stay under Gemma 4's
  128K max-model-len. Nemotron supports 256K, so you can run an
  additional probe at 200K against Nemotron only:
  `LENGTHS=200000 bash bench-prefill.sh nvidia/...`

---

## Step 6 — Decision point (5 min)

You now have enough data to know whether to continue. Three branches:

- **Prefill clearly faster AND tool calling passed** → Step 7
- **Prefill faster BUT tool calling broken** → Step 7a only (skip
  opencode); Nemotron stays a swap option, not default
- **Prefill is a wash or worse** → Step 8 Path C (revert + writeup)

---

## Step 7 — Real workflow probes (~90 min, only if proceeding)

### 7a. CampaignGenerator (always, even if tool calling broken)

Run the same scene/session you used for the Gemma 4 Phase B1 baseline
in `gemma4-26b-moe-observations.md` so the quality comparison is
apples-to-apples. Grep that doc first to identify which campaign:

```bash
grep -nE "campaign|session|scene" \
  /home/kroussos/src/dgx/gemma4-26b-moe-observations.md \
  | head -20
```

Then:

```bash
cd ~/src/CampaignGenerator
DGX_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  python session_doc.py \
    --campaign <same-as-baseline> \
    --session <same> \
    --scenes <same> \
    --dgx-endpoint http://192.168.1.147:8001/v1
```

**Record:**

- Wallclock vs the Gemma 4 baseline number
- Total tokens (reasoning + content) — expect content roughly similar,
  total substantially higher
- Subjective voice-fidelity on 2-3 voice-critical scenes
- **Specifically watch for reasoning leaking into narrative output** —
  a known failure mode where reasoning models break the fourth wall
  mid-prose

### 7b. opencode (skip if Step 4 failed)

Edit `~/.config/opencode/opencode.json` locally — add a third model
entry, do NOT commit yet:

```json
"nemotron-3-nano-30b": {
  "id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
  "name": "NVIDIA Nemotron 3 Nano 30B A3B BF16",
  "limit": { "context": 262144, "output": 8192 },
  "tool_call": true,
  "temperature": true
}
```

Set top-level `"model": "dgx/nemotron-3-nano-30b"` and run a small
representative task — e.g. "open ~/src/dgx and explain the difference
between the three Gemma spin-up scripts." This exercises long context,
tool calling, and reasoning-in-an-agentic-loop simultaneously.

**Record:**

- Wallclock vs gut feel for Gemma 4 on similar tasks
- Whether think tokens ever leaked into user-visible output
- Whether tool calls stayed reliable across multiple turns (the open
  HF bug sometimes surfaces only after the first few calls, not on
  the first)
- Whether the reasoning visibly *helped* (caught something Gemma 4
  would miss) or was just latency overhead

---

## Step 8 — Decide and update docs (~30 min)

### Path A — Promote Nemotron to default

Update in a single commit:

- `current-setup.md` §3 (run command, flags, measured behaviour)
- `current-setup.md` §"Ports in use" (model name)
- `current-setup.md` §"VRAM budget" (numbers from Step 3)
- `current-setup.md` §6 (MemPalace + llm_wiki + CampaignGenerator +
  opencode model ids)
- `current-setup.md` §"Snapshot ... as of" date
- `current-setup.md` §8 (add Nemotron to swap commands, keep Gemma 4
  swap as the revert path)
- `opencode.json` — add the entry from Step 7b and flip the default
- Create `nemotron3-nano-30b-observations.md` mirroring the Gemma 4
  doc structure, with Steps 1/5 numbers as primary content and Step 7
  notes as Phase B

### Path B — Keep Gemma 4 default, Nemotron as a swap option

- Add Nemotron to `current-setup.md` §8 swap commands only
- Create `nemotron3-nano-30b-observations.md` documenting why it's
  not the default (which axis lost: tool calling, voice quality,
  latency feel)
- Revert: `ssh kostadis@192.168.1.147 'bash
  ~/spin-up-vllm-gemma4-26b-moe-longctx.sh'`

### Path C — Abandon

- Revert (same command as Path B)
- Add a short entry to `model-comparisons.md` explaining what was
  wrong (most likely candidates: parser plugin incompatible, OOM at
  startup, reasoning leaks unacceptable)
- Leave the spin-up script and this test plan in place — future-you
  may want to retry when vLLM ships a fix

---

## Pre-flight checklist before you start

- [ ] Confirm `vllm-chat` on the Spark is currently serving Gemma 4
      and is responsive: `curl -sS http://192.168.1.147:8001/v1/models`
- [ ] No active opencode / CampaignGenerator / llm_wiki sessions using
      `vllm-chat` (Step 2 will kick them off mid-request)
- [ ] Identify the campaign/session/scenes used for Gemma 4 Phase B1
      so Step 7a is matched-pair
- [ ] At least 3 hours of focused time, or commit to running through
      Step 5 before stopping

---

## What this plan does NOT test (intentional)

- **MemPalace mining workflows.** Embeddings go through `vllm-embed`
  on port 8000, not `vllm-chat`. The chat-palace is in
  known-broken state per `current-setup.md` §6. Out of scope.
- **The FP8 variant.** If BF16 looks promising in Step 5, FP8 is worth
  a follow-up (smaller weights, faster decode on Blackwell FP8 cores),
  but adding it here triples the synthetic-measurement runtime.
- **The 1M context mode.** Requires `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
  and would dominate the test budget. Defer until 256K is validated.
- **Comparisons to Llama 70B + spec decode.** That experiment is in
  the Gemma 4 observations doc already; we're testing whether
  Nemotron changes the prefill ceiling, not redoing baselines.

---

## Related files

- `gemma4-26b-moe-observations.md` — the Gemma 4 baseline this plan
  compares against. Phase A, Phase B1, Phase B2, and the
  "Cross-cutting finding: prefill vs decode" section are the most
  relevant.
- `dgx-spark-calibration-report.md` — earlier prefill-vs-decode
  calibration that established the methodology this plan follows.
- `CLAUDE.md` — the rule on keeping `current-setup.md` honest when
  you complete Step 8 Path A or B.
