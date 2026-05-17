# Next experiment: Gemma 4 26B MoE (A4B) on the DGX Spark

## Context

The DGX Spark calibration project has so far covered dense models:
Qwen 2.5-14B AWQ (current vllm-chat baseline), Llama 3.3-70B AWQ
(replaced vllm-chat tonight, with speculative decoding successfully
wired in). Both are dense architectures, both bandwidth-bound on the
Spark's 273 GB/s, both single-expert per token.

The next experiment is **Gemma 4 26B MoE (A4B)** — 26B total weights,
~4B active per token via mixture-of-experts routing, 256K native
context, released 2026-04 (see `~/src/dgx/model-comparisons.md`
§"26B MoE (A4B)"). This is the first non-dense model in the experiment
sequence and the first model with PLE (Per-Layer Embeddings). The
calibration value is in *serving behavior unique to MoE* — expert
routing overhead, decoupling of VRAM cost from active-param compute,
batch-size sensitivity, anything in the vLLM logs that doesn't exist
for dense models.

User's stated goal: **both phases** — observe serving characteristics
first, then run a real workload through it for quality comparison.
The Phase B workloads are the two the user actually uses today:
**CampaignGenerator** (D&D session prep, chat-completion only, no
tool calling — exercises pure narrative/creative language quality)
and **opencode** (agentic coding CLI, requires working tool calling —
exercises function-call reliability and multi-turn agent loops).
Together they cover both halves of the model's quality envelope.

## Decisions locked in

| dimension | choice | rationale |
|---|---|---|
| HF model | `google/gemma-4-26b-a4b-it` | Official Google FP16/BF16 release. ~52 GB weights. |
| Quantization | none (BF16) | Native dtype. Avoids AWQ × MoE × PLE compound risk. |
| Container slot | replace vllm-chat on port 8001 (solo) | ~52 GB FP16 weights need maximum VRAM; llama70b stack stood down for the experiment. vllm-embed (5%) and vllm-gemma (15%) stay up. |
| Spec decode | **off** | MoE × spec decode in vLLM is bleeding-edge. One variable at a time. We learned the spec decode plumbing on llama70b; here we learn MoE. |
| Tool calling | **off in Phase A, on starting Phase B2** | Phase A + Phase B1 (CampaignGenerator) don't need it; keeping it off isolates pure-MoE failure modes. Phase B2 (opencode) requires it — add `--enable-auto-tool-choice` + `--tool-call-parser <gemma>` via a container restart between B1 and B2. Parser identifier needs verification (see Known risks). |
| `--max-model-len` | 32768 (32K) | Conservative starting point. 256K KV cache for a 26B-parameter model is ~130 GB, infeasible. 32K leaves real headroom; can bump in v2 if needed. |
| `--gpu-memory-utilization` | 0.75 (~96 GB cap) | Available: 128 GB − 6 GB embed − 19 GB gemma-2 = ~103 GB. 96 GB cap leaves 7 GB margin. |
| `--dtype` | `bfloat16` (explicit) | Gemma 4 trained in BF16. Don't let vLLM auto-pick FP16. |
| `--trust-remote-code` | yes | Gemma 4 + PLE may ship custom modeling code. |

## Critical files to be modified / created

- **NEW**: `/home/kroussos/src/dgx/spin-up-vllm-gemma4-26b-moe.sh` — new spin-up
  script. Pattern source: `/home/kroussos/src/dgx/spin-up-vllm-llama70b.sh`
  (the "stop vllm-chat, replace it" pattern), not the gemma-2 sidecar
  pattern. Reuse the same structure: env-var overrides, 25-min startup
  budget, stop-existing-then-launch, smoke test loop, error-pattern
  detection for early failure.
- **NEW**: `/home/kroussos/src/dgx/gemma4-26b-moe-observations.md` —
  Markdown notes captured during the run. Phase-A observations (serving
  behavior, tokens/sec, MoE-specific log lines), Phase-B observations
  (workload quality vs Qwen 14B and Llama 70B outputs). Lives next to
  the other experiment docs; not for memory.

No edits to existing scripts. Cleanup is via `bash ~/spin-up-vllm-llama70b-specdecode.sh`
to restore the previous state.

## Functions / utilities to reuse

- **Container-replace pattern** (`spin-up-vllm-llama70b.sh:53-57`) —
  stop-and-remove on name collision. Copy verbatim.
- **Startup-watch loop with error patterns** (`spin-up-vllm-llama70b.sh:83-105`) —
  poll docker logs every 15s, abort early on Error/CUDA error/OOM/Traceback.
  Add MoE-specific error patterns: `expert_parallel`, `MoE`, `routing`.
- **HF cache bind mount** (`-v ~/.cache/huggingface:/root/.cache/huggingface`) —
  same as all three existing scripts. Lets vLLM auto-download the model
  on first launch; no host-side `huggingface-cli` needed.
- **Smoke-test pattern**: `printf` → file → `curl -d @file` (lesson
  from tonight — heredocs and inline JSON break on this user's terminal
  due to paste-wrapping).

## Implementation outline

### Script: `spin-up-vllm-gemma4-26b-moe.sh`

Save the following to `/home/kroussos/src/dgx/spin-up-vllm-gemma4-26b-moe.sh`,
`chmod +x` it, then `scp` to the Spark like the existing scripts.

```bash
#!/usr/bin/env bash
#
# spin-up-vllm-gemma4-26b-moe.sh — replace vllm-chat on the DGX Spark
# with Gemma 4 26B MoE (A4B) at BF16. ~52 GB weights, 256K native
# context (we cap at 32K for KV-cache realism), MoE architecture.
#
# WHAT THIS DOES
#   1. Stop + remove the existing vllm-chat container (Llama 70B + spec
#      decode, or whatever else is sitting in the slot).
#   2. Start a vLLM container on port 8001 serving google/gemma-4-26b-a4b-it
#      in BF16, no quantization, no spec decode, no tool calling (yet).
#   3. Wait for "Application startup complete" — 30-min budget because
#      first run pulls ~52 GB and torch.compile may take longer on the
#      PLE / MoE paths than on dense models.
#   4. Smoke-test /v1/models and a tiny chat completion.
#
# USAGE
#   scp spin-up-vllm-gemma4-26b-moe.sh kostadis@192.168.1.147:~/
#   ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-gemma4-26b-moe.sh'
#
# CONFIGURATION
#   GEMMA_MODEL    HF model id; default google/gemma-4-26b-a4b-it
#   GEMMA_PORT     host port; default 8001 (REPLACES vllm-chat)
#   GPU_UTIL       --gpu-memory-utilization; default 0.75 (~96 GB of 128 GB)
#                  Drop to 0.7 if startup hits OOM.
#   MAX_LEN        --max-model-len; default 32768
#                  256K native is infeasible (~130 GB KV cache at FP16).
#                  Bump to 65536 only if you have a real long-context need
#                  and willing to lose concurrency headroom.
#
# REVERTING TO LLAMA 70B + SPEC DECODE
#   bash ~/spin-up-vllm-llama70b-specdecode.sh
#
set -euo pipefail

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-4-26b-a4b-it}"
GEMMA_PORT="${GEMMA_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.75}"
MAX_LEN="${MAX_LEN:-32768}"
CONTAINER_NAME="vllm-chat"
IMAGE="vllm/vllm-openai:latest"

echo "=== spin-up-vllm-gemma4-26b-moe ==="
echo "  model:    ${GEMMA_MODEL}"
echo "  port:     ${GEMMA_PORT}"
echo "  gpu_util: ${GPU_UTIL}  (~$(awk -v u="${GPU_UTIL}" 'BEGIN{printf "%.0f", u*128}') GB of 128 GB)"
echo "  max_len:  ${MAX_LEN}"
echo ""

# Stop existing vllm-chat (whatever's in the slot).
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "→ stopping existing ${CONTAINER_NAME}..."
    docker stop "${CONTAINER_NAME}" 2>&1 || true
    docker rm "${CONTAINER_NAME}" 2>&1 || true
fi

# GPU healthcheck before allocating ~96 GB.
echo "→ GPU status pre-launch:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | head -3
echo ""

echo "→ starting ${CONTAINER_NAME} with ${GEMMA_MODEL}..."
docker run -d \
    --runtime nvidia --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${GEMMA_PORT}:${GEMMA_PORT}" \
    --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${IMAGE}" \
    "${GEMMA_MODEL}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --host 0.0.0.0 --port "${GEMMA_PORT}"

# Wait for healthy. 30-min budget — bigger weights download + possible
# extra torch.compile passes for the PLE / MoE paths.
echo ""
echo "→ waiting for 'Application startup complete' in container logs..."
echo "  (first run includes ~52 GB BF16 download — be patient)"
echo ""
DEADLINE=$(( $(date +%s) + 1800 ))  # 30 min
while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
    if docker logs "${CONTAINER_NAME}" 2>&1 | grep -q "Application startup complete"; then
        echo "  ✓ ready"
        break
    fi
    if docker logs "${CONTAINER_NAME}" 2>&1 | grep -qE "Error|CUDA error|out of memory|Traceback|404 Client Error|Repository Not Found|unrecognized arguments|expert.*error|routing.*error"; then
        echo ""
        echo "  ✗ container errored during startup. Last 60 log lines:"
        docker logs --tail 60 "${CONTAINER_NAME}" 2>&1
        echo ""
        echo "  Common causes:"
        echo "   - OOM at startup: drop MAX_LEN to 16384, then GPU_UTIL to 0.7."
        echo "   - 'Repository Not Found': verify the model is publicly accessible"
        echo "     and (if gated) HF_TOKEN is set in the container env."
        echo "   - MoE / expert routing errors: vLLM Gemma 4 MoE support may be"
        echo "     incomplete on this image tag. Try a pinned older or newer tag."
        echo "   - PLE / custom modeling errors: --trust-remote-code is already on;"
        echo "     check that the HF repo includes the modeling_*.py files."
        exit 1
    fi
    sleep 15
    elapsed=$(( 1800 - (DEADLINE - $(date +%s)) ))
    last=$(docker logs --tail 1 "${CONTAINER_NAME}" 2>&1 | tr -d '\r' | cut -c1-100)
    echo "  [${elapsed}s] ${last}"
done

if [ "$(date +%s)" -ge "${DEADLINE}" ]; then
    echo ""
    echo "  ✗ 30-minute startup budget exceeded. Last 60 log lines:"
    docker logs --tail 60 "${CONTAINER_NAME}" 2>&1
    exit 1
fi

# Smoke tests.
echo ""
echo "→ smoke-test: GET /v1/models"
curl -sS --max-time 5 "http://localhost:${GEMMA_PORT}/v1/models" | python3 -m json.tool 2>&1 | head -20

echo ""
echo "→ smoke-test: tiny chat completion"
printf '%s' "{\"model\":\"${GEMMA_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply only with the word OK.\"}],\"max_tokens\":10}" > /tmp/req_smoke.json
curl -sS --max-time 60 "http://localhost:${GEMMA_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d @/tmp/req_smoke.json | python3 -m json.tool 2>&1 | head -30

echo ""
echo "=== done ==="
echo ""
echo "Phase A observation commands (see plan file):"
echo "  docker logs vllm-chat 2>&1 | grep -iE 'moe|expert|router|gate|a4b|active.param|per.layer' | head -40"
echo "  docker logs -f vllm-chat   # tail and watch stats logger"
echo ""
echo "To revert: bash ~/spin-up-vllm-llama70b-specdecode.sh"
```

Startup budget: **30 min** (longer than llama70b because we're pulling
~52 GB instead of ~40 GB, and Gemma 4 may need additional `torch.compile`
passes for the PLE path).

### Phase A — Serving behavior verification

Run **after** "Application startup complete." Sequence:

1. **Container check** — `docker ps --filter name=vllm-chat`, confirm
   it's `Up`. Note GPU memory via `nvidia-smi` (expect ~52 GB resident
   for weights; KV cache pre-allocated according to `gpu-memory-utilization`).

2. **`/v1/models` ping** — confirm endpoint up:
   ```bash
   curl -sS http://localhost:8001/v1/models | python3 -m json.tool
   ```

3. **MoE-specific log scan** — these lines tell us the architecture is
   actually being treated as MoE, not silently flattened:
   ```bash
   docker logs vllm-chat 2>&1 | grep -iE 'moe|expert|router|gate|a4b|active.param|per.layer' | head -40
   ```
   Capture verbatim into `gemma4-26b-moe-observations.md` §Phase A.

4. **Single-stream throughput probe** — `temperature=0.0`, 400 tokens,
   identical prompt to the one used in the llama70b experiment for
   apples-to-apples comparison:
   ```bash
   printf '%s' '{"model":"google/gemma-4-26b-a4b-it","messages":[{"role":"user","content":"Write a Python function that implements binary search on a sorted list. Include docstring, type hints, and one example call. Return only the code, no explanation."}],"max_tokens":400,"temperature":0.0}' > /tmp/req_gemma_long.json

   time curl -sS --max-time 600 http://localhost:8001/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d @/tmp/req_gemma_long.json > /tmp/resp_gemma.json
   ```
   Record: `real` wallclock, `completion_tokens`, computed tok/s.

5. **Periodic stats observation** — leave a `docker logs -f vllm-chat`
   tail running for ~5 minutes while light traffic flows. Look for:
   - The `loggers.py:271` line — same as we saw on llama70b. Sustained
     generation throughput is the headline.
   - **Any new metric lines that don't exist for dense models** — vLLM
     0.21.x may surface expert-utilization or routing-imbalance metrics.
     Capture verbatim; these are the MoE-unique signals.
   - KV cache usage at sustained load — MoE typically has dense
     attention, so KV cache shape should look like a 26B dense model,
     not a 4B-active model.

6. **Batched concurrency probe** — send 4 parallel requests, observe
   whether throughput scales sub-linearly (MoE routing overhead) or
   super-linearly (expert parallelism win). 4 because higher numbers
   make it harder to read individual decode rates from the logs.

### Phase B — Workload quality comparison

Only after Phase A completes cleanly. Two sub-workloads, run in order
because B2 requires a config change (tool calling on) that B1 doesn't.

#### Phase B1 — CampaignGenerator (no tool calling required)

CampaignGenerator's session-prep flows hit the DGX endpoint via the
existing `--dgx-endpoint` plumbing (see `spin-up-vllm-llama70b.sh:138-140`
for the env-var contract: `DGX_MODEL=<hf-id> python session_doc.py
... --dgx-endpoint http://192.168.1.147:8001/v1`). Plain chat
completions, no functions, exercises narrative/creative voice
quality — the area where Gemma is historically more verbose and
refusal-prone than Qwen (per `model-comparisons.md` §Instruction-tuning
style).

1. Pick **one representative session_doc generation** the user has
   already run against both Qwen 14B and Llama 70B recently. Same
   campaign, same source VTT, same prompt sequence.

2. Run the same prep against Gemma 4 26B MoE:
   ```bash
   DGX_MODEL=google/gemma-4-26b-a4b-it python session_doc.py ... \
     --dgx-endpoint http://192.168.1.147:8001/v1
   ```

3. Capture in `gemma4-26b-moe-observations.md` §Phase B1:
   - Wallclock end-to-end vs Qwen baseline / Llama 70B baseline
   - Sustained tokens/sec from the vLLM stats logger (different from
     the Phase A code-completion number — narrative prose is a harder
     decode workload, expect lower)
   - **Sample narration output** — 2-3 scene narrations side-by-side
     against the same scenes from Qwen / Llama. The user's qualitative
     verdict (voice consistency, refusal incidents, verbosity) is the
     deliverable.
   - Any prompt that triggered a refusal or over-cautious output —
     Gemma's known weak spot vs Qwen.

#### Phase B2 — opencode (tool calling required)

Requires a container restart with tool-call flags enabled. Sequence:

1. **Determine the tool-call parser.** As of vLLM 0.21.x, parser
   support for Gemma 4 is the unknown — possibilities:
   - `gemma` (if vLLM has a Gemma-specific parser by now)
   - `pythonic` (the parser used for models that emit Python-style
     `<tool_call>` blocks)
   - none yet (Gemma 4 is ~1 month old; community parser may be missing)

   Verify by:
   ```bash
   docker exec vllm-chat vllm serve --help 2>&1 | grep -A2 tool-call-parser
   ```
   That dumps the list of valid parser names for the running vLLM
   version. Pick the right one. If no Gemma-family parser exists,
   skip Phase B2 — file an observation in the doc that "Gemma 4 tool
   calling not yet supported by vLLM 0.21.x" and move on. Not a
   failure of the experiment; a calibration data point.

2. **Edit the spin-up script** to add (right before `--host`):
   ```
       --enable-auto-tool-choice \
       --tool-call-parser <chosen-parser> \
   ```
   Re-run `bash ~/spin-up-vllm-gemma4-26b-moe.sh`. Warm-cache restart,
   ~5 min.

3. **Verify tool calling works** with the same curl probe used on
   llama70b — a `get_weather` function with `tool_choice: "auto"`,
   expect `content: null` + populated `tool_calls`. Identical
   pass/fail criteria.

4. **Point opencode at it.** Same env vars / config we used for the
   llama70b setup:
   ```bash
   export OPENAI_API_BASE=http://192.168.1.147:8001/v1
   export OPENAI_MODEL=google/gemma-4-26b-a4b-it
   opencode
   ```

5. **Run a real opencode task** — pick a small coding task you'd
   normally hand to Claude or Llama-70B-via-opencode (e.g. "add a
   helper function to file X with these specs"). Observe:
   - Does Gemma 4 correctly emit function calls when needed?
   - Does the agent loop converge, or does it loop / hallucinate?
   - Quality of the actual code changes proposed
   - Frequency of malformed JSON in `arguments` (a known weakness
     of smaller models on complex tool schemas)

6. Capture in `gemma4-26b-moe-observations.md` §Phase B2:
   - Tool-call success rate (rough — out of N calls in the session,
     how many were well-formed)
   - Whether opencode could complete the task without manual
     intervention
   - User's verdict: does Gemma 4 MoE feel like a viable opencode
     backend, or does it need to stay on Llama 70B?

**No automated quality score for either sub-phase.** Eyeball
comparison only — the user's judgment is the ground truth, and
quantitative semantic-similarity metrics would conflate "different"
with "worse."

## Verification (end-to-end)

The experiment is "successful" not by hitting a throughput target but
by **producing enough observation to answer**:

1. **Does Gemma 4 26B MoE run cleanly on vLLM 0.21.x on the Spark?**
   Pass = "Application startup complete" + 5+ minutes of traffic with
   no crashes. Fail = any post-startup engine traceback or hang.

2. **Where does it sit on the speed-quality curve vs the dense
   alternatives?** Recorded as a row appended to the comparison table
   in `model-comparisons.md` (per-real-data, replacing the
   "uncertain (~10-20?)" placeholder).

3. **Is MoE serving on vLLM mature enough to be worth using?** A
   subjective call — the answer goes in
   `gemma4-26b-moe-observations.md` §Conclusion and is the artifact
   future-you actually wants when revisiting.

## Known risks

- **vLLM MoE support is described as "still maturing"** in
  `model-comparisons.md` §Open questions. Possible failure modes:
  silent expert-router degradation, AWQ-kernel mismatch (mitigated
  here by going FP16/BF16), `torch.compile` issues with the PLE path.
  If startup fails, the script's error-pattern detection will print
  the last 50 log lines; we debug from there.

- **52 GB FP16 weights + 32K KV cache + workspace** may exceed the
  0.75 GPU-memory-utilization budget despite the math saying it fits.
  vLLM's allocator is sometimes optimistic. If OOM at startup: drop
  `MAX_LEN` to 16384 first, then drop `GPU_UTIL` to 0.7 if still OOM.

- **256K context advertised, 32K usable** — the gap is a real
  calibration finding worth recording, not a defect. The
  256K-token KV cache requirement for a 26B-class model
  (~130 GB at FP16) is structural, not specific to this setup.

- **Backing out** is one command — `bash ~/spin-up-vllm-llama70b-specdecode.sh`
  restores the Llama 70B + spec decode stack on port 8001. Allow
  ~3 minutes for warm-cache restart.

- **Gemma tool-call reliability is a known risk area** per
  `model-comparisons.md` §"Where it loses" — small Gemma instruct
  variants have historically been weaker on structured output / tool
  calling than Qwen. Even with the right parser wired up, expect
  rougher edges than the llama70b + opencode pairing achieved
  tonight. If Phase B2 reveals frequent malformed tool calls, that's
  a real and informative result, not a setup bug.

- **Gemma is more refusal-prone and verbose than Qwen** historically.
  For CampaignGenerator's narrative prompts that lean into morally
  grey content (combat, monster cruelty, NPC manipulation), Gemma 4
  may insert hedges, soften descriptions, or refuse outright. Capture
  any such incidents — they're the main quality-vs-Qwen risk for the
  CampaignGenerator workload specifically.

---

## Runbook — copy-paste command sequence

Linear sequence of commands to execute the plan. Each block assumes
the previous one succeeded.

### Step 1: Save the spin-up script locally

Copy the script content above (the entire `#!/usr/bin/env bash` block)
into `/home/kroussos/src/dgx/spin-up-vllm-gemma4-26b-moe.sh`, then:

```bash
chmod +x /home/kroussos/src/dgx/spin-up-vllm-gemma4-26b-moe.sh
```

### Step 2: Deploy to the Spark and launch

```bash
scp /home/kroussos/src/dgx/spin-up-vllm-gemma4-26b-moe.sh kostadis@192.168.1.147:~/
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-gemma4-26b-moe.sh'
```

Expect ~30 min on first run (52 GB download + torch.compile +
flashinfer autotune).

### Step 3: Phase A — serving observation

On the Spark (via ssh):

```bash
# Container running?
docker ps --filter name=vllm-chat

# MoE-specific log scan — does vLLM treat this as MoE?
docker logs vllm-chat 2>&1 | grep -iE 'moe|expert|router|gate|a4b|active.param|per.layer' | head -40

# GPU residency
nvidia-smi

# Throughput probe — same prompt used on llama70b for apples-to-apples
printf '%s' '{"model":"google/gemma-4-26b-a4b-it","messages":[{"role":"user","content":"Write a Python function that implements binary search on a sorted list. Include docstring, type hints, and one example call. Return only the code, no explanation."}],"max_tokens":400,"temperature":0.0}' > /tmp/req_gemma_long.json

time curl -sS --max-time 600 http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @/tmp/req_gemma_long.json > /tmp/resp_gemma.json

python3 -c "import json; r=json.load(open('/tmp/resp_gemma.json')); print('completion_tokens:', r['usage']['completion_tokens']); print(); print(r['choices'][0]['message']['content'])"

# Stats logger watch — leave running while light traffic flows
docker logs -f --tail 20 vllm-chat
```

Capture `real` time and `completion_tokens` → tok/s. Note any
MoE-unique log lines that don't exist for dense models.

### Step 4: Phase B1 — CampaignGenerator

From your dev machine (the WSL host where CampaignGenerator lives):

```bash
cd ~/src/CampaignGenerator
DGX_MODEL=google/gemma-4-26b-a4b-it python session_doc.py \
  <your-usual-args> \
  --dgx-endpoint http://192.168.1.147:8001/v1
```

(Fill in `<your-usual-args>` with whatever campaign + session VTT you
last ran against Llama 70B, so the comparison is apples-to-apples.)

While it's running, on the Spark:

```bash
docker logs -f --tail 5 vllm-chat
```

Watch for sustained tok/s in the `loggers.py:271` line, any refusal
incidents, any error responses.

### Step 5: Phase B2 — opencode (requires container restart)

#### 5a: Identify the tool-call parser

On the Spark:

```bash
docker exec vllm-chat vllm serve --help 2>&1 | grep -A2 -i tool-call-parser
```

Look for any of `gemma`, `pythonic`, or a Gemma-family entry in the
list. **If none exists, skip Phase B2** — file the observation and
stop. Otherwise, note the parser name.

#### 5b: Restart the container with tool calling enabled

On your dev machine, edit `spin-up-vllm-gemma4-26b-moe.sh` and add two
lines to the `docker run` args, right before `--host`:

```
    --enable-auto-tool-choice \
    --tool-call-parser <chosen-parser-from-5a> \
```

Re-deploy and restart:

```bash
scp /home/kroussos/src/dgx/spin-up-vllm-gemma4-26b-moe.sh kostadis@192.168.1.147:~/
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-gemma4-26b-moe.sh'
```

Warm-cache restart, ~5 min.

#### 5c: Verify tool calling

On the Spark:

```bash
printf '%s' '{"model":"google/gemma-4-26b-a4b-it","messages":[{"role":"user","content":"What is the weather in Paris right now?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"}},"required":["location"]}}}],"tool_choice":"auto","max_tokens":150}' > /tmp/req_tool.json

curl -sS --max-time 60 http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @/tmp/req_tool.json | python3 -m json.tool
```

Look for `content: null` + populated `tool_calls`. If the call comes
back as plain text in `content`, the parser is wrong — try the
fallback `--chat-template` per the script's header note, or accept
that Gemma 4 tool calling isn't ready and file the observation.

#### 5d: Point opencode at Gemma 4

From wherever you run opencode:

```bash
export OPENAI_API_KEY=dummy
export OPENAI_API_BASE=http://192.168.1.147:8001/v1
export OPENAI_MODEL=google/gemma-4-26b-a4b-it
opencode
```

Run a real coding task. Observe.

### Step 6: Capture observations

Save a file `/home/kroussos/src/dgx/gemma4-26b-moe-observations.md`
with the template from the next section. Fill in as you go.

### Step 7: Revert when done

```bash
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-llama70b-specdecode.sh'
```

Restores Llama 70B + spec decode on port 8001. ~5 min warm-cache restart.

---

## Observations template

Save the following to `/home/kroussos/src/dgx/gemma4-26b-moe-observations.md`
as a starting skeleton, then fill in during the run:

```markdown
# Gemma 4 26B MoE (A4B) — observations

Date: <YYYY-MM-DD>
vLLM image tag: <output of `docker inspect vllm-chat | grep Image`>
vLLM version: <output of `docker exec vllm-chat vllm --version`>
Model: google/gemma-4-26b-a4b-it
Flags: --max-model-len 32768 --gpu-memory-utilization 0.75 --dtype bfloat16 --trust-remote-code

## Phase A — serving behavior

### Startup
- Wallclock to "Application startup complete": <Xm Ys>
- Resident VRAM after startup: <from nvidia-smi, GB>
- Any errors / warnings during startup: <verbatim or "none">

### MoE-specific log lines
<paste output of the moe/expert/router/gate grep here, verbatim>

### Throughput probe (code, T=0, 400 tokens)
- `real` wallclock: <X.XXXs>
- completion_tokens: <N>
- Computed tok/s: <N / X.XXX>
- Comparison: Llama 70B AWQ + spec decode hit ~23 tok/s on this same prompt.

### Stats logger observations (5 min of light traffic)
- Sustained generation throughput: <tok/s>
- KV cache usage at typical request: <%>
- Prefix cache hit rate (if any): <%>
- Any metric lines unique to MoE (vs the llama70b reference): <verbatim>

### Concurrency probe (4 parallel requests)
- Per-request tok/s under concurrency: <approx>
- Aggregate tok/s: <approx>
- Sub-linear / linear / super-linear scaling: <which>

## Phase B1 — CampaignGenerator

### Configuration
- Campaign: <name>
- Source: <VTT or whatever input>
- Comparison baseline: Llama 70B output for same input, dated <YYYY-MM-DD>

### Wallclock
- Total session-prep run: <Xm Ys> (vs Llama 70B baseline of <Xm Ys>)
- Sustained tok/s during narrative generation: <from vLLM stats>

### Quality verdict
- Voice consistency vs Llama 70B: <better / comparable / worse>
- Verbosity: <more concise / similar / more verbose>
- Refusal incidents: <count, with prompts>
- Sample side-by-side: <paste 2-3 scene narrations from each model>

## Phase B2 — opencode

### Tool-call parser used
<parser-name-or-"none-supported-skip-phase-B2">

### Tool-call verification curl
- `content` field: <null or "...">
- `tool_calls` populated: <yes / no>

### opencode session
- Task attempted: <brief description>
- Tool calls emitted: <count>
- Well-formed JSON in `arguments`: <count well-formed / total>
- Did the agent loop converge: <yes / no / required intervention>
- Quality of code changes: <verdict>

## Conclusion

### Where does Gemma 4 26B MoE sit on the speed-quality curve?
<one paragraph>

### Is MoE serving on vLLM 0.21.x mature enough to be useful?
<verdict + evidence>

### Should this replace any current production usage?
- vllm-chat (currently Llama 70B): <yes / no / for which workloads>
- vllm-gemma-2 (currently gemma-2-9b-it): <yes / no — possibly the MoE
  replaces the small-Gemma slot, not the big-chat slot>

### Open questions surfaced for next experiment
<bullet list>
```

