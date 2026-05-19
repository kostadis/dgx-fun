# Current DGX Spark Setup

Snapshot of what's actually running on the Spark as of 2026-05-18. Use
this as a "rebuild from scratch" reference if the box wipes, or as
inventory when debugging.

> **Experimental state warning**: `vllm-chat` is currently serving
> **Nemotron 3 Nano 30B A3B** (BF16), swapped in from Gemma 4 26B MoE
> on 2026-05-18. Phase A serving behaviour validated (see
> `nemotron3-nano-30b-observations.md`); Phase B real-workflow
> validation still pending. **Client configs in §6 below still point
> at Gemma 4** — they will send 400s against the current Nemotron
> deployment until updated, OR you can revert with `bash
> ~/spin-up-vllm-gemma4-26b-moe-longctx.sh` (~10 min warm restart).

> **Drift note for Claude**: if you change anything in this list
> (swap a model, add a flag, replace a container), update this file
> in the same change. See `CLAUDE.md`.

## Hardware

- **DGX Spark** (GB10 — Grace + Blackwell, 128 GB unified memory)
- Hostname: `gx10-46ea`
- LAN IP: `192.168.1.147`
- Memory bandwidth: ~273 GB/s
- Filesystem: EXT4 local

## Ports in use

| port | service | purpose |
|---:|---|---|
| 11434 | Ollama (systemd) | LLM serving (legacy, kept for fallback) |
| 8000 | vllm-embed (docker) | Embeddings — `nomic-embed-text-v1.5` |
| 8001 | vllm-chat (docker) | Chat completions — `Nemotron 3 Nano 30B A3B` BF16, 256K context, reasoning + tool calling on |

## VRAM budget (steady state)

| service | reserved cap | actual model size | notes |
|---|---:|---:|---|
| vllm-embed | ~6 GB (0.05 × 128) | ~600 MB | KV cache fits in cap |
| vllm-chat | ~102 GB (0.80 × 128) | ~60 GB BF16 weights + 34.32 GiB KV cache @ 256K context | KV cache holds **5,769,184 tokens** (vLLM-measured). At `--max-num-seqs 8` × 256K = 2.1M tokens used, 36% pool utilisation. Theoretical headroom: 22.01× concurrent 256K sessions if uncapped. See §3. |
| Ollama (idle) | ~0 | unloads after `OLLAMA_KEEP_ALIVE` | 5 min default |
| Ollama (loaded) | varies | qwen2.5:14b ≈ 14.5 GB, nomic ≈ 600 MB | only when actively serving |

Both vLLM containers stay resident; Ollama unloads on idle (different
design — see `spark-llm-serving-learnings.md`). Combined steady-state
reservation: ~108 GB (the vLLM allocator's view of its own budget).
**Per-token KV cost on Nemotron is ~6 KB vs Gemma 4's ~73 KB** — a
~12× reduction driven by Nemotron having only 6 attention layers (of
52 total) compared to Gemma 4's ~60. Any third vLLM sidecar (e.g. the
optional `vllm-gemma` gemma-2-9b-it container on port 8002) will need
vllm-chat's `--gpu-memory-utilization` dropped before it can fit
cleanly.

**Padding waste**: vLLM logs `Add 1 padding layers, may waste at most
4.35% KV cache memory` for Nemotron — the 6 attention layers get
padded to 7 for alignment. Minor; not actionable.

**CUDA-graph memory deduction**: vLLM 0.21+ deducts CUDA-graph memory
from `--gpu-memory-utilization`. At 0.80 set, effective is 0.7924. To
restore nominal KV size, bump to 0.8076 — but with 64% of the pool
free at the recipe's design point, no point.

---

## 1. Ollama (systemd service)

Installed via the standard `curl https://ollama.com/install.sh | sh`
script. Service runs as user `ollama`, group `ollama`.

### Service config

`/etc/systemd/system/ollama.service`:

```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/home/kostadis/.local/bin/:/usr/local/cuda/bin:/opt/bin/:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"

[Install]
WantedBy=default.target
```

### Override (the actual tuning)

`/etc/systemd/system/ollama.service.d/override.conf`:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_NUM_PARALLEL=8"
```

Apply with `sudo systemctl daemon-reload && sudo systemctl restart ollama`.
Verify env reached the running process:

```bash
sudo cat /proc/$(pgrep -f 'ollama serve')/environ | tr '\0' '\n' | grep OLLAMA_
```

### Models pulled

```bash
ollama list
# nomic-embed-text:latest    274 MB
# qwen2.5:14b                8.99 GB (Q4_K_M GGUF)
```

### Caveats noted

- `OLLAMA_NUM_PARALLEL=8` does **not** apply to `/api/embed` on Ollama
  0.23.2 — that path serializes regardless. Verified by client-side
  concurrency probe (capped at ~1.9× speedup at 8-way concurrency).
- It does affect `/api/chat` and `/api/generate`.

### Status today

Kept around as fallback / for casual model rotation. **Not actively used
by any production workload** since vLLM moved everything off it.
Optional: `sudo systemctl stop ollama` to free its baseline overhead.

---

## 2. vllm-embed (Docker container, port 8000)

Embedding service for MemPalace and anything else that wants
OpenAI-compatible `/v1/embeddings`.

### Run command

```bash
docker run -d --runtime nvidia --gpus all \
  --name vllm-embed \
  -p 8000:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  nomic-ai/nomic-embed-text-v1.5 \
  --trust-remote-code \
  --gpu-memory-utilization 0.05 \
  --host 0.0.0.0 --port 8000
```

### Why these flags

- **Model is positional**, not `--model`. Modern vLLM CLI changed; `--model`
  is deprecated and will be removed.
- **`--trust-remote-code`**: nomic ships custom modeling code in their HF
  repo. Required.
- **`--gpu-memory-utilization 0.05`**: tight cap because the model is tiny
  (~600 MB) and we need to leave room for vllm-chat. **Spec this container
  first** so vllm-chat sees most of the GPU as free at boot.
- **No `--task embed`**: removed in modern vLLM, task is auto-detected.
  Falls back to `--runner pooling` if auto-detect ever fails.

### Smoke test

```bash
curl -sS http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"nomic-ai/nomic-embed-text-v1.5","input":"hello"}'
# expect: {"object":"list","data":[{"object":"embedding","embedding":[...768 floats...],"index":0}],...}
```

### Measured throughput

Batch=1024 single client: **~11,400 tok/s** (vs Ollama's ~300 tok/s for
the same model on the same hardware).

---

## 3. vllm-chat (Docker container, port 8001)

Chat completions + tool calling + reasoning service. Backs llm_wiki,
CampaignGenerator, opencode, future chat clients (see
`desktop-chat-clients.md`), and any code calling
`/v1/chat/completions`.

Currently serving **NVIDIA Nemotron 3 Nano 30B A3B** in BF16.
Architecture is a **hybrid Mamba-Transformer MoE**: 52 total layers
comprising 23 Mamba-2 layers + 23 MoE layers (128 routed experts + 1
shared, 6 active per token) + 6 attention layers with GQA. ~30B total
params / ~3.5B active per token. Native 256K context (1M with
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`).

Slot history: previously served Qwen 2.5 14B AWQ → Llama 3.3 70B AWQ
+ spec-decode → Gemma 4 26B MoE (A4B). Swap-in scripts are
`spin-up-vllm-llama70b.sh`, `spin-up-vllm-llama70b-specdecode.sh`,
`spin-up-vllm-gemma4-26b-moe.sh`, and
`spin-up-vllm-gemma4-26b-moe-longctx.sh`. (Note: `spin-up-vllm-gemma.sh`
is a *different* slot — it spins up the optional gemma-2-9b-it
sidecar on port 8002 as container `vllm-gemma`, not a replacement for
vllm-chat.)

### Run command

Launched via `./spin-up-vllm-nemotron3-nano-30b.sh`. The script now
auto-extracts `HF_TOKEN` from the user's `~/.bashrc` and passes it
into the container (`-e HF_TOKEN=...`) so HF downloads happen at
authenticated rate. Effective command:

```bash
docker run -d --runtime nvidia --gpus all \
  --name vllm-chat \
  -p 8001:8001 \
  --ipc=host \
  -e HF_TOKEN="${HF_TOKEN}" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/vllm-plugins:/plugins:ro \
  vllm/vllm-openai:latest \
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype auto \
  --dtype bfloat16 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin /plugins/nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3 \
  --host 0.0.0.0 --port 8001
```

The custom `nano_v3_reasoning_parser.py` is fetched once from the HF
repo into `~/vllm-plugins/` and mounted into the container at
`/plugins:ro`. The spin-up script handles the wget.

### Why these flags

These follow NVIDIA's [vLLM recipe for DGX Spark / Jetson Thor](https://docs.vllm.ai/projects/recipes/en/latest/NVIDIA/Nemotron-3-Nano-30B-A3B.html).
Unlike Gemma 4, this is a tuned-for-hardware recipe rather than a
generic config we adapted.

- **`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`**: ~30B total /
  ~3.5B active per token. Hybrid Mamba-2 + MoE — only 6 of 52 layers
  are quadratic attention, so long context is structurally cheap
  (KV pool holds 5.77M tokens at the current config).
- **`--max-model-len 262144`** (256K): NVIDIA recipe default for DGX
  Spark. Bumping to 1M requires `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`;
  not enabled here.
- **`--max-num-seqs 8`**: NVIDIA recipe default. Caps concurrent
  sequences at 8 × 256K = 2.1M tokens in the KV pool (36%
  utilisation). Theoretical headroom is 22.01× per vLLM's startup
  report.
- **`--gpu-memory-utilization 0.80`**: ~102 GB cap. ~60 GB weights
  leave 34.32 GiB measured KV pool. Drop to 0.75 if startup OOMs;
  the recipe doesn't specify a util value so this is our pick.
- **`--kv-cache-dtype auto`**: BF16 KV cache (matches the model
  dtype). Flip to `fp8` only if you also swap to the `-FP8` weight
  variant.
- **`--dtype bfloat16`**: native Nemotron 3 Nano dtype.
- **`--trust-remote-code`**: Nemotron ships custom modeling code
  (`configuration_nemotron_h.py` + the hybrid Mamba kernels).
- **`--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder`**:
  NVIDIA reused the Qwen 3 coder tool-call parser for Nemotron 3.
  Tested working — see `test-gemma4-toolcall.sh` (override `MODEL=`).
- **`--reasoning-parser-plugin /plugins/nano_v3_reasoning_parser.py`
  + `--reasoning-parser nano_v3`**: custom parser plugin shipped in
  the HF repo. Strips `<think>...</think>` blocks from `content`
  into a separate `reasoning` field on the response. **Required for
  any client that expects clean `content`** — without this, the
  reasoning text appears inline in the response.
  - **Important field-name quirk**: the parser emits `reasoning`,
    NOT the OpenAI-convention `reasoning_content`. Clients that
    inspect reasoning need to check both field names. Our
    `bench-prefill.sh` and `bench-decode.sh` scripts handle this.

### Known perf characteristics (Spark-specific)

Unlike Gemma 4, Nemotron's startup picks **tuned kernels** on GB10:

- **MoE backend**: `Using FlashInfer CUTLASS Unquantized MoE backend
  out of potential backends: ['FlashInfer TRTLLM', 'FlashInfer
  CUTLASS', 'TRITON', 'BATCHED_TRITON']` — the tuned path. (Gemma 4
  got the TRITON fallback on the same hardware.)
- **Attention backend**: `Using FLASH_ATTN attention backend out of
  potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN',
  'FLEX_ATTENTION']` + `Using FlashAttention version 2`. (Gemma 4
  was forced to `TRITON_ATTN` by mismatched head dims.)
- **Padding waste**: 4.35% of KV cache wasted on 6→7 attention-layer
  padding. Logged at startup, not actionable.
- **Reasoning overhead**: the model emits reasoning tokens before the
  final answer. Smoke test of "Reply with OK" cost 21 prompt tokens
  + 234 completion tokens (232 reasoning + 2 content). A 116:1
  reasoning:content ratio on trivial prompts. Real workloads have
  smaller ratios (reasoning is bounded by problem complexity, not
  output length) but it's a meaningful token-cost multiplier vs
  Gemma 4. Latency per tool-call turn: ~5s wallclock under reasoning.

### Smoke test

```bash
curl -sS http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16","messages":[{"role":"user","content":"Reply only with OK."}],"max_tokens":2048}'
# expect: choices[0].message.content == "\nOK"
#         choices[0].message.reasoning is a non-empty string
#         usage.completion_tokens >> length(content) (mostly reasoning)
```

**Don't use `max_tokens=10`** — the reasoning phase will consume the
budget before any content tokens are emitted. 2048 is the safe floor
for smoke tests.

Tool-calling end-to-end probe (works with reasoning enabled):

```bash
ssh kostadis@192.168.1.147 'MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 bash ~/test-gemma4-toolcall.sh'
```

### Measured behaviour (Phase A only — Phase B pending)

Phase A serving-behaviour measurements captured 2026-05-18 (see
`nemotron3-nano-30b-observations.md` for the full record):

- **Init engine** (post-load): 73.11s total, 11.98s compile. Warm
  restarts will be much faster than Gemma 4's ~10 min — most of
  Nemotron's ~18 min cold-start was the HF download itself.
- **Cold-start wallclock**: ~18 min on first run (download ~60 GB
  weights + 73 s init). With `HF_TOKEN` set, future cold starts
  should be substantially faster.
- **Decode tok/s** (organic, from vLLM stats logger during an
  ad-hoc probe): **~23.4 tok/s** single-stream — basically identical
  to Gemma 4's ~23 tok/s. Decode is bandwidth-bound and dominated by
  active params (3.5B vs 4B — comparable).
- **Tool calling**: PASS with reasoning enabled. The open HF
  discussion #3 about tool-call + reasoning being broken does NOT
  reproduce on the current vLLM image.
- **Prefill calibration synthetic probe was attempted but data was
  too noisy to publish.** Both Gemma 4 baseline and Nemotron runs
  showed non-monotonic curves consistent with first-call CUDA-graph
  JIT effects. Phase B real-workflow comparisons supersede the
  synthetic probe as the authoritative measurement.

**Pending Phase B validation:**
- CampaignGenerator session run (matched-pair vs Gemma 4 baseline)
- opencode session under reasoning + tool-call loop
- Subjective voice-fidelity check on narrative output
- Whether reasoning leaks into narrative (a known failure mode for
  reasoning models)

Decision criteria for promoting Nemotron permanently or reverting
are in `nemotron3-nano-30b-test-plan.md` §"Decision criteria".

### Restart cost

- Cold start (first time, with HF download): **~16.5 min** (994s
  observed). HF download accounts for ~6.5 min (~52 GB at ~133 MB/s);
  rest is the single 35 GB shard load + `torch.compile` + flashinfer
  autotune + CUDA graph capture. Multi-modal warmup adds ~20s.
- Warm restart (cached weights, second run measured): **~10 min** (618s
  observed). HF download is skipped; most of the time is shard load and
  `torch.compile`. Pick `--max-model-len`, `--gpu-memory-utilization`,
  and the tool-call flags generously up front so you don't pay this
  cost twice.

---

## 4. Filesystem layout

| path | purpose | shared between |
|---|---|---|
| `~/.cache/huggingface` | HF model downloads | both vLLM containers (mounted in) |
| `~/.ollama/models` | Ollama GGUF blobs | Ollama only |
| `/var/lib/docker` | Docker images, vLLM container layers | Docker daemon |
| `/etc/systemd/system/ollama.service.d/override.conf` | Ollama tuning | systemd |

All on the local EXT4 root filesystem. The HF cache is shared between
containers via the `-v` mount, so pulling a model in one container makes
it instantly available to the other.

---

## 5. Network exposure

All three services listen on `0.0.0.0` and are reachable from any host
on the LAN:

| from | to | URL |
|---|---|---|
| laptop, desktop, etc. | Ollama LLM | `http://192.168.1.147:11434/api/...` |
| laptop, desktop, etc. | Ollama OpenAI-compat | `http://192.168.1.147:11434/v1/...` |
| laptop, desktop, etc. | vllm-embed | `http://192.168.1.147:8000/v1/embeddings` |
| laptop, desktop, etc. | vllm-chat | `http://192.168.1.147:8001/v1/chat/completions` |

No auth on any of them — fine for a private LAN, do not expose any of
these ports past the router.

---

## 6. Client-side configuration

> **State drift warning (2026-05-18)**: vllm-chat now serves Nemotron 3
> Nano, but most client configs below still list Gemma 4 model ids. vLLM
> returns 400 on model-id mismatch, so **any client config still on
> Gemma 4 will be broken against the current Nemotron deployment until
> you either**:
>
> 1. **Update the client config** to use
>    `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, OR
> 2. **Revert vllm-chat to Gemma 4**:
>    `ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-gemma4-26b-moe-longctx.sh'`
>
> **Updated for Nemotron as of 2026-05-18**: the
> `opencode-spark-longctx.sh` wrapper script (see opencode section
> below) defaults to the Nemotron model id + 256K. It's the
> recommended way to drive opencode during Phase B validation. All
> other client configs (MemPalace, llm_wiki, CampaignGenerator, and
> the opencode `opencode.json` provider config) remain in Gemma 4
> form intentionally — don't promote them to Nemotron ids until
> Phase B of `nemotron3-nano-30b-test-plan.md` has been completed and
> the decision criteria say to promote.

### MemPalace (`~/.mempalace/config.json` on laptop)

```json
{
  "embedding_provider": "openai-compat",
  "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
  "embedding_endpoint": "http://192.168.1.147:8000",
  "llm_endpoint": "http://192.168.1.147:8001",
  "llm_model": "google/gemma-4-26b-a4b-it"
}
```

mempalace mining embeds via vllm-embed and calls the chat LLM during
the "convos" extraction phase. The `llm_model` field **must match the
model id actually served at port 8001** — every swap of vllm-chat
requires updating this field too, or LLM calls return 400. The chat
palace at `~/.mempalace/palaces/chat/` is currently in a
known-broken state (chroma collection expects 384-dim embeddings, the
embedder produces 768-dim) and pending the rebuild plan in
`~/src/mempalace/chat-palace-rebuild-runbook.md`.

### llm_wiki (Tauri app on Windows desktop)

App Settings → OpenAI-compatible endpoint:
- Endpoint: `http://192.168.1.147:8001/v1`
- Model: `google/gemma-4-26b-a4b-it`

Settings persist to `%APPDATA%\com.llmwiki.app` on Windows.

### CampaignGenerator (`~/src/CampaignGenerator`)

```bash
DGX_MODEL=google/gemma-4-26b-a4b-it python session_doc.py ... \
  --dgx-endpoint http://192.168.1.147:8001/v1
```

### opencode

Two ways to point opencode at the Spark — pick one:

**Option A — wrapper script (preferred for Phase B Nemotron testing)**:
`./opencode-spark-longctx.sh` in this repo. As of 2026-05-18 it
targets the current Nemotron 3 Nano 30B A3B deployment by default
(`MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, `MIN_CTX=262144`)
and exports `OPENAI_API_BASE` / `OPENAI_MODEL` / `OPENAI_API_KEY` before
exec'ing opencode. It prechecks that vllm-chat is actually serving the
expected model at >= MIN_CTX before launching, so a server/wrapper
mismatch fails loud rather than silently spending a session.

To use the wrapper with Gemma 4 (after reverting vllm-chat):

```bash
MODEL_ID=google/gemma-4-26b-a4b-it MIN_CTX=131072 \
  ./opencode-spark-longctx.sh
```

**Option B — opencode's own provider config**: opencode also reads
`~/.config/opencode/opencode.json`. The DGX provider plus the two
previously-validated chat models are defined there. **This file has
not been updated for Nemotron** (the model id below is still Gemma 4)
— if you launch with bare `opencode` instead of the wrapper, you'll
hit a 400 against the current Nemotron deployment until either this
file is updated or vllm-chat is reverted to Gemma 4.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "dgx": {
      "api": "openai",
      "name": "DGX Spark (vLLM)",
      "options": {
        "baseURL": "http://192.168.1.147:8001/v1",
        "apiKey": "ignored"
      },
      "models": {
        "llama-3.3-70b": {
          "id": "casperhansen/llama-3.3-70b-instruct-awq",
          "name": "Llama 3.3 70B Instruct AWQ",
          "limit": { "context": 65536, "output": 8192 },
          "tool_call": true,
          "temperature": true
        },
        "gemma-4-26b-moe": {
          "id": "google/gemma-4-26b-a4b-it",
          "name": "Gemma 4 26B MoE (A4B) BF16",
          "limit": { "context": 32768, "output": 8192 },
          "tool_call": true,
          "temperature": true
        }
      }
    }
  },
  "model": "dgx/gemma-4-26b-moe"
}
```

Launch with bare `opencode` — no env vars needed. To switch the default,
flip the top-level `"model"` field between `"dgx/gemma-4-26b-moe"` and
`"dgx/llama-3.3-70b"` and re-run the matching spin-up script on the
Spark to bring up that model on port 8001. The chosen vllm-chat
container must have the tool-call flags enabled (already set above for
Gemma 4; the llama70b-specdecode script also sets them).

---

## 7. Rebuild-from-scratch order

If the Spark is wiped or you need to recreate from zero, do the steps
in this order so the VRAM budgeting works:

1. **Install Docker + nvidia-container-toolkit** (DGX Sparks ship with
   these but verify with `docker run --rm --gpus all nvidia/cuda:12.4.0-base nvidia-smi`).

2. **Install Ollama** (optional but kept around):
   ```bash
   curl https://ollama.com/install.sh | sh
   sudo systemctl edit ollama
   # paste the [Service] block from section 1
   sudo systemctl daemon-reload && sudo systemctl restart ollama
   ollama pull qwen2.5:14b
   ollama pull nomic-embed-text
   ```

3. **Start vllm-embed FIRST** (so it grabs its tight 5% cap before
   vllm-chat boots and assumes most of the GPU is free):
   ```bash
   # see section 2 for the docker run command
   ```
   Wait for `Application startup complete.` in `docker logs -f vllm-embed`.

4. **Start vllm-chat SECOND** via the current spin-up script
   (Nemotron 3 Nano 30B):
   ```bash
   # Ensure HF_TOKEN is set on the Spark before this — the script will
   # auto-pick it up from ~/.bashrc:
   ssh kostadis@192.168.1.147 'echo "export HF_TOKEN=hf_..." >> ~/.bashrc'

   scp spin-up-vllm-nemotron3-nano-30b.sh kostadis@192.168.1.147:~/
   ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-nemotron3-nano-30b.sh'
   ```
   Expect ~18-30 min on first run (HF pulls ~60 GB on a fresh box;
   authenticated downloads with `HF_TOKEN` are substantially faster).
   The script downloads `nano_v3_reasoning_parser.py` from the HF repo
   into `~/vllm-plugins/` and mounts it into the container. Waits for
   `Application startup complete` and smoke-tests on its own. To
   revert to Gemma 4 (e.g., if Phase B validation rejects Nemotron),
   use `spin-up-vllm-gemma4-26b-moe-longctx.sh` instead.

5. **Smoke-test both** with the curl commands above.

6. **Update client configs** (section 6): MemPalace on the laptop,
   llm_wiki on the desktop, CampaignGenerator env vars, opencode env
   vars.

Total cold-start time: ~30-45 min on a fresh box, mostly download +
torch.compile.

---

## 8. Common operational commands

```bash
# Are the vLLM containers running?
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep vllm

# Watch a vLLM container's logs
docker logs -f vllm-embed
docker logs -f vllm-chat

# Restart a vLLM container after a config change
docker restart vllm-embed   # (~30s)
docker restart vllm-chat    # (~several min — see "Restart cost" above)

# Stop everything (free all VRAM)
docker stop vllm-embed vllm-chat
sudo systemctl stop ollama

# Bring it all back
sudo systemctl start ollama
docker start vllm-embed
# wait
docker start vllm-chat

# What's loaded in Ollama right now?
curl -sS http://192.168.1.147:11434/api/ps | python3 -m json.tool

# What models does vllm-embed serve?
curl -sS http://192.168.1.147:8000/v1/models

# What models does vllm-chat serve?
curl -sS http://192.168.1.147:8001/v1/models

# Watch GPU utilization while a request is in flight
nvidia-smi dmon -s u -c 30

# Disk usage of HF cache (shared between vLLM containers)
du -sh ~/.cache/huggingface

# Swap vllm-chat to a different model on port 8001 (one-liner each):
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-nemotron3-nano-30b.sh'     # Nemotron 3 Nano 30B A3B @ 256K (CURRENT — experimental, Phase A only)
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-gemma4-26b-moe-longctx.sh' # Gemma 4 26B MoE @ 128K context (last validated default)
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-gemma4-26b-moe.sh'         # Gemma 4 26B MoE @ 32K (high-concurrency variant)
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-llama70b-specdecode.sh'    # Llama 3.3 70B AWQ + 1B draft (spec decode)
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-llama70b.sh'               # Llama 3.3 70B AWQ alone (no spec decode)

# Bring up the OPTIONAL gemma-2 sidecar on port 8002 as `vllm-gemma`
# (separate slot — does NOT replace vllm-chat):
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-gemma.sh'
```

---

## See also

- `nemotron3-nano-30b-test-plan.md` — methodology + decision criteria
  for the current Phase A/B experiment. Source of truth for the
  current `vllm-chat` swap.
- `nemotron3-nano-30b-observations.md` — experimental record for
  Nemotron 3 Nano 30B. Phase A complete; Phase B pending.
- `spark-llm-serving-learnings.md` — the "why" and the hardware ceiling
  math behind the choices in this doc.
- `gemma4-26b-moe-observations.md` — full experimental record of the
  Gemma 4 26B MoE deployment (the last validated default; Phase A
  serving behavior, Phase B1 CampaignGenerator, Phase B2 opencode +
  tool calling). Baseline for the current Nemotron comparison.
- `gemma4-26b-moe-runbook.md` — the original plan + runbook for the
  Gemma 4 swap, including the parser-identification step and revert
  path.
- `desktop-chat-clients.md` — Windows-side recipes for chatting with
  vllm-chat from a desktop GUI.
- `bench-prefill.sh` / `bench-decode.sh` — synthetic throughput
  probes used in Phase A measurements. Run from the laptop, point at
  the Spark.
- `CLAUDE.md` — instruction to Claude about keeping this file in sync
  with reality.
