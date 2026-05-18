# Current DGX Spark Setup

Snapshot of what's actually running on the Spark as of 2026-05-17. Use
this as a "rebuild from scratch" reference if the box wipes, or as
inventory when debugging.

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
| 8001 | vllm-chat (docker) | Chat completions — `Gemma 4 26B MoE (A4B)` BF16, 128K context, tool calling on |

## VRAM budget (steady state)

| service | reserved cap | actual model size | notes |
|---|---:|---:|---|
| vllm-embed | ~6 GB (0.05 × 128) | ~600 MB | KV cache fits in cap |
| vllm-chat | ~96 GB (0.75 × 128) | ~48.5 GiB BF16 weights + ~40.5 GiB KV cache @ 128K context | KV cache holds ~557K tokens total (~4 concurrent 128K sessions). 32K variant gives ~17 sessions in the same pool — see §3. |
| Ollama (idle) | ~0 | unloads after `OLLAMA_KEEP_ALIVE` | 5 min default |
| Ollama (loaded) | varies | qwen2.5:14b ≈ 14.5 GB, nomic ≈ 600 MB | only when actively serving |

Both vLLM containers stay resident; Ollama unloads on idle (different
design — see `spark-llm-serving-learnings.md`). Combined steady-state
reservation: ~102 GB (the vLLM allocator's view of its own budget).
**Measured `free -h` while vllm-chat is fully warmed shows only ~4 GiB
available with 7.9 GiB of swap in use** — the 26 GB nominal headroom is
the allocator's bookkeeping, not host-free RAM. Any third vLLM sidecar
(e.g. the optional `vllm-gemma` gemma-2-9b-it container on port 8002)
will need vllm-chat's `--gpu-memory-utilization` dropped before it can
fit cleanly.

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

Chat completions + tool calling service. Backs llm_wiki, CampaignGenerator,
opencode, future chat clients (see `desktop-chat-clients.md`), and any
code calling `/v1/chat/completions`.

Currently serving **Gemma 4 26B MoE (A4B)** in BF16. The slot has held
Qwen 2.5 14B AWQ and Llama 3.3 70B AWQ + spec-decode in earlier
experiments; vllm-chat swap-in scripts are `spin-up-vllm-llama70b.sh`
and `spin-up-vllm-llama70b-specdecode.sh`. (Note: `spin-up-vllm-gemma.sh`
is a *different* slot — it spins up the optional gemma-2-9b-it sidecar
on port 8002 as container `vllm-gemma`, not a replacement for vllm-chat.)

### Run command

Currently launched via `./spin-up-vllm-gemma4-26b-moe-longctx.sh` (the
128K variant, chosen for opencode plan-driven sessions where context
pressure dominates). Effective command:

```bash
docker run -d --runtime nvidia --gpus all \
  --name vllm-chat \
  -p 8001:8001 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  google/gemma-4-26b-a4b-it \
  --max-model-len 131072 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.75 \
  --dtype bfloat16 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --host 0.0.0.0 --port 8001
```

To swap to the 32K / high-concurrency variant: `bash ~/spin-up-vllm-gemma4-26b-moe.sh`.
Same model and flags, different `--max-model-len` (32768 instead of 131072).

### Why these flags

- **`google/gemma-4-26b-a4b-it`**: ~26B total params, ~4B active per token
  via MoE routing. Resolved architecture is
  `Gemma4ForConditionalGeneration` — Gemma 4 is multimodal (vision-language),
  not text-only. We only use the text path today but the vision encoder
  loads and warms at startup either way (~20s extra).
- **`--max-model-len 131072`** (128K): Chosen as the operating point
  for opencode plan-driven sessions where context fills up as tool-call
  results accumulate. Native is 256K but a 256K KV cache for a 26B-class
  model would be ~130 GB — infeasible. The current 128K cap leaves ~4
  concurrent maxed sessions in the KV cache, plenty for the single-user
  Spark. Tradeoff: bigger context = more prefill per turn, and Gemma 4
  MoE already pays the no-tuned-MoE-kernel + TRITON_ATTN prefill
  penalty. Sessions get slightly slower as they grow. If turn latency
  starts dominating, revert to the 32K variant. Concurrency scaling
  reference (from the longctx script header): 32K → ~17 sessions,
  65K → ~8, 128K → ~4 (current), 200K → ~2-3, 256K → ~2.
- **`--max-num-batched-tokens 8192`**: Required, not optional. Without
  it, vLLM fails at startup with `Chunked MM input disabled but
  max_tokens_per_mm_item (2496) is larger than max_num_batched_tokens
  (2048)` because Gemma 4 is multimodal-bidirectional.
- **`--gpu-memory-utilization 0.75`**: ~96 GB cap. 48.5 GiB weights leave
  ~40.5 GiB for KV cache. Drop to 0.70 if startup OOMs.
- **`--dtype bfloat16`**: native Gemma 4 dtype. Don't let vLLM auto-pick FP16.
- **`--trust-remote-code`**: Gemma 4 ships custom modeling code (PLE,
  routing).
- **`--enable-auto-tool-choice` + `--tool-call-parser gemma4`**: vLLM
  0.20.2+ ships a Gemma 4-specific tool-call parser. Required for
  opencode and any other agentic client that needs OpenAI-style
  `tool_calls` in the response. Verify the parser is still available
  in your image with:
  `docker exec vllm-chat vllm serve --help=Frontend | grep tool-call-parser`.

### Known perf ceilings (Spark-specific)

- **No tuned fused-MoE kernel for this hardware**: vLLM looks for
  `configs/E=128,N=704,device_name=NVIDIA_GB10.json` and doesn't find it.
  Performance is on the fallback. This is the biggest single ceiling lever.
- **Heterogeneous attention head dims** (`head_dim=256, global_head_dim=512`)
  force `TRITON_ATTN` instead of FlashAttention.
- **CUDA-graph memory**: vLLM 0.21+ deducts CUDA-graph memory from the
  `--gpu-memory-utilization` budget. With 0.75 set, effective is ~0.741.

### Smoke test

```bash
curl -sS http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"google/gemma-4-26b-a4b-it","messages":[{"role":"user","content":"Say only OK"}],"max_tokens":10}'
# expect: {"id":"chatcmpl-...","choices":[{"message":{"role":"assistant","content":"OK","tool_calls":[],...}}],...}
```

Tool-calling end-to-end probe: `./test-gemma4-toolcall.sh`.

### Measured behaviour

**Headline production finding**: Gemma 4 MoE wins decisively on
read-heavy workloads (opencode reading files, RAG, document
summarization) vs the Llama 3.3 70B + spec-decode alternative — because
prefill is compute-bound and scales with active params (4B vs 70B,
~17× gap). Decode tok/s ties on synthetic benchmarks. For short-input
long-code-output workloads, Llama 70B + spec decode wins on its
~91% draft acceptance on code at T=0. **Default backend choice should be
driven by prefill axis, not decode axis.**

Synthetic measurements:

- Single-sequence decode (262 tokens, T=0 code prompt): **~23 tok/s**
  (matches Llama 70B AWQ + spec-decode on the same prompt).
- 4-parallel concurrency (identical prompts): aggregate **~54.6 tok/s**,
  per-stream **~13.65 tok/s**. Sub-linear scaling (2.35×) — expected for
  bandwidth-bound decode; identical prompts route to identical experts
  so we get no expert-parallelism win.
- Full prefill-vs-decode write-up: `dgx-spark-calibration-report.md` §
  "Cross-cutting finding: prefill vs decode" and
  `gemma4-26b-moe-observations.md` §"Post-experiment correction".

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

opencode reads its provider config from
`~/.config/opencode/opencode.json`. The DGX provider plus the two
currently-supported chat models are defined there:

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

4. **Start vllm-chat SECOND** via the spin-up script (longctx variant
   is the current default — 128K context):
   ```bash
   scp spin-up-vllm-gemma4-26b-moe-longctx.sh kostadis@192.168.1.147:~/
   ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-gemma4-26b-moe-longctx.sh'
   ```
   Expect ~30 min on first run (HF pulls ~52 GB on a fresh box).
   Script waits for `Application startup complete` and smoke-tests on
   its own. If you don't need long context and would rather have higher
   concurrency in the KV pool, use `spin-up-vllm-gemma4-26b-moe.sh`
   instead (32K variant).

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
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-gemma4-26b-moe-longctx.sh' # Gemma 4 26B MoE @ 128K context (CURRENT)
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-gemma4-26b-moe.sh'         # Gemma 4 26B MoE @ 32K (high-concurrency variant)
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-llama70b-specdecode.sh'    # Llama 3.3 70B AWQ + 1B draft (spec decode)
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-llama70b.sh'               # Llama 3.3 70B AWQ alone (no spec decode)

# Bring up the OPTIONAL gemma-2 sidecar on port 8002 as `vllm-gemma`
# (separate slot — does NOT replace vllm-chat):
ssh kostadis@192.168.1.147 'bash ~/spin-up-vllm-gemma.sh'
```

---

## See also

- `spark-llm-serving-learnings.md` — the "why" and the hardware ceiling
  math behind the choices in this doc.
- `gemma4-26b-moe-observations.md` — full experimental record of the
  Gemma 4 26B MoE deployment (Phase A serving behavior, Phase B1
  CampaignGenerator, Phase B2 opencode + tool calling).
- `gemma4-26b-moe-runbook.md` — the original plan + runbook for the
  swap, including the parser-identification step and revert path.
- `desktop-chat-clients.md` — Windows-side recipes for chatting with
  vllm-chat from a desktop GUI.
- `CLAUDE.md` — instruction to Claude about keeping this file in sync
  with reality.
