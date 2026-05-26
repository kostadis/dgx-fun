# Current DGX Spark Setup

**Current `vllm-chat` model ids** (copy-paste for client configs):

```
spark1 (192.168.1.147:8001):  Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
spark2 (192.168.1.69:8001):   casperhansen/deepseek-r1-distill-qwen-32b-awq
```

Snapshot of what's actually running on **both** DGX Sparks as of
2026-05-25. Use this as a "rebuild from scratch" reference if either
box wipes, or as inventory when debugging.

> **Two-box layout.** `spark1` is the primary box that backs production
> LLM clients (MemPalace, llm_wiki, CampaignGenerator, opencode). It
> runs `vllm-embed` (port 8000), `vllm-chat` (port 8001), and Ollama
> (port 11434, mostly idle). `spark2` is the experimental sidecar: it
> runs models that are incompatible with the primary clients (e.g.
> reasoning models that emit `<think>` traces llm_wiki can't strip).
> Only spark1 is wired into production workflows; spark2 is for
> opencode sandboxing and side-by-side comparison.

> **Active swap (2026-05-21): Qwen3-Next 80B A3B Instruct FP8 on spark1.**
> `vllm-chat` was swapped from Gemma 4 26B MoE longctx to
> `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` via
> `bash ~/spin-up-vllm-qwen3-next-80b.sh`. 80B total / ~3B active per
> token, hybrid attention (Gated DeltaNet + full attention), FP8
> weights, 128K context. Smoke + tool-call probes (hermes parser) both
> pass. Spin-up required stopping the long-running `vllm-gemma`
> sidecar on port 8002 (held 17 GiB and prevented the 0.88 GPU_UTIL
> budget from fitting). Instruct variant chosen (not Thinking) to
> avoid the Nemotron `<think>`-leak failure mode.
>
> **Nemotron 3 Nano experiment: rejected.** `vllm-chat` ran Nemotron 3
> Nano 30B A3B from 2026-05-18 to 2026-05-19 (Phase A/B per
> `nemotron3-nano-30b-test-plan.md`). Phase B real-workflow validation
> failed: llm_wiki has no parser for Nemotron's `<think>` reasoning
> traces, so the chat box filled with raw thinking output. The
> `opencode.json` still has a `nemotron3-nano-30b` entry as an
> alternate; promoting it back to default requires client-side
> stripping of `<think>` blocks first. Full record:
> `nemotron3-nano-30b-observations.md`.

> **Drift note for Claude**: if you change anything in this list
> (swap a model, add a flag, replace a container), update this file
> in the same change. See `CLAUDE.md`.

## Hardware

Both boxes are GB10 — Grace + Blackwell, 128 GB unified memory,
sm_121, ~273 GB/s memory bandwidth, EXT4 local filesystem.

| field | spark1 (primary) | spark2 (experimental) |
|---|---|---|
| ssh alias | `spark` | `spark2` |
| hostname | `gx10-46ea` | `gx10-3e5c` |
| LAN IP | `192.168.1.147` | `192.168.1.69` |
| docker group for `kostadis` | yes | yes (added manually post-install) |
| nvidia-container-toolkit | configured | configured manually post-install (`nvidia-ctk runtime configure --runtime=docker`) |

## Ports in use

### spark1 (192.168.1.147)

| port | service | purpose |
|---:|---|---|
| 11434 | Ollama (systemd) | LLM serving (legacy, kept for fallback) |
| 8000 | vllm-embed (docker) | Embeddings — `nomic-embed-text-v1.5` |
| 8001 | vllm-chat (docker) | Chat completions — `Qwen3-Next 80B A3B Instruct FP8`, 128K context, hybrid attention, tool calling on |

### spark2 (192.168.1.69)

| port | service | purpose |
|---:|---|---|
| 8001 | vllm-chat (docker) | Chat completions — `DeepSeek R1 distill Qwen 32B AWQ`, 16K context, reasoning + tool calling on |

(No `vllm-embed`, no Ollama — spark2 is single-container.)

## VRAM budget (steady state)

### spark1

| service | reserved cap | actual model size | notes |
|---|---:|---:|---|
| vllm-embed | ~6 GB (0.05 × 128) | ~600 MB | KV cache fits in cap |
| vllm-chat | ~107 GB (0.88 × ~121.7 GiB) | ~80 GiB FP8 weights + ~25 GiB for KV cache (fp8) + activations @ 128K context | Measured ~106 GiB resident with vllm-embed. Hybrid attention: most layers are Gated DeltaNet (no KV), only periodic full-attention layers carry KV — net effect is KV cost materially lower than a Llama-shape at the same context. |
| Ollama (idle) | ~0 | unloads after `OLLAMA_KEEP_ALIVE` | 5 min default |
| Ollama (loaded) | varies | qwen2.5:14b ≈ 14.5 GB, nomic ≈ 600 MB | only when actively serving |

Both vLLM containers stay resident; Ollama unloads on idle (different
design — see `spark-llm-serving-learnings.md`). The 80 GB FP8 weights
+ KV cache + activations leave very little headroom on the 128 GB
unified-memory device. **Any third vLLM sidecar (e.g. the optional
`vllm-gemma` gemma-2-9b-it container on port 8002) will not fit
without dropping vllm-chat's `--gpu-memory-utilization` below 0.88.**
The 2026-05-21 swap to Qwen3-Next OOMed at startup until `vllm-gemma`
was stopped to free 17 GiB.

### spark2

| service | reserved cap | actual model size | notes |
|---|---:|---:|---|
| vllm-chat | ~60 GB (0.5 × ~121.7 GiB) | ~18 GiB AWQ weights (4-bit, awq_marlin) + ~3 GiB KV @ 16K context + activations | Measured ~65 GiB host-side resident at idle. Plenty of headroom: a second container (e.g. an embed sidecar, or a draft model for speculative decoding) easily fits. |

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
# qwen2.5:14b                8.99 GB  (Q4_K_M GGUF)
# qwen2.5:32b                18.5 GB  (Q4_K_M GGUF)
# llama3.3:70b               40 GB    (Q4_K_M GGUF)
```

The 32B and 70B were pulled for ad-hoc comparison against the vLLM
slot; not used by any production client. Safe to `ollama rm` if disk
gets tight.

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

Currently serving **Qwen3-Next 80B A3B Instruct FP8** (hybrid attention,
~3B active per token, ~80B total). Slot history: Qwen 2.5 14B AWQ →
Llama 3.3 70B AWQ + spec-decode → Gemma 4 26B MoE → Gemma 4 26B MoE
longctx → **Nemotron 3 Nano 30B A3B (2026-05-18 to 2026-05-19, rejected
after Phase B — see `nemotron3-nano-30b-observations.md`)** → Gemma 4
26B MoE longctx → **Qwen3-Next 80B A3B Instruct FP8 (2026-05-21 →
current)**. vllm-chat swap-in scripts are
`spin-up-vllm-qwen3-next-80b.sh` (current),
`spin-up-vllm-gemma4-26b-moe-longctx.sh`,
`spin-up-vllm-gemma4-26b-moe.sh`, `spin-up-vllm-llama70b.sh`,
`spin-up-vllm-llama70b-specdecode.sh`, and
`spin-up-vllm-nemotron3-nano-30b.sh` (kept for reference; Nemotron
rejected after Phase B — see top of doc).
(Note: `spin-up-vllm-gemma.sh` is a *different* slot — it spins up the
optional gemma-2-9b-it sidecar on port 8002 as container `vllm-gemma`,
not a replacement for vllm-chat. **Bringing it up will OOM the
Qwen3-Next container** unless GPU_UTIL is dropped — see VRAM budget
above.)

### Run command

Currently launched via `./spin-up-vllm-qwen3-next-80b.sh`. Effective
command:

```bash
docker run -d --runtime nvidia --gpus all \
  --name vllm-chat \
  -p 8001:8001 \
  --ipc=host \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --max-model-len 131072 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 0.0.0.0 --port 8001
```

### Why these flags

- **`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`**: ~80B total, ~3B active
  per token. Hybrid architecture: Gated DeltaNet (linear attention,
  O(n), no KV cache) on most layers + periodic full-attention layers
  + MoE FFN. Native context 256K (extendable to 1M with YaRN); we run
  128K. **Instruct** variant chosen explicitly over Thinking to avoid
  re-tripping the Nemotron `<think>`-leak failure mode in llm_wiki.
- **`--max-model-len 131072`** (128K): conservative starting point vs
  256K native. Memory is the binding constraint on a 128 GB unified
  box with 80 GB of weights — see the fallback ladder in the script
  header.
- **`--max-num-seqs 4`**: KV is the bottleneck, not compute. Don't
  over-batch.
- **`--gpu-memory-utilization 0.88`**: ~107 GiB cap on the ~121.7 GiB
  device. Tight — required stopping `vllm-gemma` (17 GiB) before
  startup would fit. Fallback ladder if it OOMs: MAX_LEN=65536 first,
  then GPU_UTIL=0.85, then 0.82.
- **`--kv-cache-dtype fp8`**: halves KV memory vs BF16. Set
  `KV_CACHE_DTYPE=auto` if the image build lacks FP8 KV kernels for
  the hybrid attention impl (also drop MAX_LEN to compensate for 2×
  KV memory).
- **`--trust-remote-code`**: Qwen3-Next ships custom modeling code.
- **`--enable-auto-tool-choice` + `--tool-call-parser hermes`**:
  default Qwen3 chat-template parser. Verified end-to-end with
  `test-toolcall.sh` (PASS — null content + tool_calls[get_weather]
  + parseable arguments). Alternate: `TOOL_PARSER=qwen3_coder` (the
  stricter parser designed for Qwen3-Coder's tool format).

### Known perf ceilings (Spark-specific)

- **Hybrid-attention kernel maturity on sm_121**: Qwen3-Next's Gated
  DeltaNet + full-attention path is newer in vLLM than the Gemma 4
  MoE path. If vLLM hasn't shipped a tuned CUDA kernel for this arch
  + GPU combination, it falls through to a Triton kernel — perf
  ceiling, not a correctness bug.
- **CUDA-graph memory**: vLLM 0.21+ deducts CUDA-graph memory from
  the `--gpu-memory-utilization` budget.

### Smoke test

```bash
curl -sS http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Next-80B-A3B-Instruct-FP8","messages":[{"role":"user","content":"Say only OK"}],"max_tokens":10}'
# expect: {"id":"chatcmpl-...","choices":[{"message":{"role":"assistant","content":"OK","tool_calls":[],"reasoning":null,...}}],...}
```

Tool-calling end-to-end probe:
`MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 ./test-toolcall.sh`.

### Measured behaviour

**Pending Phase A.** No benchmarks yet under this slot — measure
prefill vs decode against the Gemma 4 26B MoE baseline (see
`gemma4-26b-moe-observations.md` and `dgx-spark-calibration-report.md`).
Expected: prefill should be in the same ballpark as Gemma 4 (4B active
vs 3B active) or better, because linear-attention layers are O(n) at
long context. The open question is whether vLLM's hybrid-kernel path
on GB10 (sm_121) is mature enough to realize that.

### Restart cost

- Cold start (first time, with HF download): **measured 2026-05-21**
  — HF download (~80 GB FP8 weights) + shard load + compile + warmup
  fit within the 40-min budget. First successful run completed in
  ~10-15 min after the failed-and-fixed `vllm-gemma` precondition was
  resolved (exact timing not captured separately from the failed
  attempt).
- Warm restart (cached weights): expected ~5-10 min. HF download is
  skipped; most of the time is shard load and `torch.compile`.

---

## 4. spark2 vllm-chat (Docker container, port 8001)

Single-container experimental slot on the second box. Currently
serving **DeepSeek R1 distill Qwen 32B AWQ** — a reasoning model
that emits `<think>` traces, with a vLLM-side reasoning parser so
clients get a clean `reasoning` field separate from `content`.

Slot history on spark2: Nemotron 3 Nano 30B A3B BF16 (2026-05-22 →
~2026-05-24) → **DeepSeek R1 distill Qwen 32B AWQ (~2026-05-24 →
current)**. Both rejected from spark1 because llm_wiki has no
`<think>` parser; spark2 is where reasoning models live.

### Run command

```bash
docker run -d --runtime nvidia --gpus all \
  --name vllm-chat \
  -p 8001:8001 \
  --ipc=host \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  casperhansen/deepseek-r1-distill-qwen-32b-awq \
  --quantization awq_marlin \
  --dtype float16 \
  --max-model-len 16384 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.5 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1 \
  --host 0.0.0.0 --port 8001
```

(No `spin-up-vllm-deepseek-r1-distill.sh` script committed yet —
container was launched ad-hoc. Reconstruct from this block on
rebuild, or add a script and link it here.)

### Why these flags

- **`casperhansen/deepseek-r1-distill-qwen-32b-awq`**: 32B dense
  Qwen-arch model, distilled to mimic DeepSeek R1's reasoning trace,
  AWQ-quantized to 4-bit (~18 GiB on disk). Same architecture family
  as Qwen 2.5 / Qwen 3, so AWQ tooling and the `hermes` tool-call
  parser both apply.
- **`--quantization awq_marlin`**: Marlin is the fast AWQ kernel
  path; explicit selection rather than letting vLLM pick (some image
  versions default to a slower fallback for AWQ on sm_121).
- **`--dtype float16`**: AWQ kernels expect FP16 activations.
- **`--max-model-len 16384`** (16K): conservative — much smaller
  than spark1's 128K because the use case (reasoning Q&A, opencode
  experiments) doesn't need long context, and the model's native
  context tops out around 32K anyway.
- **`--max-num-seqs 4`**: matches spark1's conservative batching.
- **`--gpu-memory-utilization 0.5`** (~60 GiB cap): leaves room for
  a second container later. The 32B AWQ model only needs ~22 GiB
  weights+KV, so the 60 GiB cap is more about reserving headroom
  than fitting the model.
- **`--enable-auto-tool-choice` + `--tool-call-parser hermes`**:
  matches spark1. Hermes parser works for the Qwen tokenizer
  family.
- **`--reasoning-parser deepseek_r1`** *(key difference vs spark1)*:
  vLLM strips the `<think>...</think>` block from `content` and
  surfaces it under `choices[0].message.reasoning`. This is what
  makes the model usable for clients that *do* understand a
  `reasoning` field (opencode) — but llm_wiki still won't render
  it, which is why this model lives on spark2 and not spark1.
- **`-e HF_TOKEN`**: passthrough so vLLM can pull the AWQ weights
  on first run. The `.profile` export-keyword quirk applies here
  too — verify with a grandchild process if the token is missing.

### Smoke test

```bash
curl -sS http://192.168.1.69:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"casperhansen/deepseek-r1-distill-qwen-32b-awq","messages":[{"role":"user","content":"Say only OK"}],"max_tokens":20}'
# expect: {"choices":[{"message":{"content":"\n\nOK","reasoning":"\n\n",...}}],...}
# note: "reasoning" field is populated even for trivial prompts because the parser is always on.
```

### Measured behaviour

No benchmarks logged yet on spark2. For comparison against spark1's
Qwen3-Next, see `model-comparisons.md` (todo — currently spark1-only).

### Restart cost

- Cold start (first time, with HF download): ~5–10 min (AWQ weights
  are ~18 GB).
- Warm restart (cached weights, what `docker start vllm-chat` does
  post-reboot): ~30–60 s. Much faster than spark1 because the model
  is 4× smaller and 4-bit quantized.

---

## 5. Filesystem layout

| path | host | purpose | shared between |
|---|---|---|---|
| `~/.cache/huggingface` | spark1 | HF model downloads | both spark1 vLLM containers (mounted in) |
| `~/.ollama/models` | spark1 | Ollama GGUF blobs | Ollama only |
| `/var/lib/docker` | spark1, spark2 | Docker images, vLLM container layers | Docker daemon |
| `/etc/systemd/system/ollama.service.d/override.conf` | spark1 | Ollama tuning | systemd |
| `~/.cache/huggingface` | spark2 | HF model downloads (~114 GB resident) | spark2 vllm-chat only |

All on the local EXT4 root filesystem of each box. **The two HF caches
are not shared between boxes** — pulling a model on spark1 does not
make it available to spark2 and vice versa. Within a single box, the
cache is shared between containers via the `-v` mount.

---

## 6. Network exposure

All services on both boxes listen on `0.0.0.0` and are reachable from
any host on the LAN:

| from | to | URL |
|---|---|---|
| laptop, desktop, etc. | spark1 Ollama LLM | `http://192.168.1.147:11434/api/...` |
| laptop, desktop, etc. | spark1 Ollama OpenAI-compat | `http://192.168.1.147:11434/v1/...` |
| laptop, desktop, etc. | spark1 vllm-embed | `http://192.168.1.147:8000/v1/embeddings` |
| laptop, desktop, etc. | spark1 vllm-chat | `http://192.168.1.147:8001/v1/chat/completions` |
| laptop, desktop, etc. | spark2 vllm-chat | `http://192.168.1.69:8001/v1/chat/completions` |

No auth on any of them — fine for a private LAN, do not expose any of
these ports past the router.

---

## 7. Client-side configuration

### MemPalace (`~/.mempalace/config.json` on laptop)

```json
{
  "embedding_provider": "openai-compat",
  "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
  "embedding_endpoint": "http://192.168.1.147:8000",
  "llm_endpoint": "http://192.168.1.147:8001",
  "llm_model": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
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
- Model: `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`

Settings persist to `%APPDATA%\com.llmwiki.app` on Windows.

### CampaignGenerator (`~/src/CampaignGenerator`)

```bash
DGX_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 python session_doc.py ... \
  --dgx-endpoint http://192.168.1.147:8001/v1
```

### opencode

opencode reads its provider config from
`~/.config/opencode/opencode.json`. The DGX provider has five
registered chat models; the active default is `dgx/qwen3-next-80b`:

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
        "qwen3-next-80b": {
          "id": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
          "name": "Qwen3-Next 80B A3B Instruct FP8 @ 128K (hybrid attention, tools)",
          "limit": { "context": 131072, "output": 8192 },
          "tool_call": true,
          "temperature": true
        },
        "nemotron3-nano-30b": {
          "id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
          "name": "Nemotron 3 Nano 30B A3B BF16 (256K, reasoning+tools)",
          "limit": { "context": 262144, "output": 8192 },
          "tool_call": true,
          "reasoning": true,
          "temperature": true
        },
        "gemma-4-26b-moe-longctx": {
          "id": "google/gemma-4-26b-a4b-it",
          "name": "Gemma 4 26B MoE (A4B) BF16 @ 128K",
          "limit": { "context": 131072, "output": 8192 },
          "tool_call": true,
          "temperature": true
        },
        "gemma-4-26b-moe": {
          "id": "google/gemma-4-26b-a4b-it",
          "name": "Gemma 4 26B MoE (A4B) BF16 @ 32K (high concurrency)",
          "limit": { "context": 32768, "output": 8192 },
          "tool_call": true,
          "temperature": true
        },
        "llama-3.3-70b": {
          "id": "casperhansen/llama-3.3-70b-instruct-awq",
          "name": "Llama 3.3 70B Instruct AWQ",
          "limit": { "context": 65536, "output": 8192 },
          "tool_call": true,
          "temperature": true
        }
      }
    }
  },
  "model": "dgx/qwen3-next-80b"
}
```

Launch with bare `opencode` — no env vars needed. To switch the default,
flip the top-level `"model"` field to one of the five registered ids
and re-run the matching spin-up script on the Spark to bring up that
model on port 8001. The chosen vllm-chat container must have the
tool-call flags enabled (already set in all current spin-up scripts).

The `nemotron3-nano-30b` entry is preserved as an alternate but is
**not the default** after Phase B testing rejected Nemotron: llm_wiki
has no parser for `<think>` reasoning traces, so the chat box filled
with raw thinking (`nemotron3-nano-30b-observations.md`). The
`opencode-spark-longctx.sh` wrapper from Phase B still works for Gemma
4 if invoked with `MODEL_ID=google/gemma-4-26b-a4b-it MIN_CTX=131072
./opencode-spark-longctx.sh` — but the bare `opencode` invocation
using `opencode.json` is the standard path.

opencode also has a second provider `dgx2` pointed at spark2
(`http://192.168.1.69:8001/v1`). Switch to a spark2 model by setting
the top-level `"model"` field to `dgx2/<model-id>`.

> **Drift flag (2026-05-25):** the `dgx2` provider currently registers
> exactly one model id, `nemotron3-nano-30b` →
> `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`. **That id no longer
> matches reality** — spark2 is serving
> `casperhansen/deepseek-r1-distill-qwen-32b-awq`. Until
> `opencode.json` is updated, `dgx2/nemotron3-nano-30b` will return
> HTTP 400 from vLLM. Fix: replace the entry with a
> `deepseek-r1-distill-qwen-32b` entry whose `id` matches the served
> model, `reasoning: true`, `limit.context: 16384`. Left unfixed
> intentionally to surface here — owner's call whether to keep the
> Nemotron entry as a placeholder for a future swap-back.

---

## 8. Rebuild-from-scratch order

### Post-reboot (containers exist, just stopped)

If both boxes were rebooted but the docker containers are intact
(this is the common case), skip the rebuild and use `docker start`
— it preserves all flags from the prior run:

```bash
# spark1: embed first, then chat
ssh spark 'docker start vllm-embed'
until curl -sS --max-time 2 http://192.168.1.147:8000/v1/models 2>/dev/null | grep -q '"id"'; do sleep 3; done
ssh spark 'docker start vllm-chat'
until curl -sS --max-time 2 http://192.168.1.147:8001/v1/models 2>/dev/null | grep -q '"id"'; do sleep 10; done

# spark2 in parallel (independent box)
ssh spark2 'docker start vllm-chat'
until curl -sS --max-time 2 http://192.168.1.69:8001/v1/models 2>/dev/null | grep -q '"id"'; do sleep 5; done
```

Warm-restart cost: vllm-embed ~30s, spark1 vllm-chat ~5–10 min
(80 GB weights), spark2 vllm-chat ~30–60s (18 GB AWQ weights).

### Full rebuild from a wiped box

If a box is wiped and you need to recreate from zero, do the steps
in this order so the VRAM budgeting works.

#### spark1 (primary)

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

4. **Start vllm-chat SECOND** via the spin-up script (Qwen3-Next 80B
   FP8 is the current default — 128K context, hybrid attention):
   ```bash
   scp spin-up-vllm-qwen3-next-80b.sh lib-vllm-spinup.sh test-toolcall.sh spark:~/
   ssh spark 'docker pull vllm/vllm-openai:latest'
   ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b.sh'
   ```
   Expect ~40 min on first run (HF pulls ~80 GB of FP8 weights on a
   fresh box). Script waits for `Application startup complete` and
   smoke-tests on its own. Then verify tool calling:
   `ssh spark 'MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 ~/test-toolcall.sh'`.
   To swap to a different model later, see §9.

5. **Smoke-test both** with the curl commands above.

6. **Update client configs** (section 7): MemPalace on the laptop,
   llm_wiki on the desktop, CampaignGenerator env vars, opencode env
   vars.

Total spark1 cold-start time: ~30–45 min on a fresh box, mostly
download + torch.compile.

#### spark2 (experimental)

spark2 ships from NVIDIA with a few defaults that bit us:

1. **Add `kostadis` to the `docker` group** (not done by default on
   spark2):
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker   # or log out/in
   ```

2. **Configure nvidia-container-toolkit for Docker** (installed but
   not wired in by default on spark2):
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   docker run --rm --gpus all nvidia/cuda:12.4.0-base nvidia-smi  # verify
   ```

3. **Export `HF_TOKEN` so it reaches ssh-launched scripts.** Add to
   `~/.profile` with the `export` keyword — a bare `HF_TOKEN=...`
   assignment is shell-local and won't reach a `docker run -e
   HF_TOKEN="$HF_TOKEN"` invoked over ssh. Verify with a grandchild
   process: `ssh spark2 'bash -c "echo \$HF_TOKEN"'`.

4. **Start vllm-chat** with the DeepSeek R1 distill block from §4
   above. Expect ~5–10 min on first run (HF pulls ~18 GB of AWQ
   weights). No spin-up script committed yet — paste the `docker
   run` block from §4 directly.

5. **Smoke-test** with the curl command from §4.

6. **Update opencode `dgx2` provider** to register the served model
   id (see drift flag in §7).

---

## 9. Common operational commands

### spark1

```bash
# Are the vLLM containers running?
ssh spark 'docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep vllm'

# Watch a vLLM container's logs
ssh spark 'docker logs -f vllm-embed'
ssh spark 'docker logs -f vllm-chat'

# Restart a vLLM container after a config change
ssh spark 'docker restart vllm-embed'   # (~30s)
ssh spark 'docker restart vllm-chat'    # (~5-10 min — see §3 "Restart cost")

# Stop everything (free all VRAM)
ssh spark 'docker stop vllm-embed vllm-chat && sudo systemctl stop ollama'

# Bring it all back (embed before chat for VRAM-budget reasons)
ssh spark 'sudo systemctl start ollama && docker start vllm-embed'
# wait for embed (see §8 post-reboot block for the until-loop)
ssh spark 'docker start vllm-chat'

# What's loaded in Ollama right now?
curl -sS http://192.168.1.147:11434/api/ps | python3 -m json.tool

# What models does vllm-embed / vllm-chat serve?
curl -sS http://192.168.1.147:8000/v1/models
curl -sS http://192.168.1.147:8001/v1/models

# Watch GPU utilization while a request is in flight
ssh spark 'nvidia-smi dmon -s u -c 30'

# Disk usage of HF cache (shared between vLLM containers)
ssh spark 'du -sh ~/.cache/huggingface'

# Swap vllm-chat to a different model on port 8001 (one-liner each):
ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b.sh'         # Qwen3-Next 80B A3B Instruct FP8 @ 128K (CURRENT)
ssh spark 'bash ~/spin-up-vllm-gemma4-26b-moe-longctx.sh' # Gemma 4 26B MoE @ 128K context
ssh spark 'bash ~/spin-up-vllm-gemma4-26b-moe.sh'         # Gemma 4 26B MoE @ 32K (high-concurrency variant)
ssh spark 'bash ~/spin-up-vllm-llama70b-specdecode.sh'    # Llama 3.3 70B AWQ + 1B draft (spec decode)
ssh spark 'bash ~/spin-up-vllm-llama70b.sh'               # Llama 3.3 70B AWQ alone (no spec decode)
# Nemotron experiment concluded — rejected after Phase B (see top of doc + observations).
# Script kept in repo for reference:
# ssh spark 'bash ~/spin-up-vllm-nemotron3-nano-30b.sh'

# Bring up the OPTIONAL gemma-2 sidecar on port 8002 as `vllm-gemma`
# (separate slot — does NOT replace vllm-chat):
ssh spark 'bash ~/spin-up-vllm-gemma.sh'
```

### spark2

```bash
# Is vllm-chat running?
ssh spark2 'docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'

# Watch / restart / stop / start
ssh spark2 'docker logs -f vllm-chat'
ssh spark2 'docker restart vllm-chat'   # (~30-60s warm restart)
ssh spark2 'docker stop vllm-chat'
ssh spark2 'docker start vllm-chat'

# What model is served?
curl -sS http://192.168.1.69:8001/v1/models

# Smoke-test (model id has to match — see top of doc)
curl -sS http://192.168.1.69:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"casperhansen/deepseek-r1-distill-qwen-32b-awq","messages":[{"role":"user","content":"Say only OK"}],"max_tokens":20}'

# HF cache size (spark2-only — not shared with spark1)
ssh spark2 'du -sh ~/.cache/huggingface'

# To swap the spark2 model: no committed spin-up script yet — adapt the
# docker run block from §4 with the new model id and re-run.
```

---

## See also

- `nemotron3-nano-30b-test-plan.md` — methodology + decision criteria
  for the 2026-05-18→05-19 Nemotron experiment (rejected; see top of
  doc).
- `nemotron3-nano-30b-observations.md` — experimental record for
  Nemotron 3 Nano 30B. Phase A complete; Phase B failed on llm_wiki
  reasoning-trace leakage, so vllm-chat was reverted to Gemma 4.
- `spark-llm-serving-learnings.md` — the "why" and the hardware ceiling
  math behind the choices in this doc.
- `gemma4-26b-moe-observations.md` — full experimental record of the
  Gemma 4 26B MoE deployment (the current default; Phase A serving
  behavior, Phase B1 CampaignGenerator, Phase B2 opencode + tool
  calling). Baseline that the Nemotron experiment was measured
  against.
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
