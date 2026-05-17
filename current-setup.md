# Current DGX Spark Setup

Snapshot of what's actually running on the Spark as of 2026-05-10. Use this
as a "rebuild from scratch" reference if the box wipes, or as inventory
when debugging.

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
| 8001 | vllm-chat (docker) | Chat completions — `Qwen2.5-14B-Instruct-AWQ` |

## VRAM budget (steady state)

| service | reserved cap | actual model size | notes |
|---|---:|---:|---|
| vllm-embed | ~6 GB (0.05 × 128) | ~600 MB | KV cache fits in cap |
| vllm-chat | ~64 GB (0.5 × 128) | ~9 GB AWQ + ~10-15 GB KV @ 32K context | 0.5 cap leaves headroom for both |
| Ollama (idle) | ~0 | unloads after `OLLAMA_KEEP_ALIVE` | 5 min default |
| Ollama (loaded) | varies | qwen2.5:14b ≈ 14.5 GB, nomic ≈ 600 MB | only when actively serving |

Both vLLM containers stay resident; Ollama unloads on idle (different
design — see `spark-llm-serving-learnings.md`). Combined steady-state
reservation: ~70 GB. Of 128 GB unified, that leaves ~58 GB for the OS,
caches, and any third model you might want to spin up.

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

Chat completions service. Backs llm_wiki, future chat clients (see
`desktop-chat-clients.md`), and any code calling `/v1/chat/completions`.

### Run command

```bash
docker run -d --runtime nvidia --gpus all \
  --name vllm-chat \
  -p 8001:8001 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  Qwen/Qwen2.5-14B-Instruct-AWQ \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.5 \
  --host 0.0.0.0 --port 8001
```

### Why these flags

- **`Qwen/Qwen2.5-14B-Instruct-AWQ`**: AWQ-quantised Qwen 2.5 14B Instruct.
  ~9 GB weights vs ~28 GB for FP16, basically no quality loss for
  instruct/JSON tasks. AWQ-Marlin kernel runs well on Blackwell.
- **`--max-model-len 32768`**: Qwen's native max context. Earlier set to
  8192 to save VRAM but llm_wiki ingest pushed past that — restarted at
  32K. KV cache cost is real but the Spark has the headroom.
- **`--gpu-memory-utilization 0.5`**: ~64 GB cap. Holds ~9 GB of weights
  plus enough KV cache for many concurrent 32K-context sequences.

### Smoke test

```bash
curl -sS http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":"Say only OK"}],"max_tokens":10}'
# expect: {"id":"chatcmpl-...","choices":[{"message":{"role":"assistant","content":"OK"},...}],...}
```

### Measured behaviour

- Single-sequence decode: **~15 tok/s** (bandwidth-bound by Spark's 273 GB/s).
- Prompt prefill: **~700 tok/s**.
- Aggregate at 16-way concurrency: theoretical ~190 tok/s (untested in
  practice — see open questions in `spark-llm-serving-learnings.md`).

### Restart cost

"Warm" restart (cached weights, cached torch.compile) still takes
**~2.5-3 minutes**. Don't restart casually. Pick `--max-model-len` and
`--gpu-memory-utilization` generously up front.

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
  "embedding_endpoint": "http://192.168.1.147:8000"
}
```

This is the change shipped in PR #6 (commit 79c09fe on `kostadis-dev`).
mempalace mining now embeds via vllm-embed.

### llm_wiki (Tauri app on Windows desktop)

App Settings → OpenAI-compatible endpoint:
- Endpoint: `http://192.168.1.147:8001/v1`
- Model: `Qwen/Qwen2.5-14B-Instruct-AWQ`

Settings persist to `%APPDATA%\com.llmwiki.app` on Windows.

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

4. **Start vllm-chat SECOND**:
   ```bash
   # see section 3 for the docker run command
   ```
   Wait for `Application startup complete.`

5. **Smoke-test both** with the curl commands above.

6. **Update `~/.mempalace/config.json`** on the laptop (section 6) and
   the llm_wiki Settings on the desktop (section 6).

Total cold-start time: ~10 min (most of which is waiting for HF
downloads and torch.compile on first vllm-chat boot).

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
docker restart vllm-chat    # (~3 min — see "Restart cost" above)

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
```

---

## See also

- `spark-llm-serving-learnings.md` — the "why" and the hardware ceiling
  math behind the choices in this doc.
- `desktop-chat-clients.md` — Windows-side recipes for chatting with
  vllm-chat from a desktop GUI.
