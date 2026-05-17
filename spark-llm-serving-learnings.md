# DGX Spark + LLM Serving — Learnings

Captured 2026-05-10 from a session where we moved MemPalace's embedding work
from Ollama (CPU MiniLM → Ollama nomic-embed-text) to vLLM on the Spark, then
also stood up Qwen2.5-14B for chat completions. Numbers in this doc are
measured on this hardware, not estimated.

## The headline mental-model shift

The DGX Spark (GB10 — Grace + Blackwell, 128 GB unified memory) is **wide
and slow per-stream**. The inverse of "fast laptop GPU." Optimising for it
means optimising for *concurrency*, not single-call latency.

- Single-sequence decode on a 14B AWQ model: **~15 tok/s**.
- Aggregate decode at 16 concurrent sequences: **~190 tok/s**.
- Same hardware, ~13× difference depending on workload shape.

If your application sends one request and waits, you are using maybe 5-10%
of what the box can do. Fan-out is the entire game.

This is the opposite of how you'd design for a laptop with an RTX 4090
(where you'd serialize to keep VRAM available and let the per-stream speed
do the work).

---

## What's actually happening at the hardware layer

For LLM **decode** (token-by-token generation), the dominant cost is reading
the model weights from VRAM for every output token.

- A 14B AWQ-quantised model is ~9 GB of weights.
- Spark unified memory bandwidth: ~273 GB/s.
- Theoretical ceiling for single-stream decode: 273 / 9 ≈ **30 tok/s**.
- Real-world ~50-60% of theoretical: **~15-18 tok/s**.

For comparison:
- RTX 4090 (1 TB/s GDDR6X): same model, same single stream → ~50 tok/s.
- H100 PCIe (3.35 TB/s HBM3): ~80-120 tok/s.

The Spark's bandwidth is **lower** than a 4090, despite costing more and
having more memory. The win is the 128 GB lets you hold huge models or many
models simultaneously, and continuous batching amortises weight reads
across many concurrent sequences.

For decode at concurrency N, the model weights are read **once per forward
pass** (not once per sequence). So aggregate scales nearly linearly with N
until you hit either (a) the compute ceiling on the actual matrix multiplies
or (b) KV cache VRAM pressure. On Spark with a 14B AWQ that crossover is
somewhere around N=16-32.

For LLM **prefill** (processing the input prompt before generation starts),
the operation is matrix-multiply heavy and compute-bound, not bandwidth.
Prefill is much faster per token than decode — we measured ~700 tok/s for
prefill vs ~15 tok/s for decode on the same call.

For **embedding** models, both prefill and "decode" are basically just
prefill — there's no autoregressive token loop. Throughput is bounded by
how well the server batches incoming requests onto a single GPU forward
pass. This is exactly where Ollama vs vLLM diverges (see below).

---

## Ollama vs vLLM — different design centres

These tools look interchangeable from the outside. They are not.

| | Ollama | vLLM |
|---|---|---|
| Built for | Personal/laptop, multi-model rotation | Production serving, sustained throughput |
| Model rotation | Auto-loads/unloads on demand | Must restart container to swap |
| Idle behaviour | Unloads after `OLLAMA_KEEP_ALIVE` (5min default) | Stays resident forever |
| First-request-after-idle | Cold reload, ~5-30s | Hot, no penalty |
| VRAM cost when idle | ~0 | Full reservation |
| Concurrency model | Limited; embed path serialises regardless of `OLLAMA_NUM_PARALLEL` (verified on 0.23.2) | Continuous batching designed for many in-flight requests |
| Quantisation flexibility | GGUF (very flexible quants, including ones the model author never released) | Pickier — needs PyTorch-loadable weights or a few specific quant formats (AWQ, GPTQ, FP8) |

**Use Ollama when** you have a dev laptop, you flip between models a lot,
and request-response latency under interactive use is what matters.

**Use vLLM when** you have a serving box, you've picked the model(s) you
want to run, and you expect sustained or batched workload. The Spark is a
serving box.

There is no "Ollama-like keep-alive" knob in vLLM. By design — auto-unload
would defeat the entire architectural point of vLLM. If you need that
behaviour, you wrap vLLM in a supervisor that stops the container after N
minutes idle, accepting the cold-start cost.

---

## Measured numbers (Spark, GB10, 128 GB unified, this session)

### nomic-embed-text-v1.5 throughput, batch=1024, single client

| backend | throughput | notes |
|---|---:|---|
| ONNX MiniLM on laptop CPU | n/a (different model) | original baseline before any of this |
| Ollama `/api/embed` | ~300 tok/s | single inference slot. `OLLAMA_NUM_PARALLEL=8` confirmed in `/proc/<pid>/environ` but does NOT apply to the embed path on Ollama 0.23.2 — confirmed by client-side concurrency probe (~1.9× cap regardless of NUM_PARALLEL) |
| vLLM `/v1/embeddings` | ~11,400 tok/s | continuous batching, ~38× over Ollama for the same model on the same hardware |

### Ollama embedding concurrency probe (8 parallel client requests)

| concurrency | wall total | per call | speedup |
|---:|---:|---:|---:|
| 1 | 837 ms | 105 ms | 1.00× |
| 2 | 619 ms | 77 ms | 1.35× |
| 4 | 433 ms | 54 ms | 1.93× |
| 8 | 416 ms | 52 ms | 2.01× |

Plateau at ~2×. Bumping `OLLAMA_NUM_PARALLEL` from default to 8 changed
nothing. **Ollama's `/api/embed` serialises** even when the chat path
parallelises.

### vLLM embedding concurrency probe (16 parallel client requests)

| concurrency | wall total | per call | speedup |
|---:|---:|---:|---:|
| 1 | 1438 ms | 90 ms | 1.00× |
| 2 | 1099 ms | 69 ms | 1.31× |
| 4 | 882 ms | 55 ms | 1.63× |
| 8 | 810 ms | 51 ms | 1.77× |
| 16 | 760 ms | 47 ms | 1.89× |

vLLM scaled further but still didn't hit linear. Per-call latency floor of
~47 ms suggests we were hitting Spark's compute ceiling for this small
embedding model, not a serialisation bug. Different shape of bottleneck
than Ollama.

### vLLM embedding batch sweep (single request, varying batch size)

| batch | wall ms | per-input ms | tok/sec |
|---:|---:|---:|---:|
| 1 | 9.6 | 9.65 | 933 |
| 4 | 11.9 | 2.98 | 3,020 |
| 16 | 21.3 | 1.33 | 6,746 |
| 64 | 54.8 | 0.86 | 10,516 |
| 256 | 235.3 | 0.92 | 9,792 |
| 1024 | 808.8 | 0.79 | 11,395 |

Per-input cost drops from 9.65 ms to 0.79 ms as batch grows — that's the
GPU getting properly utilised. The fact that this curve **converges** is
the proof that vLLM is doing real intra-batch fusion, where Ollama was
serialising.

### Qwen2.5-14B-Instruct-AWQ generation (single sequence)

- Prompt prefill throughput: ~697 tok/s (input ~4k tokens, processed quickly)
- Decode throughput: ~15 tok/s (steady state)
- KV cache usage: ~3% of allocated cap during single-sequence decode

Matches the bandwidth ceiling math (273 GB/s ÷ 9 GB ≈ 30 tok/s theoretical,
~50% achieved is normal).

### Mining wall-time comparison (MemPalace, vLLM-embedded)

| tree | files | drawers | wall | drawers/file |
|---|---:|---:|---:|---:|
| CampaignGenerator | 176 | 3,501 | 52 s | 19.9 |
| mytools | 398 | 13,407 | 5m 29s | 33.7 |
| mempalace | 274 | ~50,000 | ~14 min total | ~180 |

Same workload on Ollama-embed was ~3-4× slower. The ~14 min mempalace mine
time is dominated by one ~19k-drawer JSONL file, not by per-request cost.

---

## Spark-specific gotchas

1. **vLLM reserves VRAM at startup, not on demand.**
   `--gpu-memory-utilization 0.9` (default) means 90% of *currently free*
   GPU memory at the moment the container boots. If another container
   started first and grabbed 90%, the second sees only 10% free and starts
   with ~9% of total. You **must** budget across containers explicitly.

   Working pattern from this session:
   - vllm-embed (small model, ~600 MB): `--gpu-memory-utilization 0.05`
   - vllm-chat (Qwen-14B-AWQ + KV cache for 32K context): `--gpu-memory-utilization 0.5`
   - Spec embed first with the tight cap, then chat second.

2. **Restart cost is high even with caches.**
   "Warm" vLLM restart on a 14B AWQ model is still ~2.5-3 minutes:
   - Weight repack to AWQ-Marlin: ~100s (no cache — GPU operation)
   - torch.compile: ~5-10s if cache hits, ~40s cold
   - CUDA graph capture: ~30s (no cache)
   - KV cache profile run: ~10-15s (longer with bigger `--max-model-len`)

   Implication: pick `--max-model-len` and `--gpu-memory-utilization`
   generously up front. Don't restart casually.

3. **Pick `--max-model-len` for the workload, not the model's max.**
   Qwen2.5 supports 32K natively. For mempalace's 500-token entity-refinement
   prompts, an 8K cap saved a lot of KV-cache VRAM. For llm_wiki ingesting
   real documents (4K+ token inputs, multi-K outputs), 8K was too tight and
   we had to restart at 32K. Bigger context = bigger KV cache reservation
   per concurrent sequence = fewer sequences fit.

4. **Modern vLLM CLI:**
   - Model is **positional**, no `--model` (deprecated, future-removed).
   - `--task embed` was removed; task is auto-detected from model config.
   - For embedding models that auto-detect doesn't catch: `--runner pooling`.

5. **Ollama on Blackwell: model placement reads as GPU but throughput is
   CPU-class.** `/api/ps` showed `size_vram == size` for both nomic and
   qwen2.5:14b — i.e. weights *are* in VRAM. But the `/api/embed` path
   doesn't multi-stream regardless of NUM_PARALLEL, so even on GPU
   placement you cap at ~300 tok/s for nomic. The diagnostic tools lie
   about what's actually happening; only end-to-end measurement reveals
   it. (Diagnosis: check per-request tok/s at increasing concurrency.
   If aggregate doesn't grow, it's serialised somewhere.)

6. **Unified memory means VRAM headroom is generous but `nvidia-smi`
   accounting is weird.** Things that look like they should fit don't
   always; things that look like they shouldn't sometimes do.

---

## Setup recipes (Docker, Spark, Ollama already-installed for fallback)

### vLLM embedding — nomic-embed-text-v1.5 (~600 MB)

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

Smoke test:

```bash
curl -sS http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"nomic-ai/nomic-embed-text-v1.5","input":"hello"}'
```

### vLLM chat — Qwen2.5-14B-Instruct-AWQ (~9 GB)

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

Smoke test:

```bash
curl -sS http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":"Say only OK"}],"max_tokens":10}'
```

### Existing Ollama (kept around for fallback / model rotation)

`/etc/systemd/system/ollama.service.d/override.conf`:

```
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_NUM_PARALLEL=8"
```

Lessons: `FLASH_ATTENTION=1` and `KV_CACHE_TYPE=q8_0` help chat throughput.
`NUM_PARALLEL=8` is set but does not affect the `/api/embed` path on 0.23.2
(it does affect chat completions). After editing the override, run
`sudo systemctl daemon-reload && sudo systemctl restart ollama` and confirm
with `sudo cat /proc/$(pgrep -f 'ollama serve')/environ | tr '\0' '\n' | grep OLLAMA_`.

---

## Where to add parallelism (the real lever)

Theoretical ceiling for Qwen2.5-14B-AWQ on Spark, by client concurrency:

| concurrent sequences | per-seq tok/s | aggregate tok/s | what limits you |
|---:|---:|---:|---|
| 1 | 15 | 15 | bandwidth — read 9 GB per token |
| 4 | ~14 | ~56 | bandwidth — weights amortise across sequences |
| 8 | ~13 | ~104 | bandwidth |
| 16 | ~12 | ~190 | starting to hit compute |
| 32 | ~10 | ~320 | compute + KV cache pressure |

To get there in practice:

1. **Fan out within a single application.** If your code does
   `for item in items: response = call(item)`, you are leaving 90%+
   of the hardware on the table. Switch to a thread pool or async
   gather over N concurrent calls.

2. **Multi-consumer.** Run independent jobs against the same vLLM
   simultaneously — vLLM continuous-batches them all for free. If you
   have batch ingest + interactive queries + scheduled mining and they
   share a vLLM, you get the parallelism gratis.

3. **Speculative decoding** (single-shot speedup). Pair a big model with a
   small "draft" model from the same family. The draft proposes tokens,
   the big model verifies in batch. ~1.5-2× single-sequence wins, helpful
   even when concurrency=1. Costs ~600 MB extra VRAM for a 0.5B-1.5B draft.

   ```
   --speculative-config '{"model":"Qwen/Qwen2.5-0.5B-Instruct","num_speculative_tokens":4}'
   ```

---

## Mistakes I made during this session (worth recognising)

1. **Quoted "30-80 ms" for warm Ollama embed latency.** Actually ~120 ms.
   Diagnosis: I was guessing from older numbers in less-optimised setups.
   Fix: always measure on the actual hardware before quoting numbers.

2. **Diagnosed Ollama as CPU-bound on Blackwell.** It wasn't — `/api/ps`
   showed weights in VRAM. The actual problem was that Ollama's `/api/embed`
   path doesn't multi-stream regardless of `NUM_PARALLEL`. Two different
   bugs (wrong placement vs serialised inference) look the same from the
   client (low throughput). Diagnostic: check `size_vram == size` *and*
   probe concurrency separately.

3. **Claimed "warm vLLM restart in 15-30 seconds."** Actual ~2.5-3 minutes
   on a 14B model. torch.compile caches help, weight load and CUDA graph
   capture do not. Lesson: cache != skip.

4. **Asserted "qwen2.5 LLM refinement was the new bottleneck after fixing
   embedding."** Wrong — `mempalace mine` doesn't use the LLM at all (only
   `mempalace init` does). The real reason mytools mine was slower per file
   than CG is just bigger files. Lesson: read the code path before
   speculating about bottlenecks.

5. **Set `--max-model-len 8192` thinking it was generous.** It was for
   mempalace entity-refinement prompts (~500 tokens each). It wasn't for
   llm_wiki document ingest (4K+ token inputs). Forced a vLLM restart to
   bump it to 32K. Lesson: when in doubt for `--max-model-len`, take the
   model's native max — KV cache cost is real but the unified-memory Spark
   has plenty of headroom.

---

## Open questions / not yet answered

- **What is Qwen2.5-14B-AWQ aggregate throughput at 16-32 concurrent
  sequences in practice?** We computed the theoretical ceiling but never
  ran the probe. Easy follow-up.

- **Speculative decoding actual gain on Spark.** Speculative decoding
  works best when the draft model is a faithful approximation of the big
  model. For Qwen2.5 family, would `Qwen2.5-0.5B-Instruct` give clean
  ~1.5× speedup? Worth measuring.

- **vLLM with a small embedding model (nomic, 137M) at 47 ms per call —
  is that compute-bound or framework overhead?** A 137M model should be
  much faster than 47 ms on Blackwell. Possibilities: tiny models pay
  fixed-cost per request that big models amortise, or there's a vLLM
  scheduler overhead per call. Could probe by trying with batch sizes
  inside one request vs many requests.

- **Does pairing vllm-embed and vllm-chat in the same process (instead
  of two containers) save anything?** Both share the GPU; would a single
  process with both models give better KV cache scheduling, or just
  complicate things? Two containers is simpler and we know it works.

- **Does the AWQ-Marlin repack at every load have a cache?** `_compile_cache`
  exists for the graph; not clear if there's an analogous AWQ-cache. If not,
  this is the dominant warm-restart cost and worth fixing upstream.

---

## TL;DR

- DGX Spark is a wide-and-slow box. Your apps need to fan out or you waste
  the hardware.
- Ollama is the wrong tool for sustained throughput on this hardware. Use
  it for casual model rotation only.
- vLLM is the right tool but requires up-front budgeting of VRAM and
  context length — restart cost is high.
- Single-sequence decode on a 14B model ≈ 15 tok/s no matter what you do.
  Aggregate at concurrency-16 ≈ 190 tok/s. The 13× gap is where all the
  optimization lives.
- Always measure end-to-end. Diagnostic surfaces (nvidia-smi, /api/ps,
  /v1/models) lie about what's actually happening; only client-side
  throughput probes reveal real behaviour.
