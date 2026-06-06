# Current DGX Spark Setup

**Current `vllm-chat` model ids** (copy-paste for client configs):

```
spark1 (192.168.1.147:8001):  Qwen/Qwen3.5-122B-A10B-FP8
spark2 (192.168.1.69:8001):   Qwen/Qwen3.5-122B-A10B-FP8
```

> **⚠️ ACTIVE CROSS-BOX EXPERIMENT (2026-06-05): both boxes are running
> one model together, production single-box slots are DOWN.** spark1 +
> spark2 are a 2-node Ray cluster serving (currently)
> **`Qwen/Qwen3.5-122B-A10B-FP8`** (122B total / 10B active, hybrid
> Gated-DeltaNet + gated-attention MoE, FP8) tensor-parallel **TP=2**
> over the RoCE cable (container `vllm-2box` on each box, image
> `local/vllm-ray:26.05`, port 8001 on spark1 =
> `http://192.168.1.147:8001/v1`, parsers
> `--tool-call-parser qwen3_coder --reasoning-parser qwen3`,
> `--max-model-len 131072`). Brought up by
> `PROFILE=qwen35 ./spin-up-vllm-2box-rdma.sh`.
> **NCCL transport: RoCE/IB verbs** — containers run with
> `--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1`
> and env `NCCL_IB_HCA=rocep1s0f0:1`, `NCCL_IB_GID_INDEX=3` (RoCE v2 /
> IPv4); NCCL log confirms `NET/IB : Using [0]rocep1s0f0:1/RoCE`.
> While this runs, the normal single-box slots below are **stopped**:
> spark1 `vllm-chat` (Qwen3-Next-80B) + `vllm-embed`, and spark2
> `vllm-chat` (Nemotron). **Clients on spark1:8001 now get
> `Qwen/Qwen3.5-122B-A10B-FP8`** — not Qwen3-Next-80B; set the client
> model id there accordingly or expect a 400.
> Measured (Qwen3.5-122B-FP8, RoCE/IB, single-stream): **decode ~20 tok/s**
> (18.9–21.5, 3 samples after warm-up, 256-tok forced gens) — the
> **fastest cross-box decode measured on this rig**: +~57% over the same
> model on TCP sockets (12.7 tok/s) and ahead of MiniMax-M2.7-NVFP4 on
> RoCE (15.5 tok/s). Prefill not re-measured this run (was ~860 tok/s on
> sockets; compute-bound, transport-insensitive). **Why Qwen3.5 here, not
> MiniMax:** the `qwen3` reasoning parser keeps the trace OUT of `content`
> (verified — no llm_wiki think-leak), and there's no MiniMax
> path-corruption bug (`file.md`→`file .md`, GH anomalyco/opencode#25690).
> Tool-calling PASS (`qwen3_coder`: `get_weather`/`{"location":"Paris"}`).
> Full record + exact commands: `qwen35-122b-2box-observations.md`.
> **Swap the model in this slot:** `PROFILE=minimax ./spin-up-vllm-2box-rdma.sh`
> (back to MiniMax-M2.7-NVFP4) — the `PROFILE` knob selects model + parsers.
> **Revert NCCL transport to sockets** (cluster stays cross-box):
> `RDMA=0 PROFILE=qwen35 ./spin-up-vllm-2box-rdma.sh`.
> **Revert to single-box production:** `ssh spark 'docker rm -f vllm-2box'`,
> `ssh spark2 'docker rm -f vllm-2box'`, then bring up the single-box
> slots per §8 (embed → chat on spark1; chat on spark2).

Snapshot of what's actually running on **both** DGX Sparks as of
2026-06-06 (single-box steady state; see the cross-box banner above for
the current live override). Use this as a "rebuild from scratch"
reference if either box wipes, or as inventory when debugging.

> **Two-box layout.** `spark1` is the primary box that backs production
> LLM clients (MemPalace, llm_wiki, CampaignGenerator, opencode). It
> runs `vllm-embed` (port 8000), `vllm-chat` (port 8001), and Ollama
> (port 11434, mostly idle). `spark2` is the experimental sidecar: it
> runs models that are incompatible with the primary clients (e.g.
> reasoning models that emit `<think>` traces llm_wiki can't strip).
> Only spark1 is wired into production workflows; spark2 is for
> opencode sandboxing and side-by-side comparison.

> **Fast interconnect up (2026-06-04): direct spark1↔spark2 200 GbE
> cable.** A QSFP Direct Attach Copper cable now links the two boxes'
> ConnectX-7 ports directly (`enp1s0f0np0`, RDMA device `rocep1s0f0`,
> RoCEv2). **IP scheme changed 2026-06-05:** the NVIDIA *sync-cluster*
> tool re-IP'd the cable to **spark1 10.100.16.1 / spark2 10.100.16.2**
> (`enp1s0f0np0`) and **10.100.17.1/.2** (`enP2p1s0f0np0`), MTU 9000
> preserved, via `/etc/netplan/99-nvidia-sync-cluster.yaml` — and it
> **disabled our `99-fastlink.yaml`** (renamed `.sync-disabled-*`). The
> old `192.168.100.x` addresses are gone; use `10.100.16.x`. Measured RDMA
> bandwidth **~110 Gb/s per port**
> (`ib_write_bw`: 109 single-QP, 112 at 8 QPs × 1 MB) — a hard
> ~56%-of-line-rate ceiling more QPs don't lift (per-port
> PCIe/host-bridge limit on GB10, ~14 GB/s). **This cable is now LIVE
> service traffic** — the cross-box `vllm-2box` slot (see banner above)
> runs its TP=2 NCCL all-reduce over it via RoCE/IB verbs
> (`NCCL_IB_HCA=rocep1s0f0:1`, GID 3). Contrary to the original "marginal
> for tensor-parallel" worry, **TP=2 works fine here** because direct DAC
> point-to-point RoCE keeps the per-token all-reduce *latency* low (the
> real decode bottleneck), not because bandwidth is plentiful; the +57%
> Qwen3.5 decode gain from sockets→RoCE is that latency win. PP=2 remains
> the obvious next experiment for decode. A second ConnectX port
> (`roceP2p1s0f0`) is also RoCE-ACTIVE (possible socket-direct second
> PCIe path, or a second cable) — untested bonding headroom toward
> 200 G. See Hardware → Fast interconnect below.

> **Active change (2026-05-30): TurboQuant KV cache + vLLM 0.22.0 on spark1.**
> `vllm-chat` still serves the SAME model
> (`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`, same FP8 *weights*, same 128K
> context, same hermes tool parser) — the only functional change is the
> **KV cache dtype: `fp8` → TurboQuant `turboquant_k8v4`** (FP8 keys +
> 4-bit values). TurboQuant is a KV-cache quantization scheme, NOT a
> weight quantization, so the model id is unchanged. The image was also
> pinned from `:latest` (was vLLM 0.21.0) to
> **`vllm/vllm-openai:v0.22.0-aarch64`** (vLLM 0.22.0, CUDA 13.0, torch
> 2.11). Driven by `bash ~/spin-up-vllm-qwen3-next-80b-turboquant.sh`.
> Engine log confirms `Using TURBOQUANT attention backend`. Smoke +
> tool-call (hermes) both PASS, output coherent. **No client config
> change needed** — the served model id is identical, so MemPalace /
> llm_wiki / CampaignGenerator / opencode keep working unchanged.
>
> **Why 0.22.0 specifically (not just the user's ">=0.20.0"):**
> hybrid-model TurboQuant support (PR #39931) shipped in 0.21.0, but the
> Qwen3-Next *degenerate-output-under-CUDA-graph* bug (#40880) was only
> fixed in 0.22.0 — running TurboQuant on 0.21.0 risks silent garbage.
> Open bug #40807 (spec-decode path) doesn't apply (no spec decode here).
> Open bug #41726 (crash on large chunked continuation prefill) is
> guarded with `--max-num-batched-tokens 4096`. Ampere bug #40124 is
> irrelevant — the Spark is Blackwell sm_121 (SM>=89).
>
> **Caveat logged at startup:** *"TurboQuant is not yet compatible with
> FlashAttention >= 3 → overriding flash_attn_version to 2"* — the
> full-attention layers run on FA2, not FA3.
>
> **Honest tradeoff (calibration note):** on a hybrid like Qwen3-Next
> only the periodic full-attention layers carry KV (the Gated DeltaNet
> layers have none), so TurboQuant's absolute memory win is small while
> its compute overhead (Hadamard rotation + FA2 fallback) is full and
> lands on the prefill path. vLLM's own study rates plain fp8 KV the
> better default. Expect this to be a touch SLOWER than fp8 with a modest
> memory saving — this is a "feel the tradeoff" calibration choice, not
> an optimization. **Instant revert to plain fp8 KV:**
> `bash ~/spin-up-vllm-qwen3-next-80b.sh`.
>
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
> **Nemotron 3 Nano on spark2 (2026-05-26).** After empirical
> calibration the user found DeepSeek R1 distill Qwen 32B AWQ
> underwhelming for programming and swapped spark2's `vllm-chat` to
> `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`. This is the same model
> that was rejected from **spark1** (2026-05-18 → 05-19) because
> llm_wiki can't strip `<think>` traces — but spark2 is the
> experimental sidecar, not wired into llm_wiki, so the rejection
> doesn't apply here. The opencode reasoning-trace leak is unchanged
> from the previous DeepSeek slot occupant (see §4 leak warning).
> Earlier history (Nemotron Phase A/B on spark1) is captured in
> `nemotron3-nano-30b-observations.md` and
> `nemotron3-nano-30b-test-plan.md`.

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

### Fast interconnect (spark1 ↔ spark2 direct cable, 2026-06-04)

Point-to-point link, **separate from the LAN**. **Now carrying live
service traffic**: the cross-box `vllm-2box` slot (banner above) runs its
TP=2 NCCL all-reduce over this cable via RoCE/IB verbs.

| field | value |
|---|---|
| medium | QSFP Direct Attach Copper, **200 GbE** negotiated |
| NIC / port | ConnectX-7 `enp1s0f0np0` (RDMA dev `rocep1s0f0`, RoCEv2), both boxes |
| addressing | spark1 `10.100.16.1` / spark2 `10.100.16.2`, /24, **MTU 9000** (NVIDIA sync-cluster tool re-IP'd from `192.168.100.x` on 2026-06-05; 2nd port `10.100.17.1/.2` on `enP2p1s0f0np0`) |
| persistence | `/etc/netplan/99-nvidia-sync-cluster.yaml` on each box (the old `99-fastlink.yaml` was renamed `.sync-disabled-*` by the sync tool) |
| measured BW | **~110 Gb/s/port** RDMA (`ib_write_bw`); ~56% of line rate, QP-count-insensitive (~14 GB/s) |
| second port | `enP2p1s0f0np0` / `roceP2p1s0f0` also RoCE-ACTIVE — untested bonding headroom |
| NCCL pinning | `NCCL_IB_HCA=rocep1s0f0:1`, `NCCL_IB_GID_INDEX=3` (RoCE v2 / IPv4); OOB bootstrap on `enp1s0f0np0`. Log: `NET/IB : Using [0]rocep1s0f0:1/RoCE` |
| current use | **live** cross-box vLLM **TP=2** (Qwen3.5-122B-FP8); RoCE/IB transport gave +57% decode vs TCP sockets. PP=2 still the next decode experiment |

The LAN (`192.168.1.0/24`) carries all SSH and every *client→vLLM*
request; the cable carries only the *inter-node* TP all-reduce for the
cross-box slot. When the cross-box slot is torn down (back to single-box),
the cable goes idle again. See `qwen35-122b-2box-observations.md` for the
full cross-box recipe and `todo_minimax_m27_two_box` in memory.

To re-create after a wipe: the cable is normally re-IP'd by the NVIDIA
sync-cluster tool to `10.100.16.1/.2` (`/etc/netplan/99-nvidia-sync-cluster.yaml`).
If doing it by hand instead, assign the IPs + MTU via single-line
`netplan set ...` one-liners (paste-safe — hand-written heredoc/printf
YAML gets mangled by terminal auto-indent), then `sudo netplan apply`.
Verify with a jumbo ping (`ping -M do -s 8972 10.100.16.2`) and
`ib_write_bw -d rocep1s0f0`. (Test the real TCP path too — ping + RDMA
both pass on a stale IP config.)

## Ports in use

### spark1 (192.168.1.147)

| port | service | purpose |
|---:|---|---|
| 11434 | Ollama (systemd) | LLM serving (legacy, kept for fallback) |
| 8000 | vllm-embed (docker) | Embeddings — `nomic-embed-text-v1.5` |
| 8001 | vllm-chat (docker) | Chat completions — `Qwen3-Next 80B A3B Instruct FP8`, 128K context, hybrid attention, **TurboQuant KV (`turboquant_k8v4`)**, vLLM 0.22.0, tool calling on |

### spark2 (192.168.1.69)

| port | service | purpose |
|---:|---|---|
| 8001 | vllm-chat (docker) | Chat completions — `Nemotron 3 Nano 30B A3B BF16`, 256K context, hybrid Mamba-2/MoE, reasoning + tool calling on |

(No `vllm-embed`, no Ollama — spark2 is single-container.)

## VRAM budget (steady state)

### spark1

| service | reserved cap | actual model size | notes |
|---|---:|---:|---|
| vllm-embed | ~6 GB (0.05 × 128) | ~600 MB | KV cache fits in cap |
| vllm-chat | ~107 GB (0.88 × ~121.7 GiB) | ~80 GiB FP8 weights + KV cache (**TurboQuant `turboquant_k8v4`**: FP8 keys + 4-bit values, on the full-attention layers only) + activations @ 128K context | Hybrid attention: most layers are Gated DeltaNet (no KV), only periodic full-attention layers carry KV — so TurboQuant only compresses those few layers; absolute KV win over plain fp8 is small. (nvidia-smi reports `[N/A]` for memory.used on this GB10/WSL box, so resident bytes aren't directly measurable here.) |
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
| vllm-chat | ~102 GB (0.80 × ~128 GB) | ~60 GiB BF16 weights + Mamba-2 state + KV cache for the 6 attention layers @ 256K context + activations | Hybrid Mamba-2/MoE: only 6 of 52 layers carry traditional KV cache, so 256K context is affordable even on BF16. 0.80 budget chosen because the recipe doesn't specify and 60 GiB weights + Mamba state needs headroom — drop to 0.75 if OOM. |

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
~3B active per token, ~80B total) with **TurboQuant KV cache
(`turboquant_k8v4`) on vLLM 0.22.0** (image
`vllm/vllm-openai:v0.22.0-aarch64`). Slot history: Qwen 2.5 14B AWQ →
Llama 3.3 70B AWQ + spec-decode → Gemma 4 26B MoE → Gemma 4 26B MoE
longctx → **Nemotron 3 Nano 30B A3B (2026-05-18 to 2026-05-19, rejected
after Phase B — see `nemotron3-nano-30b-observations.md`)** → Gemma 4
26B MoE longctx → **Qwen3-Next 80B A3B Instruct FP8, plain fp8 KV
(2026-05-21 → 2026-05-30)** → **Qwen3-Next 80B A3B Instruct FP8,
TurboQuant `turboquant_k8v4` KV + vLLM 0.22.0 (2026-05-30 → current)**.
vllm-chat swap-in scripts are
`spin-up-vllm-qwen3-next-80b-turboquant.sh` (current — TurboQuant KV,
v0.22.0),
`spin-up-vllm-qwen3-next-80b.sh` (plain fp8 KV — instant revert),
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

Currently launched via `./spin-up-vllm-qwen3-next-80b-turboquant.sh`.
Effective command:

```bash
docker run -d --runtime nvidia --gpus all \
  --name vllm-chat \
  -p 8001:8001 \
  --ipc=host \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.22.0-aarch64 \
  Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --max-model-len 131072 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype turboquant_k8v4 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 0.0.0.0 --port 8001
```

The prior plain-fp8 command (revert target) is identical except
`vllm/vllm-openai:latest` + `--kv-cache-dtype fp8` and no
`--max-num-batched-tokens` flag — i.e. `spin-up-vllm-qwen3-next-80b.sh`.

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
- **`vllm/vllm-openai:v0.22.0-aarch64`** (vLLM 0.22.0, CUDA 13.0, torch
  2.11): pinned, NOT `:latest`. TurboQuant hybrid support landed in
  0.21.0 (#39931) but the Qwen3-Next degenerate-output-under-CUDA-graph
  bug (#40880) was only fixed in 0.22.0 — so 0.22.0 is the floor for
  *this* model. `:latest` happens to point at 0.22.0 right now but will
  drift; the pin keeps this doc honest. The Spark is aarch64 (Grace);
  the image's `arch_list` is `sm_80…sm_120` (no explicit sm_121, but
  sm_120 cubins run on GB10/sm_121 via CUDA 12.x minor-version forward
  compat — proven by months of prod on this box).
- **`--kv-cache-dtype turboquant_k8v4`**: TurboQuant KV-cache quant —
  FP8 keys + 4-bit values (~2.6× on the compressed full-attention
  layers, +1.17% PPL per vLLM's published numbers). Closest analogue to
  the prior plain `fp8` KV, chosen so a future A/B isolates TurboQuant's
  machinery cost rather than an accuracy cliff. Other presets:
  `turboquant_4bit_nc` (3.8×, +2.71%), `turboquant_k3v4_nc` (3.5×,
  +10.63%), `turboquant_3bit_nc` (4.9×, +20.59%). Set `KV_CACHE_DTYPE=fp8`
  (or just run the plain script) to revert.
- **`--max-num-batched-tokens 4096`**: guard for open bug #41726 (crash
  on large chunked continuation prefill with TurboQuant). If it still
  crashes mid-prefill at long context, drop to 2048 and/or set
  `ENFORCE_EAGER=1`.
- **`--trust-remote-code`**: Qwen3-Next ships custom modeling code.
- **`--enable-auto-tool-choice` + `--tool-call-parser hermes`**:
  default Qwen3 chat-template parser. Verified end-to-end with
  `test-toolcall.sh` (PASS — null content + tool_calls[get_weather]
  + parseable arguments). Alternate: `TOOL_PARSER=qwen3_coder` (the
  stricter parser designed for Qwen3-Coder's tool format).

### Known perf ceilings (Spark-specific)

- **TurboQuant forces FlashAttention 2.** Startup logs:
  *"TurboQuant is not yet compatible with FlashAttention >= 3 →
  overriding flash_attn_version to 2."* The full-attention layers run
  on FA2, giving up FA3's throughput on exactly the layers TurboQuant
  touches.
- **TurboQuant overhead lands on a hybrid that barely needs it.** Only
  the periodic full-attention layers carry KV (GDN layers carry none),
  so the memory saved is small, but the Hadamard-rotation + dequant
  compute is paid in full, on the prefill-heavy path this box runs.
  Net expectation: slightly slower than plain fp8 KV. **Quality
  verified** 2026-05-30: long-context needle test (8K→120K, depths
  0.25/0.5/0.9) scored 12/12 PASS — recall intact, no #41726 crash. The
  fp8-vs-TurboQuant *speed* A/B is still pending. Full record:
  `turboquant-observations.md`.
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
serving **Nemotron 3 Nano 30B A3B BF16** — an NVIDIA hybrid
Mamba-2 / Transformer-MoE model with 30B total params, ~3.5B active
per token, 23 Mamba-2 + 23 MoE + 6 attention layers, native 256K
context. Reasoning is baked in: emits `<think>...</think>` blocks
that the `nano_v3` plugin strips out into a separate response field.

Slot history on spark2: Nemotron 3 Nano 30B A3B BF16 (2026-05-22 →
~2026-05-24) → DeepSeek R1 distill Qwen 32B AWQ (~2026-05-24 →
2026-05-26) → **Nemotron 3 Nano 30B A3B BF16 (2026-05-26 →
current)**. DeepSeek swapped out because the user found it
underwhelming for programming despite a full 32B working per token
under AWQ. Nemotron returns to the slot — same opencode reasoning
leak as DeepSeek (see warning below), but spark2 isn't wired into
llm_wiki, so the original Phase B blocker doesn't apply on this box.

### Run command

```bash
docker run -d --runtime nvidia --gpus all \
  --name vllm-chat \
  -p 8001:8001 \
  --ipc=host \
  -e HF_TOKEN="$HF_TOKEN" \
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

Driven by `spin-up-vllm-nemotron3-nano-30b.sh` (committed in this
repo). Run via:

```bash
scp spin-up-vllm-nemotron3-nano-30b.sh lib-vllm-spinup.sh spark2:~/
ssh spark2 'bash ~/spin-up-vllm-nemotron3-nano-30b.sh'
```

The script also downloads `nano_v3_reasoning_parser.py` from HF on
first run and drops it in `~/vllm-plugins/` on the spark2 host, then
mounts that directory read-only into the container at `/plugins`.

The script accepts `NEMO_MODEL` / `MAX_LEN` / `GPU_UTIL` / `MAX_SEQS`
/ `KV_CACHE_DTYPE` env overrides. To swap to the FP8 variant for
faster decode: `NEMO_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
KV_CACHE_DTYPE=fp8 bash ~/spin-up-vllm-nemotron3-nano-30b.sh`.

### Why these flags

- **`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`**: NVIDIA hybrid
  arch (Mamba-2 + MoE + 6 attention layers). Sm_121 (DGX Spark / GB10)
  is **officially supported** by NVIDIA's vLLM recipe for this model
  — the only model on either spark with vendor-tuned kernels for this
  exact hardware. BF16 chosen over FP8 for raw quality; FP8 variant
  is faster decode if needed.
- **`--max-model-len 262144`** (256K): recipe default. Only 6 of 52
  layers carry traditional KV cache (the rest are Mamba-2 constant
  state or MoE FFN), so 256K is affordable. Native ceiling is 1M
  with `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` — not enabled here.
- **`--max-num-seqs 8`**: recipe default. Double spark1's 4 because
  hybrid attention makes per-sequence KV cost ~10× cheaper.
- **`--gpu-memory-utilization 0.80`** (~102 GiB cap): empirical
  setting from the spin-up script. Not from the recipe — NVIDIA
  doesn't specify util in the published command. 60 GiB BF16 weights
  + Mamba state + 256K KV for the 6 attention layers fit, with
  ~25 GiB headroom. Drop to 0.75 if OOM at startup.
- **`--kv-cache-dtype auto`**: BF16 KV. The FP8 variant of the model
  pairs with `KV_CACHE_DTYPE=fp8` for additional KV savings.
- **`--dtype bfloat16`**: matches the model weights.
- **`--trust-remote-code`**: required — the model ships custom
  modeling code for the hybrid arch.
- **`--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder`**:
  NVIDIA reused the qwen3_coder tool-call format. There's an open HF
  discussion noting tool-call + reasoning is flaky in some configs
  (https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/discussions/3)
  — probe with `MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  ./test-toolcall.sh` after any restart before trusting it.
- **`--reasoning-parser-plugin /plugins/nano_v3_reasoning_parser.py`
  + `--reasoning-parser nano_v3`** *(key difference vs spark1)*:
  loads NVIDIA's custom plugin from the host-mounted `/plugins`
  directory and strips `<think>...</think>` blocks out of `content`.
- **`-e HF_TOKEN`**: passthrough so vLLM can pull the BF16 weights
  on first run (~60 GiB). The `.profile` export-keyword quirk applies
  on spark2 too — verify with a grandchild process if the token is
  missing.

### Reasoning-trace leak warning (verified 2026-05-26)

The `nano_v3` plugin routes the reasoning trace into a field literally
named `reasoning` (not the OpenAI-convention `reasoning_content`).
opencode's openai-compatible provider doesn't surface a custom
`reasoning` field, so trace tokens count against `completion_tokens`
but never display. Verified post-swap: a "Reply with only OK" probe
returned `content: "\nOK"` plus 40 silently-dropped trace tokens
under the `reasoning` key. This is the **same failure mode** the
previous DeepSeek R1 occupant had on this slot — swapping the
parser plugin didn't fix it because both NVIDIA's `nano_v3` and
vLLM 0.21's stock `deepseek_r1` parser pick the same non-standard
field name. Three fix paths are open (server-side rename, drop the
parser, or accept the leak); see memory `todo-nano-v3-reasoning-leak`.

### Smoke test

```bash
curl -sS http://192.168.1.69:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16","messages":[{"role":"user","content":"Reply with only the word OK."}],"max_tokens":256}'
# expect: {"choices":[{"message":{"content":"\nOK","reasoning":"...","reasoning_content":null,...}}],...}
# note: "reasoning" field (NOT reasoning_content) is populated; budget max_tokens accordingly.
```

### Measured behaviour

No benchmarks logged yet on spark2 for this swap. For comparison
against spark1's Qwen3-Next, see `model-comparisons.md` (todo —
currently spark1-only). Phase A perf data from the earlier spark1
Nemotron run is in `nemotron3-nano-30b-observations.md`, but the
spark1 box had a vllm-embed sidecar (port 8000) that doesn't exist
on spark2, so prefill/decode numbers don't transfer directly.

### Restart cost

- Cold start (first time, with HF download): measured 2026-05-26 —
  ~497 s (~8 min) to "Application startup complete" *with* warm HF
  cache (the previous 2026-05-22 Nemotron run on spark2 left the
  weights resident). A truly cold pull would add ~10–15 min of HF
  download for the ~60 GiB BF16 weights.
- Warm restart (cached weights, what `docker start vllm-chat` does
  post-reboot): expected ~3–5 min — shard load + `torch.compile`
  warmup dominate.

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

The direct spark1↔spark2 cable (`10.100.16.0/24` — see Hardware →
Fast interconnect) is **not** in this table: it carries no client-facing
endpoint. While the cross-box slot is up, its Ray/NCCL inter-node traffic
rides that subnet, pinned via `NCCL_SOCKET_IFNAME` (OOB bootstrap) /
`NCCL_IB_HCA=rocep1s0f0:1` (RoCE/IB data path). Clients still reach the
served model only through spark1 `192.168.1.147:8001` on the LAN.

---

## 7. Client-side configuration

> **No client changes for the 2026-05-30 TurboQuant swap.** TurboQuant
> is a server-side KV-cache dtype change only; the served model id
> (`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`) is identical, so every config
> below is unchanged from the 2026-05-21 fp8 deployment. Listed here as
> current-state inventory, not as edits made.

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
    },
    "dgx2": {
      "api": "openai",
      "name": "DGX Spark 2 (vLLM, experimental)",
      "options": {
        "baseURL": "http://192.168.1.69:8001/v1",
        "apiKey": "ignored"
      },
      "models": {
        "nemotron3-nano-30b": {
          "id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
          "name": "Nemotron 3 Nano 30B A3B BF16 @ 256K (reasoning+tools, spark2)",
          "limit": { "context": 262144, "output": 8192 },
          "tool_call": true,
          "reasoning": true,
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

The `nemotron3-nano-30b` entry **under the `dgx` (spark1) provider** is
preserved as an alternate but is **not the default** on spark1 after
Phase B testing rejected Nemotron there: llm_wiki has no parser for
`<think>` reasoning traces, so the chat box filled with raw thinking
(`nemotron3-nano-30b-observations.md`). The `opencode-spark-longctx.sh`
wrapper from Phase B still works for Gemma 4 if invoked with
`MODEL_ID=google/gemma-4-26b-a4b-it MIN_CTX=131072
./opencode-spark-longctx.sh` — but the bare `opencode` invocation
using `opencode.json` is the standard path.

opencode also has a second provider `dgx2` pointed at spark2
(`http://192.168.1.69:8001/v1`). Switch to a spark2 model by setting
the top-level `"model"` field to `dgx2/nemotron3-nano-30b`.

> **Reasoning-trace leak warning (verified 2026-05-26):** Nemotron's
> `nano_v3` reasoning parser plugin emits the trace under a field
> literally named `reasoning`, **not** `reasoning_content`. opencode's
> openai-compatible provider does not surface a custom `reasoning`
> field — so trace tokens are consumed silently against
> `completion_tokens` and opencode never displays them. This is the
> **same failure mode** the previous DeepSeek R1 occupant of this slot
> had with vLLM 0.21's stock `deepseek_r1` parser: both pick the
> non-standard field name. A post-swap "Reply with only OK" probe
> returned `content: "\nOK"` plus 40 dropped trace tokens under
> `reasoning`. Budget your `max_tokens` accordingly (~1.5–2× what
> you'd give a non-reasoning model). The `reasoning: true` flag in
> the opencode entry is documentary; it does not unlock display in
> this version.

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
(80 GB weights), spark2 vllm-chat ~3–5 min (60 GB BF16 weights).

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
   FP8 + TurboQuant KV on vLLM 0.22.0 is the current default — 128K
   context, hybrid attention):
   ```bash
   scp spin-up-vllm-qwen3-next-80b-turboquant.sh spin-up-vllm-qwen3-next-80b.sh \
       lib-vllm-spinup.sh test-toolcall.sh spark:~/
   ssh spark 'docker pull vllm/vllm-openai:v0.22.0-aarch64'
   ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b-turboquant.sh'
   ```
   Expect ~40 min on first run (HF pulls ~80 GB of FP8 weights on a
   fresh box; ~13 min observed warm-cache on 2026-05-30 — shard load +
   torch.compile dominate). Script waits for `Application startup
   complete` and smoke-tests on its own. Then verify tool calling
   (`HOST`/`PORT`, not `DGX_*`):
   `ssh spark 'MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 ~/test-toolcall.sh'`.
   If the smoke output is repeated/garbled, that's bug #40880 — re-run
   with `ENFORCE_EAGER=1`. To revert to plain fp8 KV or swap models, see §9.

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

4. **Start vllm-chat** with the Nemotron 3 Nano block from §4 above.
   Expect ~10–20 min on first run (HF pulls ~60 GB of BF16 weights;
   the spin-up script also wgets `nano_v3_reasoning_parser.py` into
   `~/vllm-plugins/` on the host before mounting it). Driven by
   `bash ~/spin-up-vllm-nemotron3-nano-30b.sh` (committed; scp it
   along with `lib-vllm-spinup.sh`).

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

# CROSS-BOX slot (CURRENTLY LIVE — both boxes, one model, TP=2 over RoCE).
# Run from the WORKSTATION (it SSHes to spark + spark2). PROFILE picks the
# model + parsers; RDMA=1 (default) = RoCE/IB transport.
PROFILE=qwen35  ./spin-up-vllm-2box-rdma.sh   # Qwen3.5-122B-A10B-FP8 @ 128K (CURRENT)
PROFILE=minimax ./spin-up-vllm-2box-rdma.sh   # nvidia/MiniMax-M2.7-NVFP4 @ 64K
RDMA=0 PROFILE=qwen35 ./spin-up-vllm-2box-rdma.sh  # same, revert transport to TCP sockets
# Tear the cross-box slot down and return to the single-box scripts below:
ssh spark 'docker rm -f vllm-2box'; ssh spark2 'docker rm -f vllm-2box'

# SINGLE-BOX vllm-chat swaps on port 8001 (one-liner each). These are the
# steady-state slots — all STOPPED while the cross-box slot above is live.
ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b-turboquant.sh' # Qwen3-Next 80B FP8 + TurboQuant KV @ 128K, vLLM 0.22.0 (single-box default)
ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b.sh'         # Qwen3-Next 80B FP8, plain fp8 KV @ 128K (revert target)
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
