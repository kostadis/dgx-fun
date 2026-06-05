# Qwen3.5-122B-A10B-FP8 across two Sparks (TP=2) — observations

Append-only experiment log. First cross-box (multi-node) vLLM deployment
on the spark1+spark2 pair, validating the Ray + distributed-vLLM harness
over the 200 GbE RoCE cable. Plan: `~/.claude/plans/hashed-riding-tulip.md`.

## Result (2026-06-05)

**It works.** `Qwen/Qwen3.5-122B-A10B-FP8` (122B total / 10B active, hybrid
Gated-DeltaNet + gated-attention MoE, 256K-capable) serves on
`http://192.168.1.147:8001/v1`, sharded **tensor-parallel TP=2** across both
Sparks via a Ray cluster over the direct cable. Smoke test coherent,
**tool-calling PASS** (`qwen3_coder` parser, `get_weather`/`{"location":"Paris"}`).

### Measured (first run, max-model-len 131072, gpu-util 0.85)

| metric | value | note |
|---|---|---|
| **Prefill** | **~860 tok/s** | 4520-token prompt in 5.3 s; compute-bound, batches cross-node comms well |
| **Decode** | **~12.7 tok/s** | every token pays a TP all-reduce round-trip over the cable — latency-bound, slow |
| KV cache free | 39.9 GiB/rank | hybrid attention → only the gated-attn layers carry KV; plenty of room |
| Weight load | ~15 min | 61 GB/box FP8 off local disk (slow) |
| Engine init | 370 s | incl. 182 s torch.compile + CUDA-graph capture |

**Takeaway:** cross-box TP=2 is the bandwidth/latency-starved case the memory
note predicted. Prefill is usable; decode is ~an order of magnitude below what
a single-box model gives. Correct for "fits only across two boxes," wrong for
"fast." A PP=2 variant (less frequent cross-node traffic, at the cost of
pipeline bubbles) is the obvious next calibration experiment for decode.

## The setup that actually worked (after the false starts below)

1. **Weights** (both boxes, ~122 GB each): `snapshot_download` via the vLLM
   image's `huggingface_hub` (caches are per-box, not shared). Public/ungated
   — no HF_TOKEN needed.
2. **Image**: stock `nvcr.io/nvidia/vllm:26.05-py3` **has Qwen3.5 support**
   (`Qwen3_5MoeForConditionalGeneration` in the registry) but **does NOT ship
   Ray**. Built a thin derived image on both boxes:
   ```dockerfile
   FROM nvcr.io/nvidia/vllm:26.05-py3
   RUN pip install --no-cache-dir "ray[default]"
   ```
   `docker build -t local/vllm-ray:26.05 .`
3. **Cable IPs**: the NVIDIA **sync-cluster** tool (run ~03:03Z) had re-IP'd the
   cable to `10.100.16.1/.2` (enp1s0f0np0) and `10.100.17.1/.2`
   (enP2p1s0f0np0), MTU 9000 preserved, and **disabled our `99-fastlink.yaml`**
   (renamed `.sync-disabled-*`, replaced by `99-nvidia-sync-cluster.yaml`).
   Use `10.100.16.x`, not the old `192.168.100.x`.
4. **Ray cluster** (containers `--network host --shm-size 10.24g`, env pinned to
   `enp1s0f0np0`):
   - Head (spark1): `ray start --head --node-ip-address=10.100.16.1 --port=6379 --block`, `VLLM_HOST_IP=10.100.16.1`
   - Worker (spark2): `ray start --address=10.100.16.1:6379 --node-ip-address=10.100.16.2 --block`, `VLLM_HOST_IP=10.100.16.2`
   - env both: `NCCL_SOCKET_IFNAME=GLOO_SOCKET_IFNAME=TP_SOCKET_IFNAME=UCX_NET_DEVICES=OMPI_MCA_btl_tcp_if_include=enp1s0f0np0`, `MASTER_ADDR=10.100.16.1`, `RAY_memory_monitor_refresh_ms=0`
   - `ray status` → **2 nodes / 2.0 GPU / 223.7 GiB**.
5. **Serve** (from inside head container, detached):
   ```
   vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
     --tensor-parallel-size 2 --distributed-executor-backend ray \
     --max-model-len 131072 --gpu-memory-utilization 0.85 \
     --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
     --trust-remote-code --host 0.0.0.0 --port 8001
   ```

## Gotchas / friction (the calibration meat)

- **RDMA hid a config break.** Ping (ICMP) and `ib_write_bw` (RDMA, bypasses the
  kernel IP stack) both worked on a stale IP, so the cable looked fine — but
  Ray's TCP to the GCS timed out (errno 11) from everywhere. Root cause wasn't a
  firewall; it was that the cable IP had silently changed (NVIDIA sync tool).
  Lesson: **test the actual TCP path, not just ping/RDMA, after any network
  change.**
- **Ray GCS is slow to bind cold (~55 s)** on this box, and binds IPv6 `::`. The
  5 s "Failed to connect to GCS" warnings during startup are transient — wait a
  full minute before judging.
- **NGC vLLM image ≠ multi-node ready** — Ray must be added (see image step).
- **Reasoning field is `reasoning`, not `reasoning_content`** (same as the
  Nemotron/DeepSeek slots). Clients that only surface `reasoning_content` (e.g.
  opencode) will silently drop the trace. Default thinking is ON; pass
  `chat_template_kwargs={"enable_thinking": false}` for terse answers.
- **vLLM warns** TP is spread across nodes "unless you have fast interconnect
  like Infiniband" — expected; our RoCE cable is that interconnect.
- This build pulled in **multimodal/vision components** (`Qwen2VLImageProcessor`,
  MM warmup) — the served slot is text+tools here, but the arch carries vision.

## 2026-06-05 (later): swapped slot to MiniMax-M2.7-NVFP4 — **NVFP4 works on sm_121**

Same harness, model swapped to **`nvidia/MiniMax-M2.7-NVFP4`** (230B/10B, the
original cable-project target). **Headline: NVFP4 runs on the GB10 (sm_121).**
vLLM detected the ModelOpt NVFP4 checkpoint (`quantization=modelopt_fp4`),
loaded ~62.75 GiB/GPU, **autotuned the TensorRT-LLM FP4 MoE GEMMs**
(`trtllm::fused_moe::gemm1/2` via flashinfer — some sm_100-specific tactics
"skipped" but working ones found), compiled, and served. No FP4-kernel error.
This **unblocks the whole NVFP4 frontier tier** (DeepSeek-V4-Flash, Qwen3.5-397B).

| metric | MiniMax-M2.7-NVFP4 (TP=2) | Qwen3.5-122B-FP8 (TP=2) |
|---|---|---|
| Decode | ~11.1 tok/s | ~12.7 tok/s |
| Prefill | ~744 tok/s | ~860 tok/s |
| Weights/GPU | ~62.75 GiB | ~61 GiB |
| KV (auto fp8_e4m3) | 36.8 GiB free, 621k tokens | 39.9 GiB free |
| Tool calls | ✅ correct fn+args | ✅ |

- **NVFP4 ≈ FP8 speed here** — the cross-box cable dominates decode, so NVFP4's
  compute (native FP4 vs dequant — couldn't cleanly tell, the "skipped tactics"
  suggest a partial/emulated path) can't show through. NVFP4's real win is
  **fitting 230B**, not speed.
- **Reasoning leaks into `content`**: the recommended `minimax_m2_append_think`
  parser keeps raw `<think>…` in the content field (smoke started with
  `<think>The user wants…`). So MiniMax+this parser **would re-trip the llm_wiki
  thinking-leak** that Qwen3.5's `qwen3` parser avoided. Tool calls still return
  correct fn+args; the think text just rides alongside in content.
- Run command: `vllm serve nvidia/MiniMax-M2.7-NVFP4 --tensor-parallel-size 2
  --distributed-executor-backend ray --tool-call-parser minimax_m2
  --reasoning-parser minimax_m2_append_think --enable-auto-tool-choice
  --trust-remote-code --gpu-memory-utilization 0.85 --max-model-len 65536`.

## 2026-06-05 (later still): NCCL TCP sockets → RoCE/IB verbs — decode +40%

Switched the cross-box TP=2 all-reduce from TCP sockets to **RDMA/RoCE over
the cable**, same MiniMax-M2.7-NVFP4 slot. Rebuilt both `vllm-2box`
containers via `spin-up-vllm-2box-rdma.sh` (workstation-driven).

**Why containers had to be recreated, not just re-env'd:** the socket build
exposed **no RDMA** to the container — `Devices=` empty, no `IPC_LOCK` cap,
no `memlock` ulimit. NCCL couldn't have opened `/dev/infiniband` even with
the right env. Added at `docker run`:
`--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1`.

**Cable RDMA facts** (both boxes, identical): HCA for `enp1s0f0np0` =
`rocep1s0f0`, port 1 `PORT_ACTIVE`, `link_layer Ethernet` (= RoCE),
`active_mtu 4096`. RoCE v2 GID = **index 3** (IPv4-mapped `…ffff:0a64:1001`
= 10.100.16.1). Env set on both: `NCCL_IB_HCA=rocep1s0f0:1`,
`NCCL_IB_GID_INDEX=3`, `NCCL_IB_DISABLE=0`; kept `NCCL_SOCKET_IFNAME=
enp1s0f0np0` for OOB bootstrap.

**Proof it actually used IB** (NCCL log, both nodes):
`NCCL INFO NET/IB : Using [0]rocep1s0f0:1/RoCE [RO]; OOB enp1s0f0np0:10.100.16.1`
— data path on IB verbs, only the bootstrap on the socket. (Benign:
`ncclTunerPlugin_v2 … undefined symbol` — the optional SHARP *tuner* plugin
fails to load; the RDMA *net* plugin loads fine.)

| metric | sockets (prior) | RoCE/IB (now) |
|---|---|---|
| Decode | ~11.1 tok/s | **~15.5 tok/s** (15.4–16.0, 4 samples, 256-tok gens) |

**~+40% decode**, confirming the bottleneck was per-token all-reduce
**latency** on the TCP path, not cable bandwidth. No PFC needed — direct DAC
point-to-point (no switch), so RoCE v2 doesn't congest.

**Gotcha:** model load is slow (~13–15 min: 15 safetensors shards ~50s each
+ FP4 autotune). The first spin-up script run exited 1 because its
endpoint-wait window (600s) was shorter than load; `vllm serve` is detached
so it kept loading and came up fine. Script's wait bumped accordingly.

**Revert transport to sockets:** `RDMA=0 ./spin-up-vllm-2box-rdma.sh`.

## Open follow-ups

- PP=2 variant for decode (vs TP=2). Confirm vLLM PP support for the GDN hybrid first.
- Raise `--max-model-len` toward 262144 (KV headroom is large).
- ~~Try NCCL over RDMA/IB-verbs vs TCP socket path~~ — **done 2026-06-05, +40% decode (above).**
- The second cable port (`enP2p1s0f0np0` / `10.100.17.x`) is unused — bonding/2-rail experiment. RoCE v2 there too, but MTU 1024 vs 4096; would need MTU fix first.
- NCCL over RDMA could go further: try `NCCL_IB_GID_INDEX` auto vs 3, GPUDirect RDMA (`NCCL_NET_GDR_LEVEL`), and 2-rail `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0`.
