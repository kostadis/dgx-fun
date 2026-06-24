# Current DGX Spark Setup

**Current `vllm-chat` model ids** (copy-paste for client configs):

```
spark1 (192.168.1.147:8001):  Qwen/Qwen3-Next-80B-A3B-Instruct-FP8  (vllm-chat, single-box TP=1, 256K ctx, util 0.80, seqs 16, NO spec decode (plain), default chunked prefill, vLLM 0.22.0)  ← THROUGHPUT
spark2 (192.168.1.121:8001):  Qwen/Qwen3-Next-80B-A3B-Instruct-FP8  (vllm-chat, single-box TP=1, 256K ctx, util 0.80, seqs 16, NO spec decode (plain), default chunked prefill, vLLM 0.22.0)  ← THROUGHPUT (matches spark1)
spark2 (192.168.1.121:8000):  Qwen/Qwen3-Embedding-0.6B  (vllm-embed, unchanged)
```
> **ctx = total context per request (prompt + generation), i.e. `--max-model-len`.** Both boxes 256K, both now on the plain build with vLLM's default chunked prefill (no `--max-num-batched-tokens` pin). The earlier no-chunked-prefill-at-256K attempt wedged spark2's host twice — default chunked prefill bounds the warmup batch and avoids that.

> **▶ LIVE (2026-06-23, latest): BOTH boxes on the THROUGHPUT config (plain, NO spec decode, seqs 16). A/B concluded — throughput config won (+47%); spark2 moved to match spark1 to serve a batch document-conversion job.**
> spark2's `vllm-chat` was swapped MTP-2/seqs-4 → **plain (no `--speculative-config`),
> `--max-num-seqs 16`** via `spin-up-vllm-qwen3-next-80b.sh` (`IMAGE`
> `v0.22.0-aarch64`, 256K, util 0.80, fp8 KV, hermes), so both boxes are now
> identical. `vllm-embed` (port 8000) left running on spark2 throughout.
> - **Why both:** the workload is batch conversion of several hundred documents —
>   total throughput matters, per-stream latency doesn't. The A/B (below) measured
>   spark1-throughput ~154 t/s @ 15 running / 0 waiting vs spark2-latency ~105 t/s
>   @ 4 running / queue growing, on the same job → **+47% aggregate** plus spark2
>   couldn't keep up (growing queue). So spark2 was moved onto the same config and
>   the doc set can be split across both endpoints (~2× the boxes).
> - **Verified spark2:** v0.22.0, `speculative_config=None`, seqs 16, KV pool
>   1.71M tok → **6.51× @ 256K**, coherent. Host RAM recovered **5 GB → 11 GB**
>   available after dropping the MTP drafter. spark2 runs tighter than spark1
>   (11 vs 18 GB) due to the embed sidecar — under heavy batch load watch `free -g`;
>   if available dips to single digits, drop spark2 to seqs 8 or util 0.78.
> - **No client config change**; `dgxlib/models.yaml` unchanged.
> **Revert spark2 to MTP-2 (the latency config):** `ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`.
>
> The A/B that drove this:
>
> **▶ PREV (2026-06-23, later): spark1 → THROUGHPUT (plain, NO spec decode, seqs 16); spark2 was LATENCY (MTP-2, seqs 4). The A/B.**
> spark1's `vllm-chat` was swapped
> MTP-2/seqs-4 → **plain (no `--speculative-config`), `--max-num-seqs 16`** via
> `spin-up-vllm-qwen3-next-80b.sh`. Same model
> (`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`), same vLLM 0.22.0
> (`vllm/vllm-openai:v0.22.0-aarch64`), 256K, util 0.80, fp8 KV, hermes tools.
> spark2 was untouched (MTP-2, seqs 4) as the control.
> - **Why:** for concurrent load the decode-tok/s levers invert. (1) MTP is a
>   low-batch trick — per step it processes `batch×(1+N)` token-positions, ~free
>   when bandwidth-bound at low batch but real wasted compute (rejected drafts)
>   once a large batch turns the box compute-bound. (2) `seqs 4` was an admission
>   cap, not a memory limit (KV pool 1.67M tok → **6.38× @ 256K**), and spark2 was
>   observed queuing 4-deep while spark1 sat idle. Plain + seqs 16 maximizes
>   **aggregate** tok/s at the cost of **single-stream** latency.
> - **READ THIS BEFORE JUDGING THE DASHBOARD:** with MTP off, spark1 at idle /
>   low concurrency reads ~30 tok/s vs spark2's ~56 — it looks *worse*. The
>   throughput win only appears under real concurrent load (`spark-tps.sh`).
> - **Config audit caught a footgun:** the plain script hardcoded
>   `IMAGE=vllm/vllm-openai:latest`, and spark1's cached `:latest` is stale at
>   **0.21.0** — below the documented Qwen3-Next 0.22.0 floor (#40880 silent
>   degenerate output). First run came up on 0.21.0; **fixed the script to default
>   `IMAGE` to the pinned `v0.22.0-aarch64`** (now overridable, matches the MTP
>   sibling) and re-ran. Live engine log confirms **v0.22.0**, `speculative_config=None`.
> - **Verified:** seqs 16, no spec decode, KV 6.38× @ 256K, coherent
>   (*"The ocean stretches endlessly under the moonlit sky, its waves whispering
>   secrets to the shore."*), host 18 GB available (dropping the MTP drafter freed
>   memory vs the MTP build's 12 GB).
> - **Residual A/B confound (decode-irrelevant):** the plain build uses vLLM's
>   default chunked prefill; spark2 pins `--max-num-batched-tokens 40960`. Affects
>   prefill/TTFT only, not the decode tok/s being compared.
> - **No client config change** — served model id unchanged; `dgxlib/models.yaml`
>   unchanged (spec-decode / seqs / version don't change request behavior).
> **Revert spark1 to MTP-2 (mirror spark2):** `ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`.
>
> ---
>
> **▶ PREV (2026-06-23, earlier): spark1 swapped to spark2's MTP config — both boxes were MTP-2.**
> spark1's `vllm-chat` was swapped from the plain build to the **MTP-2
> speculative-decode** build via `spin-up-vllm-qwen3-next-80b-mtp.sh`, so spark1
> now mirrors spark2's chat slot exactly: same model
> (`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`), **vLLM 0.22.0**
> (`vllm/vllm-openai:v0.22.0-aarch64`, was `:latest`), `--speculative-config
> '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`, **256K**
> (`--max-model-len 262144`), **chunked prefill `--max-num-batched-tokens
> 40960`**, **seqs 4** (was 8), util 0.80, hermes tools, fp8 KV. The two boxes'
> chat slots are now identical; the only remaining difference is spark2 also runs
> the `vllm-embed` sidecar (port 8000) and spark1 does not.
> - **No download** — the FP8 weights and the v0.22.0 image were already cached on
>   spark1. Stop-old → start-new → load → warmup completed in ~10 min.
> - **MTP confirmed live**: engine log shows `Loading drafter model...` and
>   `SpecDecoding metrics: Avg Draft acceptance rate ...`. Coherence check PASS (no
>   #36872 gibberish): *"The ocean stretches endlessly beneath the horizon, its
>   waves whispering secrets of the deep."* spark1 has no embed sidecar so host
>   headroom is larger than spark2's — 0.80 is comfortable.
> - **No client config change** — served model id unchanged, so MemPalace /
>   llm_wiki / CampaignGenerator / opencode keep working; `dgxlib/models.yaml`
>   unchanged (MTP/util/ctx/seqs don't change request behavior).
> **Revert spark1 to plain (non-MTP):** `ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b.sh'`.
>
> ---
>
> **▶ PREV (2026-06-20, later): both boxes re-tuned — util 0.88→0.80; both→256K; spark2→MTP-2 (chunked prefill 40k).**
> Same model on both (`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`, single-box TP=1,
> fp8 KV, hermes tools). Changes this session:
> - **`--gpu-memory-utilization` 0.88 → 0.80 on BOTH.** Unified memory: the
>   reservation steals host RAM. 0.88 left ~15 GB host and **wedged spark2**
>   (sshd couldn't fork; full reboot). 0.80 ≈ 26 GB host headroom. And it costs
>   nothing usable — decode is bandwidth-bound, so KV above real concurrency is
>   unreadable anyway. **Full reasoning: `gpu-reservation-and-kv-tradeoffs.md`**;
>   rule: memory `feedback_gpu_util_080_default`. Script defaults flipped to 0.80.
> - **spark1: 128K → 256K** (`--max-model-len 262144`, native max). Measured KV
>   pool 1,618,316 tokens → 6.17× concurrency at 256K. Plain build, chunked
>   prefill on. Smoke PASS.
> - **spark2: now runs MTP-2 speculative decode** (`--speculative-config
>   '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`) on **vLLM 0.22.0**
>   (`vllm/vllm-openai:v0.22.0-aarch64`), **256K** (`--max-model-len 262144`),
>   **chunked prefill ON, `--max-num-batched-tokens 40960`**, seqs 4. The FP8
>   checkpoint ships the MTP head (verified). Measured: **85.2% draft acceptance
>   + ~56 tok/s** effective on a realistic code/prose prompt (~2.70
>   tok/backbone-step); coherent (no #36872 gibberish). KV pool 1,010,368 tokens
>   → 3.85× at 256K. Via `spin-up-vllm-qwen3-next-80b-mtp.sh`.
>   - **256K needs chunked prefill on spark2.** With `--no-enable-chunked-prefill`
>     vLLM ties the warmup batch to `max_model_len`, so a 256K (262K-token) warmup
>     starved the host and **wedged the box twice** (reboots). Chunked prefill
>     bounds the warmup to a 40k chunk → host stayed healthy (31 GB free through
>     load). **MTP + chunked prefill IS supported on vLLM 0.22.0** (the recipe
>     disables chunked prefill, but that's not a hard requirement here) — verified
>     accepted at engine init, only a benign "min_p/logit_bias won't work with
>     spec decode" warning. The earlier `NO_CHUNKED_PREFILL=1` 64K fallback is no
>     longer needed but remains a script knob.
> - **Non-MTP baseline A/B still pending** (would need one more spark2 swap to
>   state the clean MTP speedup; step rate implies ~1.8–2×).
> - Prefix caching left OFF / not pursued (`enable_prefix_caching=False`).
> - Client model id is **unchanged** (`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`), so
>   no client config change needed; `dgxlib/models.yaml` unchanged (MTP/util/ctx
>   don't change request behavior).
> **Revert spark2 to plain (non-MTP):** `ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b.sh'`.
> **Revert util to 0.88:** `GPU_UTIL=0.88 bash ~/spin-up-vllm-qwen3-next-80b.sh` (don't — it wedges spark2).
>
> ---
>
> **▶ PREV (2026-06-20, earlier): single-box Qwen3-Next-80B-A3B-Instruct-FP8 on BOTH boxes.**
> The cross-box `vllm-2box` Qwen3.5-122B TP=2 cluster was torn down on both
> boxes and replaced with an **independent single-box `vllm-chat`** on each:
> **`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`**, 128K (`--max-model-len 131072`),
> TP=1, `--gpu-memory-utilization 0.88`, hermes tools, no reasoning parser,
> image `vllm/vllm-openai:latest`, via `spin-up-vllm-qwen3-next-80b.sh` on each
> box. **Both endpoints now serve independently** — spark1 `192.168.1.147:8001`
> AND spark2 `192.168.1.121:8001`. spark2 `vllm-embed` (port 8000,
> `Qwen/Qwen3-Embedding-0.6B`) kept running throughout. Smoke + generation PASS
> on both.
> **Why:** the 122B (10B active) decodes long structured-output render jobs
> (pdf-translators 5etools-JSON conversion) too slowly — big chapters exceeded
> the dgxlib `read_timeout` and never completed (retry-from-scratch loop). The
> 80B-A3B (3B active) decodes ~3x faster and completes the same jobs. A
> throughput/decode-rate win for long-output render, not a capability change
> (a 14B rendered the same chunk with 0 validation errors). Side finding:
> Ollama cannot co-host a model on spark1 while a vLLM chat slot is resident —
> no GPU memory, it falls back to CPU at ~3 tok/s.
> **Clients flipped to the new id:** MemPalace `llm_model`, opencode `dgx`
> default + build agent, llm_wiki custom provider, CampaignGenerator
> `DGX_DEFAULT_MODEL`.
> **Revert to cross-box 122B:** `PROFILE=qwen35 ./spin-up-vllm-2box-rdma.sh`
> from the workstation (tear down both single-box `vllm-chat` first).
>
> ---
>
> **Note (2026-06-11):** spark2 was briefly swapped to **SGLang** as a
> calibration A/B, then **reverted to vLLM the same day** — SGLang was no
> serving improvement on GB10 (decode is bandwidth-bound; the tuned
> kernels don't run on this GPU). The experiment record (two GB10 gates,
> image findings) is in `sglang-qwen3-next-spark2-observations.md` and the
> reproducible spin-up is `spin-up-sglang-qwen3-next-80b.sh`. **Live state
> below is vLLM.**

> **⚠️ PREV (2026-06-10, superseded): spark1 ran the CODER variant —
> `Qwen/Qwen3-Coder-Next-FP8` with `--tool-call-parser qwen3_coder` and
> **no reasoning parser**. spark2 stays on the Instruct variant.**
> spark1's `vllm-chat` was swapped Thinking-FP8 → Qwen3-Coder-Next-FP8 on
> 2026-06-10 via `spin-up-vllm-qwen3-coder-next.sh` (fresh weight pull ~80 GB).
> Current live state:
>
> | box | role | container | image | notes |
> |---|---|---|---|---|
> | **spark1** (192.168.1.147) | Ray HEAD, serves `spark1:8001` | `vllm-2box` | `local/vllm-ray:26.05` | Qwen3.5-122B-A10B-FP8, **256K** (`--max-model-len 262144`), TP=2, RoCE/IB (`rocep1s0f0:1`), qwen3_coder tools, qwen3 reasoning parser, gpu-util 0.85. |
> | **spark2** (192.168.1.121) | Ray WORKER (no independent endpoint) | `vllm-2box` | `local/vllm-ray:26.05` | Same model, tensor-parallel rank 1; all client traffic to spark1:8001. |
>
> **The cable is LIVE** — TP=2 NCCL all-reduce over RoCE/IB. Both
> single-box `vllm-chat` containers are stopped (torn down 2026-06-15 to
> free GPU memory).
>
> **SGLang A/B (2026-06-11, reverted):** spark2 briefly ran this model on
> SGLang (`sglang-chat`, `lmsysorg/sglang:v0.5.10.post1-cu130`) — two
> GB10-specific gates cleared (FlashInfer→Triton; DeepGEMM FP8-MoE needs
> 0.5.10 not 0.5.9), verified working, then reverted to vLLM same day (no
> serving win on GB10). Record in `sglang-qwen3-next-spark2-observations.md`;
> re-run with `bash ~/spin-up-sglang-qwen3-next-80b.sh` (stops vllm-chat).
>
> **Why no reasoning parser (verified 2026-06-10):** Qwen3-Coder-Next
> wraps its *entire* answer in `<think>…</think>` with nothing after the
> close tag. With `--reasoning-parser qwen3` active the parser pulls that
> whole block out into `reasoning` and leaves `content` **null** — clients
> reading `content` get an empty response. Deployed WITHOUT the parser, the
> raw `<think>` tags stay in `content` so clients at least see the output.
> The slot therefore runs `--tool-call-parser qwen3_coder` and **no
> `--reasoning-parser`** — which the spin-up script now does by default, so a
> plain run reproduces this state (set `REASONING_PARSER=qwen3` to opt into
> thinking mode). qwen3_coder tool calling PASSes.
>
> **Nemotron-3-Super verdict (2026-06-06):** the single-box NVFP4 hybrid
> (12B active) did NOT clear the Qwen3.5-122B coding bar. Reasoning was
> genuinely good and it correctly *saw the scope* of problems, but it got
> lost *executing* long-horizon changes — concretely, it bogged down
> partway through a Python-parser rewrite it had correctly sized up. A
> capability gap, not a latency one (MTP wouldn't fix it). Full writeup:
> `nemotron3-super-120b-observations.md` + memory
> `project_nemotron3_super_nvfp4`. The infra it proved out still stands:
> NVFP4 runs on real CUTLASS FP4 kernels on GB10/sm_121, Nemotron-H loads
> at 120B — the `spin-up-vllm-nemotron3-super-120b.sh` script is kept for
> a future re-test.
>
> **Embeddings:** still on **Ollama `nomic-embed-text` (port 11434)** —
> the `vllm-embed` container was not brought back (could fit alongside
> Qwen3-Next's 0.88 util by dropping to ~0.85, but left on Ollama for
> continuity; see §7).
> **Revert spark1 to the Nemotron-Super experiment:**
> `ssh spark 'bash ~/spin-up-vllm-nemotron3-super-120b.sh'`.
> **Restore the cross-box Qwen3.5-122B coder** (from the WORKSTATION):
> `PROFILE=qwen35 ./spin-up-vllm-2box-rdma.sh` (recreates `vllm-2box` on
> both boxes; tear down the two single-box `vllm-chat` containers first:
> `ssh spark 'docker rm -f vllm-chat'; ssh spark2 'docker rm -f vllm-chat'`).
>
> ---
>
> **▶ PREV (2026-06-17, superseded by the 2026-06-20 banner above): cross-box Qwen3.5-122B-FP8 TP=2.**
> `vllm-2box` (image `local/vllm-ray:26.05`) running on both boxes
> via `PROFILE=qwen35 ./spin-up-vllm-2box-rdma.sh`. Serving
> **`Qwen/Qwen3.5-122B-A10B-FP8`** at **256K** context
> (`--max-model-len 262144`), TP=2, RoCE/IB (`rocep1s0f0:1`, GID 3),
> `--gpu-memory-utilization 0.80`, `--max-num-seqs 20`,
> `--shm-size 2g`, `--tool-call-parser qwen3_coder`,
> `--reasoning-parser qwen3`. Endpoint: spark1:8001 only.
> Smoke PASS; **~20.1 tok/s** unloaded, ~130 tok/s at 20 concurrent.
> **Cable is LIVE** — TP=2 NCCL all-reduce over RoCE/IB.
> spark2 `vllm-embed` (port 8000, `Qwen/Qwen3-Embedding-0.6B`) running.
> **Stability note (2026-06-17):** at 0.85 util + no max-num-seqs,
> spark2 OOMed repeatedly under high concurrency (unified memory —
> Ray worker CPU heap competes with GPU reservation). Fixed by:
> 0.85→0.80 util, 10g→2g shm, max-num-seqs 20.
> **Revert to single-box Qwen3-Next-80B on each box:**
> `ssh spark 'docker rm -f vllm-2box && bash ~/spin-up-vllm-qwen3-next-80b.sh'`
> `ssh spark2 'docker rm -f vllm-2box && bash ~/spin-up-vllm-qwen3-next-80b.sh'`
> Full cross-box perf record: `qwen35-122b-2box-observations.md`.
>
> ---
>
> **▶ PREV (2026-06-15, evening): single-box Qwen3-Next-80B on
> both sparks.** Cross-box `vllm-2box` (Qwen3.5-122B-FP8 TP=2) torn
> down on both boxes. Each spark ran an independent single-box
> `vllm-chat` container (image `vllm/vllm-openai:latest`, TP=1) serving
> **`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`** — 128K context
> (`--max-model-len 131072`), `--max-num-seqs 8`, `--gpu-memory-utilization
> 0.88`, fp8 KV, `hermes` tool parser, no reasoning parser.
> spark2 `vllm-embed` (port 8000, `Qwen/Qwen3-Embedding-0.6B`) kept
> running throughout — unaffected.

Snapshot of what's actually running on **both** DGX Sparks as of
2026-06-23 (single-box Qwen3-Next-80B-A3B-Instruct-FP8 on each at
util 0.80, both @ 256K on vLLM 0.22.0, **both now on the THROUGHPUT config**
(plain, no spec decode, seqs 16) after the A/B concluded — see top LIVE
banner. spark2 also runs the `vllm-embed` sidecar (port 8000); spark1 does not).
Use this as a "rebuild from scratch" reference if either box wipes, or as
inventory when debugging.

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
| LAN IP | `192.168.1.147` | `192.168.1.121` (10 GbE wired; was `192.168.1.69` WiFi 2026-06-13) |
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
| current use | **LIVE (2026-06-17)** — cross-box TP=2 (Qwen3.5-122B-FP8, 256K), RoCE/IB verbs (`NCCL_IB_HCA=rocep1s0f0:1`, GID 3). Measured ~20.2 tok/s (128 tokens incl. prefill). PP=2 still the next decode experiment. |

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
| 11434 | Ollama (systemd) | LLM serving + **currently the live embeddings path** (`nomic-embed-text`) while vllm-embed is down |
| 8000 | vllm-embed (docker) | Embeddings — `nomic-embed-text-v1.5` — **DOWN** (stopped back during the cross-box experiment, still not restored; embeddings on Ollama 11434. Could now be restored — box is single-box again — but left on Ollama for continuity) |
| 8001 | vllm-chat (docker) | Chat completions — **`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`** single-box TP=1, **256K** (`--max-model-len 262144`), **NO spec decode (plain)**, default chunked prefill, hermes tools, no reasoning parser, **gpu-util 0.80**, **seqs 16**, image **`vllm/vllm-openai:v0.22.0-aarch64`**. Via `spin-up-vllm-qwen3-next-80b.sh`. (2026-06-23 later: swapped MTP→plain + seqs 4→16 for the THROUGHPUT side of the A/B; spin-up script now pins `IMAGE` to v0.22.0 by default — was hardcoded `:latest` which resolved to 0.21.0, below the Qwen3-Next floor.) Independent endpoint. |

### spark2 (192.168.1.121)

| port | service | purpose |
|---:|---|---|
| 8000 | vllm-embed (docker) | Embeddings — `Qwen/Qwen3-Embedding-0.6B` — 1024-dim, instruction-aware, `--runner pooling --enforce-eager`, gpu-util 0.05, image `local/vllm-ray:26.05` (entrypoint overridden: `--entrypoint ""`), **`--restart unless-stopped`** (always-on). Smoke-tested 2026-06-15: `dim=1024`. |
| 8001 | vllm-chat (docker) | Chat completions — **`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`** single-box TP=1, **256K** (`--max-model-len 262144`), **MTP-2 spec decode** (`qwen3_next_mtp`), **chunked prefill `--max-num-batched-tokens 40960`**, **gpu-util 0.80**, seqs 4, hermes tools, image **`vllm/vllm-openai:v0.22.0-aarch64`**. KV pool 1.01M tok (3.85× @ 256K); 85.2% draft acceptance, ~56 tok/s. Via `spin-up-vllm-qwen3-next-80b-mtp.sh`. (2026-06-20: was plain 128K @ 0.88.) Independent endpoint. |

(No Ollama on spark2. Briefly ran SGLang `sglang-chat` on 2026-06-11; reverted to vLLM same day — see the
note under the LIVE banner.)

## VRAM budget (steady state)

### spark1

| service | reserved cap | actual model size | notes |
|---|---:|---:|---|
| vllm-embed | — | — | **DOWN** — embeddings on spark2:8000 (`Qwen/Qwen3-Embedding-0.6B`); spark1 left without a local embedder for continuity |
| vllm-chat (single-box) | ~102 GB (**0.80** × ~128 GB) | ~76 GiB FP8 weights + fp8 KV + activations @ 256K (no MTP draft) | Qwen3-Next-80B-A3B, **plain (no spec decode), seqs 16** (THROUGHPUT side of the A/B, 2026-06-23 later), 256K, default chunked prefill (hybrid: only the periodic full-attn layers carry KV). KV pool **1.67M tok → 6.38× @ 256K**. **No embed sidecar on spark1**; dropping the MTP drafter freed memory — **18 GB host available** (vs 12 GB under the MTP build). |
| Ollama (idle) | ~0 | unloads after `OLLAMA_KEEP_ALIVE` | 5 min default |
| Ollama (loaded) | varies | qwen2.5:14b ≈ 14.5 GB, nomic ≈ 600 MB | only when actively serving |

Both vLLM containers stay resident; Ollama unloads on idle (different
design — see `spark-llm-serving-learnings.md`). The 80 GB FP8 weights
+ KV cache + activations leave very little headroom on the 128 GB
unified-memory device. Note the util is now **0.80, not 0.88** — on
unified memory the GPU reservation steals host RAM, and 0.88 starved the
host (sshd couldn't fork) and wedged spark2; see
`gpu-reservation-and-kv-tradeoffs.md`. The KV given back is unusable
anyway (bandwidth-bound). **Any third vLLM sidecar will not fit alongside
the 80B; spec the smaller container first if you ever co-host.**

### spark2

| service | reserved cap | actual model size | notes |
|---|---:|---:|---|
| vllm-chat (single-box) | ~102 GB (**0.80** × ~128 GB) | ~76 GiB FP8 weights + fp8 KV + activations @ 256K + MTP draft | Qwen3-Next-80B-A3B + **MTP-2 spec decode**, 256K via **chunked prefill 40k**. KV pool 1.01M tok → 3.85× @ 256K. With the embed sidecar co-resident, host runs ~31 GB free during load / ~12 GB steady at 0.80 — chunked-prefill's 40k warmup batch (not a 262K one) is what keeps it from wedging. |
| vllm-embed | ~6 GB (0.05 × ~128 GB) | `Qwen/Qwen3-Embedding-0.6B` weights | Always-on; coexists with vllm-chat. This sidecar is why spark2 has less host headroom than spark1 (and why 0.88 wedged spark2 but not spark1). |

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
# nomic-embed-text:latest    274 MB             (768-dim — current live embedding path)
# qwen3-embedding:0.6b       639 MB             (1024-dim — UPGRADE CANDIDATE, pulled 2026-06-11)
# qwen2.5:14b                8.99 GB  (Q4_K_M GGUF)
# qwen2.5:32b                18.5 GB  (Q4_K_M GGUF)
# llama3.3:70b               40 GB    (Q4_K_M GGUF)
```

The 32B and 70B were pulled for ad-hoc comparison against the vLLM
slot; not used by any production client. Safe to `ollama rm` if disk
gets tight.

**Embedding upgrade in progress (2026-06-11):** `qwen3-embedding:0.6b`
(current SOTA self-hosted family, 1024-dim, instruction-aware) pulled as a
candidate to replace `nomic-embed-text` (768-dim, early-2024). Serving via
**Ollama, not the vllm-embed slot** — restoring vllm-embed on spark1 is
**blocked**: vllm-chat at `gpu-util 0.88` leaves only **~2.4 GB free** of
the 128 GB *unified* (CPU+GPU) memory (measured 2026-06-11 via
`torch.cuda.mem_get_info`), so a second vLLM container OOMs. Co-hosting a
vLLM embedder would require dropping vllm-chat's util (a chat bounce); the
KV-pool headroom allows it but it wasn't worth disrupting chat for the
A/B. Ollama shares memory dynamically, so it co-exists with vLLM-chat
fine. **A/B DONE (2026-06-11): qwen3-embedding wins decisively.** Held-out-span
retrieval, n=400 on `corpus.txt`, each model with native prefixes
(`embed-ab.py`): qwen3 vs nomic = hit@1 35.0% vs 27.3%, hit@5 57.5% vs
44.5%, hit@10 68.2% vs 53.0%, MRR 0.459 vs 0.363 (~+27% relative, ~5× the
sampling noise). Cost: ~2.2× slower to embed (0.6B decoder + 1024 dims) —
matters at re-index, negligible per-query. **Recommendation: cut over.**
Remaining work is in the CLIENT repos, not here: re-embed every corpus
(768→1024) in mempalace/turbovecdb + llm_wiki, point them at
`qwen3-embedding:0.6b`, apply the query instruction prefix
(see `embed-ab.py` for the exact strings). Spin-up script for the faster
vLLM path (if chat util is freed later): `spin-up-vllm-qwen3-embed-0.6b.sh`.

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

> **PREV (2026-06-17, superseded 2026-06-20 → single-box 80B): this slot was `vllm-2box`** (cross-box TP=2),
> brought up by `PROFILE=qwen35 ./spin-up-vllm-2box-rdma.sh` on
> `local/vllm-ray:26.05`. Serving **`Qwen/Qwen3.5-122B-A10B-FP8`**,
> **256K** context (`--max-model-len 262144`), TP=2, RoCE/IB
> (`rocep1s0f0:1`, GID 3), `--gpu-memory-utilization 0.85`,
> `--tool-call-parser qwen3_coder`, `--reasoning-parser qwen3`.
> Smoke + NCCL verified PASS. Measured **~20.2 tok/s** (128 tokens, incl.
> prefill). Cable is LIVE.
>
> **Prior occupant (2026-06-15 evening):** single-box
> `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` on `vllm/vllm-openai:latest`.
>
> **Slot history:** Qwen3-Next-80B Instruct → TurboQuant KV → Thinking
> (`--reasoning-parser qwen3`, 2026-06-08 → 2026-06-10) → Qwen3-Coder-Next
> (2026-06-10 → 2026-06-15 cross-box) → cross-box Qwen3.5-122B-FP8
> (2026-06-15 morning) → **Qwen3-Next-80B Instruct (2026-06-15 evening,
> current)**. The Nemotron-3-Super NVFP4 experiment that held this slot
> on 2026-06-06 concluded — it missed the Qwen3.5-122B coding bar. Full
> writeup: `nemotron3-super-120b-observations.md`. Re-run with
> `ssh spark 'bash ~/spin-up-vllm-nemotron3-super-120b.sh'`.
>
> The TurboQuant prose below describes the **prior occupant** of this
> slot; kept as the runbook for the TurboQuant KV variant
> (`spin-up-vllm-qwen3-next-80b-turboquant.sh`), which is NOT currently
> live.

Previously serving **Qwen3-Next 80B A3B Instruct FP8** (hybrid attention,
~3B active per token, ~80B total) with **TurboQuant KV cache
(`turboquant_k8v4`) on vLLM 0.22.0** (image
`vllm/vllm-openai:v0.22.0-aarch64`). Slot history: Qwen 2.5 14B AWQ →
Llama 3.3 70B AWQ + spec-decode → Gemma 4 26B MoE → Gemma 4 26B MoE
longctx → **Nemotron 3 Nano 30B A3B (2026-05-18 to 2026-05-19, rejected
after Phase B — see `nemotron3-nano-30b-observations.md`)** → Gemma 4
26B MoE longctx → **Qwen3-Next 80B A3B Instruct FP8, plain fp8 KV
(2026-05-21 → 2026-05-30)** → **Qwen3-Next 80B A3B Instruct FP8,
TurboQuant `turboquant_k8v4` KV + vLLM 0.22.0 (2026-05-30 → 2026-06-08)**
→ **Qwen3-Next 80B A3B Thinking FP8, plain fp8 KV (2026-06-08 → 2026-06-10)**
→ **Qwen3-Coder-Next FP8, plain fp8 KV, `qwen3_coder` tools (2026-06-10 → 2026-06-15)**
→ **Qwen3-Next 80B Instruct FP8, fp8 KV, single-box (2026-06-15 evening)**
→ **Qwen3.5-122B-A10B-FP8, TP=2 cross-box (2026-06-17 → current)**.
vllm-chat swap-in scripts are
`spin-up-vllm-qwen3-coder-next.sh`,
`spin-up-vllm-qwen3-next-80b.sh` (Instruct/Thinking variant — revert),
`spin-up-vllm-qwen3-next-80b-turboquant.sh` (TurboQuant KV, v0.22.0),
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

Currently launched via `PROFILE=qwen35 ./spin-up-vllm-2box-rdma.sh` (run from the workstation). See that script for the full Ray HEAD + WORKER docker commands. Key effective vllm serve flags:

```bash
vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-model-len 262144 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 20 \
  --host 0.0.0.0 --port 8001
# NCCL transport: NCCL_IB_HCA=rocep1s0f0:1, NCCL_IB_GID_INDEX=3 (RoCE v2)
# Image: local/vllm-ray:26.05 (both boxes)
# --shm-size 2g (Ray plasma store — TP all-reduce goes over NCCL/RDMA, plasma unused)
# Why 0.80 not 0.85: GB10 unified memory — GPU KV reservation competes with CPU RAM.
# At 0.85 + vllm-embed 0.05 + OS/Ray ~10 GB, spark2 had only ~30 MB free and OOMed
# under high concurrency (Ray worker heap grows with in-flight sequences).
# Why max-num-seqs 20: caps CPU-side scheduler state regardless of client concurrency.
# Requests beyond 20 queue at the HTTP layer (near-zero memory cost) until a slot frees.
```

The **TurboQuant variant** (`spin-up-vllm-qwen3-next-80b-turboquant.sh`)
is identical except it pins `vllm/vllm-openai:v0.22.0-aarch64`, uses
`--kv-cache-dtype turboquant_k8v4`, and adds `--max-num-batched-tokens
4096` (a guard for bug #41726). It is NOT currently live — the plain
fp8 build above is, matching spark2. The "Why these flags" notes below
cover both; the TurboQuant-specific flags apply only to that variant.

### Why these flags

- **`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`**: ~80B total, ~3B active
  per token. Hybrid architecture: Gated DeltaNet (linear attention,
  O(n), no KV cache) on most layers + periodic full-attention layers
  + MoE FFN. Native context 256K (extendable to 1M with YaRN); we run
  the full **256K** native max.
- **`--max-model-len 262144`** (256K): the model's native max. Measured
  2026-06-11 the KV pool holds ~2.53M tokens, so a full 256K request
  still leaves ~9.6x concurrency headroom even with `--max-num-seqs 4` —
  KV is *not* the binding constraint on this hybrid (only the periodic
  full-attention layers carry KV). Going past 256K toward the 1M YaRN
  ceiling would need rope-scaling args this script doesn't set. Fallback
  ladder if OOM: MAX_LEN=131072, then GPU_UTIL=0.85.
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

## 4. spark2 vllm-2box (Docker container, port 8001 — Ray WORKER only)

Cross-box Ray WORKER slot on the second box. No independent endpoint — all client traffic goes to spark1:8001.

> **PREV (2026-06-17, superseded 2026-06-20 → single-box 80B): spark2 was a Ray WORKER** for the cross-box
> `Qwen/Qwen3.5-122B-A10B-FP8` TP=2 cluster. Container `vllm-2box`,
> image `local/vllm-ray:26.05`, gpu-util 0.85, RoCE/IB (`rocep1s0f0:1`,
> GID 3). spark2 `vllm-embed` (port 8000, `Qwen/Qwen3-Embedding-0.6B`)
> kept running throughout — unaffected.
> **Prior occupant (2026-06-15 evening):** single-box `vllm-chat`
> serving `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` at 128K. Before that:
> cross-box Ray WORKER for Qwen3.5-122B-FP8 (2026-06-15 morning) →
> Qwen3-Next-80B Instruct at `--max-num-seqs 8` (2026-06-11) →
> SGLang A/B (reverted same day, 2026-06-11) → Nemotron-3-Nano 30B
> BF16 (2026-05-26 → 2026-06-06).
> SGLang record: `sglang-qwen3-next-spark2-observations.md`;
> script: `spin-up-sglang-qwen3-next-80b.sh`.

Previously serving **Nemotron 3 Nano 30B A3B BF16** — an NVIDIA hybrid
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
curl -sS http://192.168.1.121:8001/v1/chat/completions \
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
| laptop, desktop, etc. | spark1 Ollama OpenAI-compat | `http://192.168.1.147:11434/v1/...` — **currently also the live embeddings path** (`/v1/embeddings`, model `nomic-embed-text`) while vllm-embed is down for the cross-box experiment |
| laptop, desktop, etc. | spark1 vllm-embed | `http://192.168.1.147:8000/v1/embeddings` — **DOWN** (stopped during the cross-box experiment, not yet restored; embeddings served by Ollama 11434) |
| laptop, desktop, etc. | spark1 vllm-chat | `http://192.168.1.147:8001/v1/chat/completions` |
| laptop, desktop, etc. | spark2 vllm-embed | `http://192.168.1.121:8000/v1/embeddings` — `Qwen/Qwen3-Embedding-0.6B`, always-on |
| laptop, desktop, etc. | spark2 vllm-chat | `http://192.168.1.121:8001/v1/chat/completions` |

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

> **⚠️ LIVE (2026-06-20): spark1:8001 AND spark2:8001 serve `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`** (single-box on each box).
> Any client that sends an explicit model id — MemPalace `llm_model`,
> llm_wiki's custom-provider Model, CampaignGenerator's `DGX_MODEL` /
> `DGX_DEFAULT_MODEL`, the opencode `dgx` provider — must send
> **`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`** or the call 400s. All four were
> flipped to the 80B id on 2026-06-20 (see the top LIVE banner).

### MemPalace (`~/.mempalace/config.json` on laptop)

> **⚠️ LIVE (2026-06-20): single-box 80B is up.** `llm_model` was set to
> `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` (done 2026-06-20). Embeddings are on
> spark2:8000 (`Qwen/Qwen3-Embedding-0.6B`). The live file should be:
>
> ```json
> {
>   "embedding_provider": "openai-compat",
>   "embedding_model": "nomic-embed-text",
>   "embedding_endpoint": "http://192.168.1.147:11434",
>   "llm_endpoint": "http://192.168.1.147:8001",
>   "llm_model": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
> }
> ```

The single-box steady-state values (rebuild target) are:

```json
{
  "embedding_provider": "openai-compat",
  "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
  "embedding_endpoint": "http://192.168.1.147:8000",
  "llm_endpoint": "http://192.168.1.147:8001",
  "llm_model": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
}
```

mempalace mining embeds via the embedding endpoint and calls the chat
LLM during the "convos" extraction phase. The `llm_model` field **must
match the model id actually served at port 8001** — every swap of
vllm-chat requires updating this field too, or LLM calls return 400.
The chat palace at `~/.mempalace/palaces/chat/` is currently in a
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
`~/.config/opencode/opencode.json`.

> **LIVE (2026-06-20):** spark1:8001 (and spark2:8001) serve
> `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`. The top-level `"model"` and the
> `build` agent model are set to `"dgx/qwen3-next-80b"` (already in the
> `models` block). The `qwen35-122b` entry below is kept as a dead/history
> entry — selecting it 400s until a 122B is re-served.

**Current live entry (already registered):**
```json
"qwen3-next-80b": {
  "id": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
  "name": "Spark1 (.147) Qwen3-Next 80B A3B Instruct FP8 @ 128K (hybrid attention, tools)",
  "limit": { "context": 131072, "output": 48192 },
  "tool_call": true,
  "temperature": true
}
```
Top-level `"model"`: `"dgx/qwen3-next-80b"`

The DGX provider's full historical entry set (active default at the time was
`dgx/qwen3-next-80b`):

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
        "baseURL": "http://192.168.1.121:8001/v1",
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
(`http://192.168.1.121:8001/v1`). Switch to a spark2 model by setting
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
until curl -sS --max-time 2 http://192.168.1.121:8001/v1/models 2>/dev/null | grep -q '"id"'; do sleep 5; done
```

Warm-restart cost: vllm-embed ~30s, spark1 vllm-chat ~5–10 min
(80 GB weights), spark2 vllm-chat ~3–5 min (80 GB FP8 weights).

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

4. **Start vllm-chat SECOND** via the spin-up script (Qwen3-Coder-Next
   FP8 on vLLM 0.22.0 is the current default — 256K context, hybrid
   attention, `qwen3_coder` tools, no reasoning parser):
   ```bash
   scp spin-up-vllm-qwen3-coder-next.sh spin-up-vllm-qwen3-next-80b.sh \
       lib-vllm-spinup.sh test-toolcall.sh spark:~/
   ssh spark 'docker pull vllm/vllm-openai:v0.22.0-aarch64'
   ssh spark 'bash ~/spin-up-vllm-qwen3-coder-next.sh'
   ```
   Expect ~40 min on first run (HF pulls ~80 GB of FP8 weights on a
   fresh box; ~13 min observed warm-cache on 2026-05-30 — shard load +
   torch.compile dominate). Script waits for `Application startup
   complete` and smoke-tests on its own. Then verify tool calling
   (`HOST`/`PORT`, not `DGX_*`):
   `ssh spark 'MODEL=Qwen/Qwen3-Coder-Next-FP8 ~/test-toolcall.sh'`.
   If the smoke output is repeated/garbled, that's bug #40880 — re-run
   with `ENFORCE_EAGER=1`. To revert or swap models, see §9.

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

# CROSS-BOX slot (NOT live as of 2026-06-15 evening — single-box on each box).
# Run from the WORKSTATION. PROFILE picks the model + parsers; RDMA=1 = RoCE/IB (default).
# To bring up: tear down single-box containers first.
PROFILE=qwen35  ./spin-up-vllm-2box-rdma.sh   # Qwen3.5-122B-A10B-FP8 @ 256K (last cross-box config)
PROFILE=minimax ./spin-up-vllm-2box-rdma.sh   # nvidia/MiniMax-M2.7-NVFP4 @ 64K
RDMA=0 PROFILE=qwen35 ./spin-up-vllm-2box-rdma.sh  # same, revert transport to TCP sockets
# Tear the cross-box slot down and return to the single-box scripts below:
ssh spark 'docker rm -f vllm-2box'; ssh spark2 'docker rm -f vllm-2box'

# SINGLE-BOX vllm-chat swaps on port 8001 (one-liner each).
# CURRENT (2026-06-15 evening): BOTH sparks run Qwen3-Next-80B Instruct FP8,
# 128K, fp8 KV, hermes tools, max-num-seqs 8. Same model on each box.
ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b.sh'         # Qwen3-Next 80B Instruct FP8, plain fp8 KV @ 128K, hermes, seqs=8 (CURRENT — both sparks)
ssh spark 'bash ~/spin-up-vllm-qwen3-coder-next.sh'  # Qwen3-Coder-Next FP8, fp8 KV @ 256K (native max), qwen3_coder tools, NO reasoning parser
ssh spark 'bash ~/spin-up-vllm-nemotron3-super-120b.sh'      # Nemotron 3 Super 120B A12B NVFP4 @ 128K (concluded experiment; reasoning+tools)
ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b-turboquant.sh' # Qwen3-Next 80B FP8 + TurboQuant KV @ 128K, vLLM 0.22.0
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
curl -sS http://192.168.1.121:8001/v1/models

# Smoke-test (model id has to match — see top of doc)
curl -sS http://192.168.1.121:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Next-80B-A3B-Instruct-FP8","messages":[{"role":"user","content":"Say only OK"}],"max_tokens":20}'

# HF cache size (spark2-only — not shared with spark1)
ssh spark2 'du -sh ~/.cache/huggingface'

# Current spark2 bring-up (committed script; scp it + lib-vllm-spinup.sh first):
ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b.sh'  # Qwen3-Next 80B Instruct FP8, 128K, fp8 KV, hermes, max-num-seqs 8 (CURRENT spark2 — seqs=8 is now the default)
# SGLang A/B alternative (reverted 2026-06-11): bash ~/spin-up-sglang-qwen3-next-80b.sh  (stops vllm-chat; see §4)
# To swap the spark2 model: override QWEN_MODEL / MAX_LEN / MAX_SEQS / etc.,
# or adapt the docker run block from §4 with the new model id and re-run.
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
