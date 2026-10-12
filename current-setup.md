# Current DGX Spark Setup

**Current `vllm-chat` model ids** (copy-paste for client configs):

```
spark1 (192.168.1.147:8001):  qwen3.8-flash-next  (container `qwen38-flash`, SINGLE-BOX TP=1, Qwen3.8-Flash-Next NVFP4 experts + blockwise-fp8 side layers ("hybrid"), 262K ctx, util 0.80, seqs 8, MTP-2 spec decode, APC ON (block_size fix), deterministic QSA top-k, reduced draft vocab, bf16 KV, PLE n-gram table mmapped from NVMe, TOOL CALLING ON (qwen3_coder parser) + reasoning parser qwen3, vLLM 0.1.dev20073+g8e685d198)  ← LIVE 2026-09-10. NOTE the served id is LOWERCASE and unlike every previous id — every client must be repointed.
spark2 (192.168.1.121:8001):  — STOPPED 2026-10-10 for the NPC prototype (container `qwen38-flash` kept, `docker stop` only; restart: `ssh spark2 'docker stop decision2-nox-4b; docker start qwen38-flash'`). Was qwen3.8-flash-next, identical to spark1.
spark2 (192.168.1.121:8002):  — STOPPED again 2026-10-06 after the frozen-dataset rerun (container `clef` kept, `docker stop` only; restart: `ssh spark2 docker start clef`). Was clef, clef-flash (Jev/SystemOne API).
spark2 (192.168.1.121:8003):  — REMOVED again 2026-10-06 (Decision-2.0-Lux-9B, container `decision2-lux-9b`, ran ~1 h for the frozen-dataset rerun). Re-create: `MODEL=Lux-9B PORT=8003 SR_SRC=x ./spin-up-decision2.sh`
spark2 (192.168.1.121:8004):  — STOPPED 2026-10-08 to give spark2 back to qwen3.8-flash-next (container `decision2-kai-0-6b` kept; restart: `ssh spark2 docker start decision2-kai-0-6b`). Decision-2.0-Kai-0.6B, ~0.2 GB RSS.
spark2 (192.168.1.121:8005):  Decision-2.0-Nox-4B (container `decision2-nox-4b`, ~19 GB RSS) — the NPC prototype's stance backend  ← LIVE again 2026-10-10 (`docker start`; /health ready, ~118 GB host available)
spark1 (192.168.1.147:11434): qwen3-embedding:0.6b  (Ollama, lazy-load — the live MemPalace embedding path; VERIFIED UNAFFECTED by the 2026-09-10 swap)
spark2 (192.168.1.121:11434): qwen3-embedding:0.6b  (Ollama, lazy-load — unloads after 5 min idle; was vllm-embed on port 8000 until 2026-06-30)
```
> **▶ DONE (2026-10-10, 17:03–21:17 UTC): cascade-paper serving ablation on spark1 finished; `qwen38-flash` is back on the documented config.** Restored via `./spin-up-vllm-qwen38-flash-next.sh` (defaults) by `clef/eval/cascade_ablation.sh`. **Verified 21:20 UTC:** served id `qwen3.8-flash-next`, `max_model_len` 262144, `num_speculative_tokens: 2`, `enable_prefix_caching=True`, `--restart unless-stopped`, coherence PASS (Paris). **KV pool this boot: 586,565 tok** (2.24× @ 262K). The ablation ran `MTP=0` (APC on), then `MTP=0 PREFIX_CACHE=0`; results in `paper-cascade/` and `~/data/decision-eval/v1/cascade/`.
> - **⚠ Do NOT run this image with `MTP=0` and `PREFIX_CACHE=1`.** In that config an exact warm-cache repeat of 864 short requests was **38% slower** than the cold pass and **changed 2 answers**. With MTP=2 (deployed) the same test was 2% faster with identical answers. The image's prefix-cache fix (Dockerfile patch 4, see the `PREFIX_CACHE` note in the spin-up script) was validated bit-identical only at MTP=2; the MTP=0 block sizes look uncovered. If you ever need MTP off, turn prefix caching off too.
>
> **▶ LIVE (2026-10-10): spark2 swapped `qwen38-flash` (:8001, stopped) → Decision-2.0-Nox-4B (:8005) to bring the NPC prototype back up. spark1 unchanged (`qwen3.8-flash-next` :8001). Kai :8004 stays stopped, Clef :8002 stays stopped.**
> - NPC prototype = mytools branch `flexai-social-situational` (worktree `~/src/mytools-flexai-social-situational`), `flexai-social/app.py` on :5105. Stances from spark2:8005 (Nox, ~310 ms with a dossier), dialogue line from **spark1**:8001 qwen3.8-flash-next (~2.1 s). Smoke-tested end to end.
> - Reason for the swap: Nox (~19 GB RSS) does not fit beside qwen38-flash (~11 GB host available).
> - **Revert spark2 to qwen3.8-flash-next:** `ssh spark2 'docker stop decision2-nox-4b; docker start qwen38-flash'`.
>
> **(superseded 2026-10-10 for spark2) ▶ LIVE (2026-10-08): BOTH BOXES run `qwen3.8-flash-next` on :8001 again (container `qwen38-flash`, single-box TP=1, identical config). spark2's Decision 2.0 models (Kai :8004, Nox :8005) stopped, not removed; Clef :8002 still stopped.**
> - spark1: `qwen38-flash` restarted (its experiment `vllm-chat` container killed). spark2: Kai + Nox `docker stop`, then `docker start qwen38-flash` (the container left stopped by the 2026-10-07 experiment).
> - **Verified from the boot logs, both boxes:** vLLM `0.1.dev20073+g8e685d198`, `max_seq_len=262144`, MTP-2, `enable_prefix_caching=True` with **attention block size 1600** (the fix is present), `kv_cache_dtype=auto`, reasoning parser `qwen3`, `--restart unless-stopped`. **KV pool: spark1 602,496 tok → 2.30× @ 262K; spark2 590,910 tok → 2.25×.** Host available ~16 GB on each. Coherence check passed on both (Paris / 51).
> - **Revert spark2 to Decision 2.0:** `ssh spark2 'docker stop qwen38-flash; docker start decision2-kai-0-6b decision2-nox-4b'`.
>
> **▶ DONE (2026-10-07): chunked state-docs experiment (CampaignGenerator `experiments/20261007-chunked-state-docs/`, PRs #507/#508, issue #505) borrowed BOTH boxes for the day; everything is back to the lines above — spark1 `qwen38-flash` :8001 (restarted via `spin-up-vllm-qwen38-flash-next.sh`, healthy, block size 1600, max_model_len 262144), spark2 Kai :8004 + Nox :8005 (`docker start`, both `/health` ready).**
> Sequence (all two-box, spark2's decision models stopped for the duration):
> 1. `qwen3.8-flash-next` on spark2:8001 too (container `qwen38-flash`, `HOST=spark2 ./spin-up-vllm-qwen38-flash-next.sh`; block size 1600, KV 574,978 tok).
> 2. `deepseek-ai/DeepSeek-V4-Flash-0731` cross-box TP=2 pair (`spin-up-vllm-dspark-2box.sh`, container `vllm-dspark`, 6 seqs, KV 849,372 tok, coherence PASS, 0 "Skipping unknown").
> 3. `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` one per box (`MAX_SEQS=8 SPEC_TOKENS=2 GPU_UTIL=0.80 ~/spin-up-vllm-qwen3-next-80b-mtp.sh`, container `vllm-chat`, image v0.22.0; **prefix caching OFF** — that image predates the Mamba block-size fix). Coherence PASS.
> - **Measured on the chunked map step (60 chunks of ~60K chars, ch002-070):** Qwen3-Next MTP-2 ×2 boxes ≈ **112 useful tok/s** (30.5 min) — the fastest; DeepSeek-V4-Flash 2-box ≈ 60 (72 min); qwen3.8 ≈ 55 per box, ~90 on two. Details: that experiment's `RESULTS.md` rounds 5-7.
> - **Stopped, NOT removed (restart with `docker start` after stopping the live containers):** `vllm-chat` (Qwen3-Next) on both boxes, `vllm-dspark` on both boxes, `qwen38-flash` on spark2. `dgxlib/models.yaml` already has entries for all three models — no sync needed.
>
> **▶ DONE (2026-10-06→07): paper rerun on a frozen dataset finished (PR #32); spark2 is back to the 2026-10-03 state below — Kai :8004 + Nox :8005 restarted 2026-10-07, Clef :8002 stays stopped, Lux :8003 stays removed.**
>
> **▶ LIVE (2026-10-03): spark2 Clef STOPPED → vLLM Semantic Router "Decision 2.0" decision models on :8005 (Nox-4B — the NPC prototype's backend) and :8004 (Kai-0.6B). Lux-9B (:8003) tested then removed. spark1 unchanged. ~84 GB available on spark2.**
> Image `decision2-runtime`, built on spark2 from `decision2/Dockerfile` (base `clef-server` ONLY for its torch 2.11.0+cu130 / triton 3.6 on sm_121; runtime = semantic-router `src/model-runtime` @ 846e120, `pip install --no-deps`). Start one model per container: `MODEL=Lux-9B PORT=8003 SR_SRC=<semantic-router checkout> ./spin-up-decision2.sh` (`BUILD=1` rebuilds). Full log: `clef-observations.md` (2026-10-03 Decision 2.0 section).
> - **GB10 patch in the image:** the runtime's placement check uses `torch.cuda.mem_get_info`, which on the GB10's unified memory excludes reclaimable page cache (17 GiB "free" with 34 GB available) — Lux refused to load. `decision2/gb10_unified_memory.patch.py` uses `/proc/meminfo` MemAvailable on integrated devices.
> - **CUDA path is upstream-"unvalidated"** (`accelerator_validated: false`, golden check `unverified`, no CUDA kernels — pure-torch fallbacks). Verified here instead: Kai CUDA vs CPU probabilities agree within ~0.005.
> - **Memory: Lux-9B ≈ 50 GB total (33 GB process RSS + device), far above its 16 GB of bf16 weights.** With Clef also loaded the box hit 1 GB available and Lux crashed/restarted once (not cgroup-OOM). Do not co-host Lux with Clef 27B.
> - **Measured median latency on GB10:** Lux 127 ms, Nox 80 ms, Kai 16 ms per request (cards claim 18.4 / — / 4.9 ms on an unnamed GPU).
> **REVERT to Clef:** `ssh spark2 'docker rm -f decision2-lux-9b decision2-kai-0-6b decision2-nox-4b; docker start clef'` (~9.5 min boot). **REVERT to qwen chat:** `ssh spark2 'docker rm -f decision2-lux-9b decision2-kai-0-6b decision2-nox-4b clef'; HOST=spark2 ./spin-up-vllm-qwen38-flash-next.sh`
>
> ---
>
> **▶ PREV — Clef stopped 2026-10-03, see banner above:** **▶ (2026-10-02): spark2 SWAPPED `qwen3.8-flash-next` (:8001) → Cloudflare Clef + Clef-flash decision models (:8002, container `clef`). spark1 unchanged.**
> Brought up with `STOP_CHAT=1 ./spin-up-clef.sh` (run from the workstation). Image `clef-server`, built on spark2 from `clef/Dockerfile` (base `vllm/vllm-openai:latest` ONLY for its torch 2.11.0+cu130 — the release's tested torch; + transformers 5.10.2 + `flash-linear-attention`/`fla-core` `--no-deps`). Full log: `clef-observations.md`.
> - **What these are:** non-generative "decision models" (Cloudflare, Apache-2.0, released 2026-10-01). One prefill pass, then a ~0.25 GB "joint schema head" scores every allowed option of every typed question (`noul`/`choice`/`score`) → calibrated probabilities. API-compatible with TypeSafe's Jev. **vLLM cannot serve them** — the HF card's `vllm serve Cloudflare/clef` is HF's auto-generated widget and would serve a bare chat backbone without the head.
> - **Memory: 68.9 GiB allocated (17.8 clef-flash + ~51 clef), host available ~41-45 GB** — far more headroom than qwen38-flash's ~13 GB. No `--gpu-memory-utilization` analogue: plain torch allocates what it uses.
> - **Boot ~9.5 min** (clef-flash 142s, clef 410s weight load); first request per model pays ~11-22s of Triton JIT. `--restart unless-stopped`.
> - **Measured (warm, single stream, GPU-serialized):** clef-flash **~136 ms** on a 346-tok request (Cloudflare H200: 38.8 ms median), prefill flat **~3,200 tok/s** to 12K; clef **~390 ms** (H200: 209 ms), flat **~1,100 tok/s** to 12K. Deterministic across runs AND across a container restart; model-card example answered correctly and flips correctly when the state flips.
> - **Kernel status:** gated-delta-rule uses fla's Triton kernels (fast); `causal_conv1d` falls back to torch (the "fast path is not available" warning) — small op, not built for sm_121, open lever.
> - **⚠ NOT YET VERIFIED against Workers AI** — the equality check (same requests local vs `@cf/cloudflare/clef`) needs a Cloudflare account token; none on the workstation.
> - **`dgxlib/models.yaml`: no entry** — the registry is for OpenAI-chat calls; Clef has no chat endpoint.
> - **Cost:** spark2 is no longer a second `qwen3.8-flash-next` endpoint, so spark1 has no overflow. No client on the workstation pointed at spark2:8001 (§7 audit).
> **REVERT spark2 to qwen3.8-flash-next** (image + checkpoint still on the box — a load, ~15 min): `ssh spark2 'docker rm -f clef'; HOST=spark2 ./spin-up-vllm-qwen38-flash-next.sh`
>
> ---
>
> **▶ PREV — superseded on spark2 by the 2026-10-02 banner above: Two independent single-box endpoints, now serving the SAME model
> (2026-09-10, later).** The cross-box DSpark pair is gone and both boxes run
> `qwen3.8-flash-next` single-box, TP=1, same image and same flags. They are two
> interchangeable endpoints, not one bigger one — point a client at either.
> `refresh-current-setup.sh` should be run WITHOUT `-C` (cluster mode was only
> for the cross-box DSpark head).
>
> **⚠ The cost of this: there is no longer a second opinion on the box.** Until
> 2026-09-10 (earlier) spark2 held `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` as the
> known-good fallback and A/B partner. It doesn't any more. The revert is cheap
> and was verified before the swap — the 77 GB Qwen3-Next checkpoint and
> `~/spin-up-vllm-qwen3-next-80b-mtp.sh` are both still on spark2, so restoring
> it is a load, not a download (~15 min):
> `ssh spark2 'docker rm -f qwen38-flash'; ssh spark2 'PREFIX_CACHING=1 MAX_SEQS=8 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`
>
> **Previous config (restore target):** both boxes single-box
> `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`, 256K, util 0.80, MTP-2 + APC,
> fp8 KV, vLLM 0.22.0 — spark1 seqs 3 (latency), spark2 seqs 8 (Cognee batch).
> **ctx = total context per request (prompt + generation), i.e. `--max-model-len`.**
> **⚠ HISTORICAL — the split described in the rest of this paragraph ended on
> 2026-09-10 (later). Both boxes now run the SAME config: `qwen3.8-flash-next`,
> 262K, util 0.80, seqs 8, MTP-2 + APC, bf16 KV. There is no latency box and no
> batch box any more. Kept because it describes the restore target above.**
> **The two boxes ran DIFFERENT configs by design (2026-07-03):** **spark1 = LATENCY** — MTP-2 latency build + APC prefix caching (`--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`, `--enable-prefix-caching`, `--max-num-seqs 3`, chunked prefill pinned at `--max-num-batched-tokens 40960`; MTP draft acceptance ~77% on a hard code prompt / higher on prose, APC gives ~5–10× TTFT on a re-sent prefix). **spark2 = MTP-8 BATCH** (as of 2026-07-04, later) — MTP-2 spec decode + APC + `--max-num-seqs 8`, chunked prefill 40960; measured fastest on the Cognee JSON load (~138 t/s vs plain-16's ~125). Point latency/agent-loop work at spark1:8001, batch/structured-output jobs at spark2:8001 — same served model id on both. (spark2 was plain-seqs16 "THROUGHPUT" until the MTP-8 crossover result; revert command in the LIVE banner if a long-context batch job needs the higher KV.) See the LIVE banner immediately below.

> **▶ LIVE (2026-09-10, later): spark2 SWAPPED `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` → `qwen3.8-flash-next`. BOTH BOXES NOW RUN THE SAME MODEL, single-box TP=1, same image, same flags, byte-identical checkpoint.**
> Brought up with `HOST=spark2 ./spin-up-vllm-qwen38-flash-next.sh` (the wrapper is now box-aware; it was hardcoding spark1's IP in its summary).
>
> - **spark2 was stocked from spark1 over the 10.100.16.x cable, NOT from HuggingFace.** It had none of the prerequisites — no recipe clone, no `qwen38-flash-dgx` image, no checkpoint. Copying beats re-downloading on three counts: it skips the ~126 GiB HF pull, it skips the ~10 min `prepare-hybrid.sh` rewrite (the prepared `-fp8hybrid` snapshot comes along), and it guarantees spark2 runs **the same bytes that were validated on spark1 today** rather than a re-derived checkpoint. Commands in §8. **Measured: image 20.7 GB in 3m48s, checkpoint 139 GB in 4m51s, ~425 MB/s** (ssh/aes128-gcm-bound, not cable-bound — the cable itself is ~110 Gb/s).
> - **⚠ One file needs root and `tar` as `kostadis` silently can't read it.** The HF cache is root-owned (it is written by the download container), and `hub/models--…/trees/<rev>.json` is mode **`600`** — the only file in 139 GB that a user-level `tar` cannot open. It fails ONE file, keeps going, and exits non-zero at the very end; in a pipeline without `pipefail` that status is discarded, so **the copy looks clean**. Caught by comparing byte counts (a 94,865-byte gap), then copied through a container. `trees/` is xet dedup metadata and is not read when vLLM is pointed at an explicit snapshot path with `HF_HUB_OFFLINE=1`, so this was cosmetic — but the failure mode (partial copy reported as success) is not.
> - **Verified byte-identical, not assumed:** parallel `md5sum` over all 428 files on both boxes → `671788c9611b59321593c21741e16f42` on spark1 and spark2. A matching `du` was not treated as sufficient.
> - **Config verified live from the engine's own log** (not the flags we passed): served id `qwen3.8-flash-next`, `quantization=modelopt_fp4`, `max_seq_len=262144`, `max_num_seqs=8`, `enable_prefix_caching=True`, `kv_cache_dtype=auto`, `SpeculativeConfig(method='mtp', num_spec_tokens=2)`, `--tool-call-parser qwen3_coder`, `--reasoning-parser qwen3`, vLLM `0.1.dev20073+g8e685d198` — **all identical to spark1**. **`--restart unless-stopped`.**
> - **✅ THE PREFIX-CACHING LANDMINE RE-CHECKED ON THIS BOX.** APC is only safe on an image carrying the Mamba block-size fix (see the PREV banner). Confirmed in spark2's own boot log: `Setting attention block size to 1600 tokens to ensure that attention page size is >= mamba page size` — **1600, not 8** — plus `Mamba cache mode is set to 'align'`. This check is cheap and the failure is invisible; re-run it on every box, every image rebuild.
> - **KV pool 576,427 tok → 2.20× @ 262K** (spark1: 553,254 → 2.11×). **Weights on card 76.75 GiB — exact match with spark1.** Boot 14.4 min (weights 620s + draft 80s, init 128s incl. 38s compile), against spark1's 14.5 min.
> - **Measured on spark2 (prefill first, per `user_workflow_read_heavy`; unique-nonce prompts so APC cannot fake a cold number).** Cold prefill **1,239 tok/s**; **warm plateau 2,256 / 2,457 tok/s at ~8.2K and 2,442 / 2,450 tok/s at ~36K — flat, no long-context cliff**; decode **32.0 / 43.4 tok/s** end-to-end on 900-token greedy generations. **APC 8.61s → 1.14s = 7.6× TTFT, proven by the hit counter (0 → 8000), not a stopwatch.** **Determinism: first-token logprobs identical across runs (`DET_TOPK=1`).** Every number is within run-to-run noise of spark1's (cold 1,137; warm 2,417–2,494; decode 30.6–37.2; APC 7.1×).
> - **Tool calling: PASS** — a real `tools`-bearing request returned `finish_reason: "tool_calls"` with a structured array and valid JSON arguments `{"location": "Paris"}`, `content` empty (no syntax leaked as text).
> - **Reasoning-field shape: IDENTICAL to spark1, so the same client trap applies.** Thinking is ON by default (28 completion tokens on "Reply with exactly: OK"), the trace lands in **`reasoning`** with **`reasoning_content` null** — the shape opencode and openclaw silently drop (`todo_nano_v3_reasoning_leak`). `chat_template_kwargs: {"enable_thinking": false}` → **2 tokens, no reasoning**. Verified on this box, not carried over.
> - **Ollama on spark2 :11434 untouched** — live 1024-dim vector returned after the swap, so the MemPalace embedding path is unaffected.
> - **`dgxlib/models.yaml` needs no new entry** — it keys on the served model id, and that id is unchanged. Only the entry's comment was widened from "on spark1" to both boxes.
> - **⚠ NOT MEASURED: cross-box output equality.** The obvious check — same greedy prompt to both boxes, compare text — was attempted and abandoned as uninformative: spark1 was saturated at the time (8 running / 31 deferred) while spark2 was idle, and vLLM's numerics depend on batch composition, so a mismatch would not have implied a bad copy. The md5 tree match is the stronger evidence and it is exact.
> - **Incidental finding that argues for this swap:** spark1 was sitting at **8 running / 31 deferred requests** during verification — exactly the multi-client contention the PREV banner flags as "NOT TUNED". A second identical endpoint is somewhere for that overflow to go.
> - **VERDICT: infra verified on both boxes; subjective quality still not judged.** The swap did not change the model, only how many boxes serve it. **What it did cost is the A/B partner** — there is no longer a different model running locally to compare against.
> **REVERT spark2 to Qwen3-Next-80B** (weights still cached on the box — a load, not a download): `ssh spark2 'docker rm -f qwen38-flash'; ssh spark2 'PREFIX_CACHING=1 MAX_SEQS=8 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`.
>
> ---
>
> **▶ PREV (2026-09-10, earlier — spark1's bring-up; this config is STILL CURRENT on spark1, superseded only in that spark2 now matches it): cross-box DSpark TORN DOWN on BOTH boxes → spark1 now serves `qwen3.8-flash-next` (Qwen3.8-Flash-Next, NVFP4+fp8 hybrid) SINGLE-BOX; spark2 restored to `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` (MTP-2 + APC, seqs 8). Two independent endpoints again.**
> Brought up with `./spin-up-vllm-qwen38-flash-next.sh` (run from the workstation) — a thin wrapper over the **blazux recipe**, cloned on spark1 at `~/qwen3.8-Flash-DGX` (github.com/blazux/qwen3.8-Flash-DGX, commit `bd60fcb1`, 2026-09-09). The recipe's `scripts/serve.sh` owns the docker invocation and the ten image patches; the wrapper owns only our choices (box, port 8001, profile) plus preflight/verify. **Do not fork `serve.sh`** — `git pull && docker build` on the box is the whole update path.
>
> - **Why this recipe and not the other one.** Two community recipes exist for this model on GB10. The first one evaluated (`krisitown/qwen38-flash-next-nvfp4-dgx-spark`) ships `GPU_MEMORY_UTILIZATION=0.9` — against `feedback_gpu_util_080_default` — a 186.5 GB checkpoint needing ~238 GB on disk after the PLE assembly, and, critically, offers `ENABLE_PREFIX_CACHING` on an image that does **not** carry the Mamba block-size fix. blazux's is newer (2026-09-08), uses a 126 GiB checkpoint, defaults to `GPU_MEM=0.80` **for the same unified-memory reason we independently derived**, and was independently reproduced on another Spark (@jschmied).
> - **⚠ THE LANDMINE WE AVOIDED — prefix caching silently corrupts this model on an unpatched image.** vLLM's EngineCore overwrites `cache_config.block_size` with the *smallest* KV-group block size (8 tokens at MTP=2 — the QSA raw-key ring) while the Mamba state block is **1600**. Two call sites used the former as the latter, so a prefix-cache hit computed the state slot as `(3200-1)//8 = 399` instead of `1`, read past the block-table row, and restored an **all-zero Mamba state**: no crash, no warning, silently different answers on every cache hit — **invisible to a coherence smoke test**, exactly like the DSpark missing-tool-parser bug. Given `reference_apc_hybrid_qwen3_next` (APC was a ~10× TTFT win on Qwen3-Next) we would certainly have enabled it. blazux's image carries the two-line fix. **Confirmed live in our own boot log:** `Setting attention block size to 1600 tokens to ensure that attention page size is >= mamba page size` — 1600, not 8. **Never enable prefix caching for this model on any other qwen38-flash-next image without checking for that fix.**
> - **Config verified live** (from the engine's own log, not the flags we passed): served id `qwen3.8-flash-next` (**LOWERCASE — clients MUST be repointed**), `quantization=modelopt_fp4`, `max_seq_len=262144`, `gpu_memory_utilization=0.8`, `max_num_seqs=8`, `enable_prefix_caching=True`, `Mamba cache mode is set to 'align'`, `speculative_config=SpeculativeConfig(method='mtp', num_spec_tokens=2)`, `kv_cache_dtype=auto` (bf16), `cudagraph_mode=PIECEWISE` with `vllm::ple_mmap_lookup` as a splitting op, `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`, image `qwen38-flash-dgx` (local build, vLLM `0.1.dev20073+g8e685d198`). **`--restart unless-stopped` — this container DOES survive a reboot, closing the gap DSpark left open since 2026-07-30.**
> - **The PLE mmap is what makes it fit.** `PLE mmap: layer 1, 128 shards, 320001536 rows x 160 B (47.7 GiB on disk), dtype F8_E4M3, 32 workers`. A token's n-gram lookup reads 16 rows ≈ 2.5 KB, so the 47.7 GiB table is served from NVMe through the page cache instead of the unified pool. **Weights on card: 76.75 GiB** (blazux measured 77.83 GiB — close match). Boot **14.5 min** total (weights 628s + draft 80s, then compile 34s, then KV alloc).
> - **GPU KV pool 553,254 tok → 2.11× @ 262K.** Lower than blazux's ~626k at YaRN-500k; not investigated, and not a problem at seqs 8.
> - **Measured (prefill first, per `user_workflow_read_heavy`; unique-nonce prompts so APC cannot fake a cold number).** **Cold-cache first pass ~1,137 tok/s** — the recipe's own documented "2–3× slower on a cold region of the table" case. **Warm plateau 2,417 / 2,452 / 2,494 tok/s at ~10.5K, and 2,408 / 2,410 tok/s at ~46K — flat, no long-context cliff.** **Steady-state decode 30.6 and 37.2 tok/s** (900-token greedy generations); vLLM's own counter read 32.0–34.7 tok/s generation and 4,584–4,625 tok/s prompt throughput during those runs. **APC: 9.38s → 1.33s on a re-sent ~10.6K prefix = 7.1× TTFT, proven by `vllm:prefix_cache_hits_total` 0 → 8000, not a stopwatch.** **Determinism: first-token logprobs identical across runs (`DET_TOPK=1`).**
> - **vs the DSpark it replaced:** prefill **2,400–2,500 warm vs ~1,000–1,180 (~2.1×)**, and at 44K specifically 2,410 vs 1,177 (2.05×); decode 30.6–37.2 vs ~21–31; context 262K vs 256K (parity); **and it needs ONE box, not two** — which is what gave spark2 back. Cold-cache prefill is only parity (~1,137 vs ~1,000–1,180), so the win is real but conditional on a warm page cache: consider `PREWARM=1` (streams the table once at boot, ~10s) if first-request latency after a restart matters.
> - **Tool calling: PASS** — a real `tools`-bearing request returned `finish_reason: "tool_calls"` with a structured array and valid JSON arguments `{"location": "Paris"}`, not syntax leaked as text.
> - **⚠ REASONING-TRACE FIELD, and thinking is ON by default.** This build populates **`reasoning`** and leaves **`reasoning_content` null** — the same shape as `nano_v3`/DeepSeek-R1 that opencode *and* openclaw silently drop (`todo_nano_v3_reasoning_leak`; verified by reading openclaw's own bundle — it looks up `reasoning_content`, 5 sites). The trace is billed either way: **24 of 28 completion tokens (86%) on a trivial "Reply with exactly: OK"**. **The switch that works is `chat_template_kwargs: {"enable_thinking": false}`** → 2 tokens, 0 reasoning. **`/no_think` in the prompt does NOT work on this model** — it is echoed into the answer verbatim and the trace still fires (41 tok, 33 reasoning); the upstream recipe's own `smoke-test.sh` uses it, so its decode benchmark is measured with thinking ON. `dgxlib/models.yaml` sets `can_think: true, thinking_default: false`, which emits exactly that kwarg via `extra_body`.
> - **⚠ NOT TUNED, known lever:** multi-client prefill stalls. With two or more agents live, a *decoding* client drops to ~0.2 tok/s for a minute or two while another client prefills a cold long prompt — structural to vLLM chunked prefill (every step carrying a prefill chunk carries exactly one token per decoder), not a recipe bug. `EXTRA='--long-prefill-token-threshold 1024'` lifts the stalled client to ~1.0 tok/s at the cost of ~36% of an 8K TTFT. Left OFF for the single-main-user case. Also unapplied: the recipe suggests `vm.swappiness=10` (spark1 is at the Spark default 60).
> - `dgxlib/models.yaml` **UPDATED** (new `qwen3.8-flash-next` entry). ⚠ The lowercase id does **not** match the `Qwen/Qwen3` prefix entry (`registry.py:93` `startswith` is case-sensitive), so without an explicit entry it fell through to `default` at **idle_timeout 120** — which would kill long-context calls mid-prefill. Verified with `resolve_model_config()` before and after.
> - **Ollama on :11434 untouched on both boxes** — verified live end-to-end (1024-dim vectors from spark1:11434), so the MemPalace embedding path worked through the whole swap.
> - **VERDICT: not yet judged.** Infra is verified; subjective quality against the DSpark bar has not been assessed. spark2 holds the known-good Qwen3-Next-80B as the A/B partner and fallback. **[NO LONGER TRUE — spark2 was swapped to `qwen3.8-flash-next` later the same day; see the LIVE banner above. Left as written because this banner is the record of what was true at the spark1 bring-up.]**
> **REVERT to the cross-box DSpark pair** (checkpoint still on disk on both boxes — a load, not a download): `ssh spark 'docker rm -f qwen38-flash'; ssh spark2 'docker rm -f vllm-chat'` then `./spin-up-vllm-dspark-2box.sh` from the workstation.
> **REVERT spark1 to Qwen3-Next-80B:** `ssh spark 'docker rm -f qwen38-flash'; ssh spark 'PREFIX_CACHING=1 MAX_SEQS=3 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`.
>
> ---
>
> **▶ PREV (2026-08-04, superseded by the 2026-09-10 banner above): cross-box endpoint UPGRADED `deepseek-ai/DeepSeek-V4-Flash-DSpark` (preview) → `deepseek-ai/DeepSeek-V4-Flash-0731` (DeepSeek's official V4-Flash release, 2026-07-31). Same TP=2 / mp / spark1-head+spark2-worker topology, same image. Not a new experiment — this is a checkpoint upgrade on top of the already-ADOPTED (2026-08-03) config.**
> Feasibility confirmed BEFORE migrating (not assumed from community recipes, which disagreed with each other and in two cases targeted a different image lineage): read the actual `config.json` of both checkpoints (byte-identical architecture, 284B/13B active, same quant format, `dspark_block_size: 5`), read the installed vLLM source inside the running preview container (`vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/tokenizers/deepseek_v4.py`/`deepseek_v4_encoding.py`) rather than trusting three mutually-inconsistent community write-ups, and live-checked the preview container's own logs (`grep -c "Skipping unknown"` = 0 on both ranks) before concluding the community-reported "silent weight-drop" bug on a different image lineage did not apply to ours. Full research trail: `deepseek-v4-flash-dspark-observations.md` (2026-08-03 entry).
> - **What changed in `spin-up-vllm-dspark-2box.sh`:** `MODEL` default → `deepseek-ai/DeepSeek-V4-Flash-0731`; `SPEC_TOKENS` default **3 → 5** (matches `dspark_block_size`; 3 booted and served on the preview but wasn't the checkpoint's native block size); added `VLLM_USE_BREAKABLE_CUDAGRAPH=0` (a real, confirmed-recognized flag in this image — not flagged as "Unknown vLLM environment variable" in the boot log, unlike `VLLM_BUILD_COMMIT`/`VLLM_BUILD_PIPELINE`/etc which genuinely are unrecognized noise). **Unchanged:** image tag (`0.1.1` already met 0731's vLLM≥0.25.0 requirement), tool-call parser (`deepseek_v4`), `gpu-memory-utilization` (0.80).
> - **Checked and NOT an issue:** 0731 introduces a `reasoning_effort` scheme (`low`/`high`/`max`, `low` = new cheap default) that this image's tokenizer wrapper predates. Traced the actual code path (not assumed): an **omitted** `reasoning_effort` (what every current client sends) passes through as `None` and is unaffected — only an explicit `reasoning_effort="low"` would be mis-mapped to `"high"` by the old wrapper, and nothing sends that today. Narrower than initially suspected; corrected before writing this banner.
> - **Weight-mapping sanity: 0 "Skipping unknown" warnings on both ranks**, live-checked immediately after boot (the exact class of bug a different DSpark image lineage hit on this same 0731 checkpoint, per community reports) — confirms the loader path verified against the preview also loads 0731's actual tensors correctly, not just in theory.
> - **KV pool 869,357 tok → 3.32× @ 256K** — matches the preview's 845,284-859,040 tok / 3.22-3.28× within run-to-run variance, as expected for an identical architecture.
> - **Coherence smoke: PASS.** **Tool calling: PASS** — a real `tools`-bearing request returned a proper structured `tool_calls` array (`finish_reason: "tool_calls"`, valid JSON arguments `{"location": "Paris"}`), not syntax leaked as text.
> - **Single-stream decode sanity (3 runs, 600-token generations, non-streaming so no chunk-undercounting): 21.0 / 31.2 / 23.5 tok/s (~25 avg).** Noisy, but in the same range as the preview's single-point 30.3 tok/s baseline — **not a controlled A/B**, just enough to confirm no gross regression. A proper same-prompt comparison against the preview is still open. Host memory at time of test: **14 GB / 11 GB available** (spark1/spark2) — consistent with the preview's documented headroom, no new risk observed.
> - `dgxlib/models.yaml` **UPDATED** (new `deepseek-ai/DeepSeek-V4-Flash-0731` entry alongside the retained preview entry).
> **Revert to the DSpark preview** (checkpoint still on disk on both boxes — a container restart, not a re-download): `MODEL=deepseek-ai/DeepSeek-V4-Flash-DSpark SPEC_TOKENS=3 ./spin-up-vllm-dspark-2box.sh` (run from the workstation).
>
> ---
>
> **▶ PREV (ADOPTED 2026-08-03, deployed 2026-07-30, superseded by 0731 above): BOTH boxes swapped Qwen3-Next-80B (single-box each) → ONE cross-box `deepseek-ai/DeepSeek-V4-Flash-DSpark` (TP=2, `mp` backend, NO Ray). spark1 = head + API on :8001; spark2 = headless worker, no API. This banner's config was replaced in-place by the 0731 upgrade above; kept for history.**
> Brought up with `./spin-up-vllm-dspark-2box.sh` (run from the workstation). Full detail: `deepseek-v4-flash-dspark-observations.md`; plan: `deepseek-v4-flash-dspark-2box-plan.md`.
> - **Config verified live:** served id `deepseek-ai/DeepSeek-V4-Flash-DSpark` (**clients MUST be repointed — different id from the Qwen it displaced**), image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (`sha256:a8394849…`, vLLM `0.25.2.dev0+g752a3a504`, torch 2.11.0+cu130 — **not** our pinned `v0.22.0-aarch64`), `--nnodes 2 --tensor-parallel-size 2 --distributed-executor-backend mp`, `--master-addr 10.100.16.1 --master-port 25440` over the DAC cable (TCP sockets, **no RoCE**), 256K ctx, util 0.80, seqs 6, `--kv-cache-dtype nvfp4_ds_mla`, `--block-size 256`, `--moe-backend flashinfer_b12x`, `--async-scheduling`, spec `{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}`, `enable_prefix_caching=True` (**on by default in this build, not passed**), `quantization=deepseek_v4_fp8` / `scale_fmt=ue8m0`, and (**added 2026-07-30, second bring-up**) `--enable-auto-tool-choice --tool-call-parser deepseek_v4` on the HEAD only. Boot ~5.5 min (weights warm in page cache). Coherent smoke ("The capital of France is Paris." + correct 1-10 count) — **no gibberish**, the known failure mode did not appear.
> - **⚠ TOOL CALLING was MISSING on the first bring-up and is now FIXED (2026-07-30).** The original `spin-up-vllm-dspark-2box.sh` passed no tool flags — unlike every other `spin-up-vllm-*.sh` in this repo — so vLLM returned **HTTP 400 `'"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'` on any request carrying a `tools` array**, while plain completions worked fine. That makes every agent client (openclaw, opencode) fail before inference with a misleading "provider rejected the request schema" message, and it is invisible to a coherence smoke test. Fixed by adding `--enable-auto-tool-choice --tool-call-parser deepseek_v4` to the HEAD invocation (`TOOL_PARSER` env override; HEAD only — the worker is `--headless` with no API server). This image offers `deepseek_v3/_v31/_v32/_v4`; **`deepseek_v4` is correct for V4-Flash** — verified by a real tool call returning a structured `tool_calls` array with `finish_reason: "tool_calls"` and valid JSON arguments, **not** tool syntax leaked as plain text (the wrong-parser signature). Confirmed in vLLM's own boot line: `'enable_auto_tool_choice': True, 'tool_call_parser': 'deepseek_v4'`.
> - **GPU KV pool 859,040 tok → 3.28× @ 256K.** (Second bring-up with tool flags reported **845,284 tok → 3.22×** — ~1.6% lower, run-to-run host-memory variance, not a tool-flag cost. The 0.82×-of-1M conclusion below is unchanged.) The recipe expected ~1.9–2.04M. **859K ÷ 1,048,576 = 0.82×, so the recipe's `--max-model-len 1048576` would NOT have booted at util 0.80** — starting at 256K per `feedback_size_context_by_kv_pool` avoided a failed bring-up. 1M needs util ≥0.85, which collides with the host-memory risk below.
> - **Measured (prefill first, per `user_workflow_read_heavy`; unique-nonce prompts so APC can't fake a cold number).** Cold prefill is **LINEAR and flat ~1,000–1,180 tok/s out to 176K** — no long-context cliff, which **contradicts the 1-box thread's alarming ~13 min at 250K** (flat 1,045 tok/s extrapolates to ~4 min): 11,100 tok → **11.04s / 1,005 t/s**; 43,999 tok → **37.39s / 1,177 t/s**; 176,428 tok → **168.83s / 1,045 t/s**. **APC scales with context and is transformative at length: 17.9× at 176K (168.83s cold → 9.45s warm)**; 5.7× at 44K; only 1.3× at 11K (anomalous, needs re-run). Single-stream decode **30.3 tok/s — ~half the advertised 60–67**, and below spark1's ~49 t/s MTP Qwen3-Next. Draft acceptance **~63%** (mean accept length 2.89 of 4; per-position 0.857/0.619/**0.419** — the 3rd draft token is mostly wasted, so `num_speculative_tokens=2` is a cheap A/B) vs **95–99%** for our own Qwen3-Next MTP-2.
> - **Real-usage throughput-mode decode (2026-08-03, user-reported from actual concurrent use, not a controlled A/B): ~100+ tok/s aggregate at 3 concurrent streams.** Any single stream still tops out at the 30-40 tok/s measured above, but this deployment is run in throughput mode against the 6-slot cross-box endpoint, where concurrent streams compound. This aggregate number — not the single-stream figure — is what actually matters for how the box is used, and it's the throughput half of the keep decision below.
> - **⚠ HOST MEMORY RISK:** ~14-15 GB available idle, ~11-12 GB at 44K prefill, **~9-10 GB at 176K prefill** — at/below the headroom that once wedged spark2 into needing a physical reboot (`feedback_gpu_util_080_default`). util 0.80 with a 156 GB 2-box model is NOT the comfortable 0.80 we know from an 80 GB single-box model. **Do not raise to 0.85 for 1M context without accepting reboot risk**; consider 0.75 if this becomes a keeper. Both boxes stayed ssh-responsive throughout (watched with a low-memory alarm).
> - **Tuning lever not yet tried:** vLLM warns at boot that `max_num_scheduled_tokens` is clipped to **8180** to make room for draft slots, and to raise `--max-num-batched-tokens`. The recipe's 8192 is a small chunked-prefill window — our Qwen3-Next builds use **40960**. That's the most promising single knob for the prefill axis we care about.
> - **⚠ Recipe corrections (the published recipes are wrong on 3 points)** — worker MUST get `--headless` and a distinct `--node-rank` ("identical command on both nodes" is false); node discovery is `--master-addr`/`--master-port` CLI flags, not env vars; and 5 of the 9 "not optional" env vars don't exist in this image. Proof in the header of `spin-up-vllm-dspark-2box.sh`.
> - **Ollama on :11434 untouched on both boxes** — it's a systemd service, unaffected by stopping the `vllm-chat` containers, so the `qwen3-embedding:0.6b` MemPalace path kept working through the swap. `:8000` (`vllm-embed`) is not running on either box and was not involved.
> - `dgxlib/models.yaml` **UPDATED** (new `deepseek-ai/DeepSeek-V4-Flash-DSpark` entry: `read_timeout: 1800`, `idle_timeout: 420` — the wide idle budget covers a ~2-min cold prefill at 128K before any token streams).
> - **VERDICT: ADOPTED — KEEPER (decided 2026-08-03).** *For:* flat long-context cold prefill (no cliff), a **17.9× APC payoff at 176K** (exactly the read-heavy "load context once, live in it" shape), **~100+ tok/s aggregate decode in throughput mode at 3 concurrent streams** (the single-stream 30.3 tok/s figure undersold how this box is actually used), and — the deciding factor — **subjective model quality is a real, noticeable improvement** over the Qwen3-Next-80B baseline in actual use. *Against, accepted knowingly:* single-stream decode still ~half spark1's old ~49 tok/s Qwen3-Next MTP figure, draft acceptance 63% vs 95-99%, no context-multiple advantage (3.28× vs 3.56×), it consumes **both** boxes so no separate batch endpoint remains, and host headroom sits at 9-12 GB under load (watch for the `feedback_gpu_util_080_default` wedge risk). Net: two boxes for one endpoint, permanently, bought with better long-context prefill, real throughput-mode decode, and better output quality. **Outstanding follow-up, not blocking the keep decision:** repoint the four still-broken clients (§7) and add a `--restart unless-stopped` policy to both containers (currently absent — see §5 port table).
> **REVERT to the two-box Qwen3-Next world:** `ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'` (spark1 latency: `PREFIX_CACHING=1 MAX_SEQS=3 GPU_UTIL=0.80 SPEC_TOKENS=2`) and `ssh spark2 'PREFIX_CACHING=1 MAX_SEQS=8 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`. Weights are cached — a load, not a download. **Also stop the DSpark containers first:** `ssh spark 'docker rm -f vllm-dspark'; ssh spark2 'docker rm -f vllm-dspark'`.
>
> ---
>
> **▶ PREV (2026-07-04, later): spark2 `vllm-chat` swapped plain-seqs16 → MTP-2 + APC + seqs 8 after a crossover experiment on the Cognee batch. This config MEASURED FASTEST for that workload. spark1 untouched (still MTP-2 + APC latency/production).**
> Purpose: a 3-way throughput A/B on the real Cognee graph-extraction load (decode-bound, JSON-heavy, short contexts) to find the batch sweet spot between the latency box (MTP, seqs 3) and the throughput box (plain, seqs 16).
> - **Measured aggregate generation throughput on the Cognee load:** MTP seqs 3 = **92 t/s**; plain seqs 16 = **~125 t/s**; **MTP seqs 8 = ~138 t/s (WINNER, +10% over plain-16, +50% over MTP-3).** Both extremes were on the wrong side of the crossover.
> - **Why MTP wins here:** MTP draft acceptance on the rigid `KnowledgeGraph` JSON is **86–89%** (mean accept length ~2.75 of 3, per-position 0.93/0.82) — far above the ~77% on hard code. So each MTP stream decodes ~2.2× faster; 8 fast streams (138) beat 16 bandwidth-contended plain streams (125), and rejected-draft compute waste hasn't bitten by seqs 8. Enabled via `ssh spark2 'PREFIX_CACHING=1 MAX_SEQS=8 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`.
> - **Verified live:** served id unchanged (`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` → **no client repoint**), `speculative_config={qwen3_next_mtp, num_speculative_tokens:2}`, `enable_prefix_caching=True`, `max_num_seqs=8`, chunked prefill 40960, fp8 KV, util 0.80, 256K. **GPU KV pool 933,232 tok → 3.56× @ 256K** (DOWN from plain-16's 6.41× — the MTP drafter + Mamba align cache eat KV). Reload ~17 min.
> - **WORKLOAD-SPECIFIC — do not generalize.** MTP-8 wins for Cognee's shape (short ctx, JSON, decode-bound). A long-context / high-concurrency batch job (e.g. pdf-translation at ~120K ctx) may still prefer plain-16's 6.41× KV headroom and 16 slots. Pick by workload: JSON/structured-output decode → MTP-8; long-context or >8-way concurrency → plain-16.
> - `dgxlib/models.yaml` **unchanged** (served id identical).
> **Revert spark2 to plain-16 throughput (higher KV, 16 slots):** `ssh spark2 'PREFIX_CACHING=1 bash ~/spin-up-vllm-qwen3-next-80b.sh'`.
> **Re-run this MTP-8 config:** `ssh spark2 'PREFIX_CACHING=1 MAX_SEQS=8 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`.
>
> ---
>
> **▶ PREV (2026-07-04): spark2 `vllm-chat` re-spun with APC PREFIX CACHING ON, keeping the throughput config (seqs 16, no spec decode). spark2 is now THROUGHPUT + APC — the batch box for prefix-sharing workloads. spark1 untouched (still MTP-2 + APC latency/production).**
> Purpose: the overnight Cognee grounding-digestion job (108 docs, ~39-way concurrent graph-extraction) needs a batch box AND resends a byte-stable KnowledgeGraph-extraction system prompt on every chunk call — exactly what APC caches. Enabled via `ssh spark2 'PREFIX_CACHING=1 bash ~/spin-up-vllm-qwen3-next-80b.sh'` (deployed script verified current first: seqs 16 / util 0.80 / 256K defaults, PREFIX_CACHING knob present).
> - **Verified live (docker logs + `/v1/models`):** served id `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` (**unchanged → no client repoint**), `enable_prefix_caching: True`, *"Mamba cache mode is set to 'align'"*, `speculative_config=None`, `max_num_seqs: 16`, fp8 KV, util 0.80, 256K. **GPU KV pool 1,679,392 tok → 6.41× @ 256K** (UP from the no-APC build's 6.11× — APC costs ~nothing here; the drafter is what eats KV, and there's no drafter on the throughput build). `Application startup complete`. Reload took **~17 min** (disk-bound weight load).
> - **The box split is unchanged in spirit:** **spark1 = LATENCY** (MTP-2 + APC, seqs 3); **spark2 = THROUGHPUT + APC** (plain, seqs 16, prefix caching on). Both serve the identical model id — any client works against either.
> - **CALIBRATION NOTE (2026-07-04): APC turned out IMMATERIAL for the Cognee batch it was enabled for.** That workload is output-decode-bound (short ~1k-token shared prefix ≈ 1s prefill, then a multi-thousand-token JSON decoded at ~4 t/s/stream = 30-60s+), so caching the prefix saves <3% even on a perfect hit — measured ~0% hit rate (the batch fires as a cold simultaneous burst, and APC accelerates prefill, not decode). **The batch speedup vs the earlier spark1 attempt was the 16 seqs, NOT APC.** APC left ON because it's free here (KV 6.41× ≥ the no-APC 6.11×, seqs unchanged), but don't attribute throughput to it. APC's real payoff is prefill/TTFT-bound work (agent loops re-sending long context) — which is why it stays on spark1 (latency).
> - `dgxlib/models.yaml` **unchanged** (served id identical; APC changes prefill speed, not how the model is *called*).
> **Revert spark2 to plain throughput (no APC):** `ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b.sh'`.
> **Revert spark2 to latency (MTP-2 + APC):** `ssh spark2 'PREFIX_CACHING=1 MAX_SEQS=3 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`.
>
> ---
>
> **▶ PREV (2026-07-03): spark2 `vllm-chat` swapped MTP-2 + APC (latency) → PLAIN THROUGHPUT (seqs 16, no spec decode, no APC) — spark2 is now the dedicated BATCH box. spark1 untouched (still MTP-2 + APC latency/production).**
> Purpose: keep a high-concurrency endpoint standing for batch jobs while spark1 stays the low-latency single-stream box. Swapped via `ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b.sh'` (deployed script verified current first: seqs 16, util 0.80, 256K, image `v0.22.0-aarch64`, `--restart unless-stopped`).
> - **Verified live (docker logs + `/v1/models`):** served id `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` (**unchanged → no client repoint**), `max_num_seqs: 16`, `speculative_config=None`, `enable_prefix_caching=False`, fp8 KV, util 0.80, 256K. **GPU KV pool 1,601,403 tok → 6.11× @ 256K** (up from the APC+MTP build's 3.54× — dropping the MTP drafter + Mamba align cache frees KV). Coherent smoke (*"The ocean is a vast, dynamic expanse of saltwater that covers over 70% of Earth's surface..."*), `reasoning:null`, no `<think>` leak. Host RAM **~13 GB available** (healthy). Restart took **~17 min** (disk-bound weight load — slower than the usual ~10).
> - **The box split is now intentional:** **spark1 = LATENCY** (MTP-2 + APC, seqs 3, ~49 tok/s single-stream) for agent-loop / read-heavy work; **spark2 = THROUGHPUT** (plain, seqs 16, 6.11× KV) for batch / high-concurrency. Both serve the identical model id, so any client works against either — the difference is latency vs aggregate-throughput characteristics.
> - `dgxlib/models.yaml` **unchanged** (served id identical; seqs / spec-decode / APC change how *fast* it serves, not how it's *called*).
> **Revert spark2 to latency (MTP-2 + APC):** `ssh spark2 'PREFIX_CACHING=1 MAX_SEQS=3 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`.
>
> ---
>
> **▶ PREV (2026-07-02): spark1 powercycled overnight and came back with `vllm-chat` DEAD (`Exited (255)` — the container had no restart policy). Relaunched on the unchanged APC+MTP config, and the failure class is FIXED: `vllm-chat` now runs `--restart unless-stopped` on BOTH boxes.**
> - **Relaunch:** `PREFIX_CACHING=1 MAX_SEQS=3 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh` (deployed script verified byte-identical to the repo copy first). Config, served id, and client wiring all unchanged — **the 2026-07-01 banner below remains the accurate description of what both boxes serve.** spark2 was verified healthy throughout the incident (chat + embed answering).
> - **Auto-restart fix:** applied `docker update --restart unless-stopped vllm-chat` to the live containers on spark1 AND spark2 (no container restart needed), and both spin-up scripts (`spin-up-vllm-qwen3-next-80b.sh`, `spin-up-vllm-qwen3-next-80b-mtp.sh`) now default `RESTART_POLICY=unless-stopped` and pass `--restart` (updated copies pushed to both boxes). A reboot/powercycle now brings the chat slot back automatically (~5–10 min weight reload + compile before it answers). `docker stop`/`docker rm -f` still stick — a manually stopped container stays stopped — so the swap scripts and experiments behave exactly as before.
> - **Gotcha to remember:** an auto-restarted container comes back with its ORIGINAL flags — fine today (both boxes run the production config), but if a box reboots mid-experiment, the *experiment* comes back, not production. Ollama (spark2 embed) already auto-starts via systemd, unchanged.
>
> ---
>
> **▶ PREV (2026-07-01 — CONFIG STILL LIVE, see banner above): APC PREFIX CACHING enabled on BOTH boxes' MTP-2 `Qwen3-Next-80B-A3B-Instruct-FP8`. Validated on spark2 (DFlash Coder-30B → this) first, then rolled onto spark1 production. Agent-loop/TTFT win.**
> The question: does vLLM's automatic prefix caching engage on the HYBRID Qwen3-Next (Gated DeltaNet/Mamba + sparse full-attn), where the GDN layers carry recurrent state, not radix-cacheable KV blocks — and can it coexist with MTP spec decode? Both answered **YES** (measured, not reasoned).
> - **APC engages via experimental Mamba 'align' mode.** vLLM 0.22.0 accepts `--enable-prefix-caching` on `Qwen3NextForCausalLM` and auto-logs *"Mamba cache mode is set to 'align' ... support for Mamba layers is experimental."* Ported the `PREFIX_CACHING=1` knob into `spin-up-vllm-qwen3-next-80b-mtp.sh` (it previously only existed on the plain `spin-up-vllm-qwen3-next-80b.sh`).
> - **Measured — APC (fixed ~7k-token prefix, resent):** APC-only build → cold TTFT **5.94s / 0 hits** → warm **0.57s / 6432 prefix-cache hits (~92% of the prefix)**, **~10× TTFT** (repeat-controlled: on a 2nd run even request A was warm, so it's caching, not first-inference warmup). APC **+ MTP-2** build → warm run **~5520 hits (~79%), TTFT ~6.7s→~1.0s**; the *hit count going 0→5520* is the unambiguous proof (the exact TTFT ratio is looser here — fresh container, not repeat-controlled). Hit rates are approximate (block-alignment varies).
> - **Measured — MTP is NOT degraded by APC (clean same-prompt A/B, 300-tok code-gen, temp 0):** run *before* the spark1 rollout, with spark1 as the MTP-only control — spark2 (APC+MTP) **76.1% per-token accept, 2.52 tok/backbone-step** vs spark1 (MTP-only) **77.1%, 2.54** — within noise, so **APC and MTP compose (prefill win + decode win, no trade)**. That result is what justified enabling APC on spark1 too. (Both sit below the doc's headline 95–99% because acceptance is prompt-dependent — this code prompt is just harder to draft; the A/B controls for it by using one prompt on both boxes.) Output coherent throughout (no #36872 gibberish). APC is a **prefill/TTFT win only** — the exact axis the read-heavy / agent-loop workload cares about, complementary to MTP's decode win.
> - **KV cost (APC + MTP @ 256K):** spark1 pool **1,002,438 tok / 3.82×**, spark2 pool **927,989 tok / 3.54×** (spark1 slightly roomier — no embed sidecar). Both down from the plain build's ~6.5× (drafter + Mamba align cache both consume memory) — fine at seqs 3, watch if concurrency rises. APC-only (no MTP) was **6.43×**, so most of the drop is the MTP drafter, not APC.
> - **Corrects the earlier "prefix caching left OFF / not pursued" note** (2026-06-20, below): on 0.22.0 it works on this hybrid. Served id unchanged (`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`), so no client repoint; `dgxlib/models.yaml` unchanged (APC changes prefill speed, not how the model is *called*).
> - **spark1 rollout verified 2026-07-01:** APC probe warm **5520 hits / TTFT 0.578s**, coherent (*"The ocean stretches endlessly beneath the horizon..."*). ~10 min restart; clients (MemPalace, llm_wiki, CampaignGenerator, opencode) reconnected on the unchanged id.
> **Revert a box to bare production MTP-2 80B (no APC):** `ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'` (spark1) / `ssh spark2 '…'` (spark2).
> **Re-run this (APC + MTP), either box:** `PREFIX_CACHING=1 MAX_SEQS=3 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh`.
> **Re-run the DFlash experiment (spark2):** `ssh spark2 'bash ~/spin-up-vllm-qwen3-coder-30b-a3b-dflash.sh'` (weights cached, ~9 min).
>
> ---
>
> **▶ PREV (2026-06-30): spark2 swapped to a DFlash speculative-decoding EXPERIMENT — `Qwen3-Coder-30B-A3B-Instruct` + `z-lab/Qwen3-Coder-30B-A3B-DFlash`. spark1 untouched (still production MTP-2 80B).**
> DFlash (UC San Diego, Feb 2026, `z-lab/dflash`) drafts a whole BLOCK of up to N future tokens in one non-causal diffusion forward pass, verified by the target in a single batched pass — a higher per-step ceiling than MTP's 1-2 token autoregressive draft. Tried via `spin-up-vllm-qwen3-coder-30b-a3b-dflash.sh` / `spin-up-vllm-qwen3-coder-next-dflash.sh` (both new in this repo).
> - **CONFIRMED: `vllm/vllm-openai:v0.22.0-aarch64` (already cached on both boxes) ships DFlash natively** — no git clone, no pip install, no image rebuild, no vLLM downgrade. Select via `--speculative-config '{"method":"dflash","model":"<drafter-id>","num_speculative_tokens":N}'`.
> - **HARD CONSTRAINT — fp8 KV cache is incompatible with DFlash on this vLLM build.** The drafter needs a non-causal attention backend; FlashAttention is the only backend that supports non-causal at all, but it rejects fp8 KV specifically. Every other backend (FlashInfer, Triton, FlexAttention, TurboQuant) fails the non-causal check outright regardless of KV dtype. Net: engine init hard-fails ("No valid attention backend found for cuda") unless `--kv-cache-dtype auto` (bf16). Must also set `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (env var — not a `vllm serve` CLI flag on this build).
> - **HARD CONSTRAINT — DFlash/EAGLE3 does NOT work on the Qwen3-Next hybrid family (Gated DeltaNet).** Tried pairing `Qwen3-Coder-Next-FP8` with `z-lab/Qwen3-Coder-Next-DFlash`: crashed with `RuntimeError: Model does not support EAGLE3 interface`. Traced to source: `Qwen3NextForCausalLM`'s actual base classes are `nn.Module, HasInnerState, SupportsLoRA, SupportsPP, QwenNextMixtureOfExperts, IsHybrid` — no `SupportsEagle`/`SupportsEagle3` mixin at all, on either plain or coder variant. By the same code trace, **Qwen3.5-122B-A10B (served via the multimodal `Qwen3_5MoeForConditionalGeneration` wrapper) would almost certainly hit the identical failure** despite its inner text backbone (`Qwen3_5ForCausalLMBase`) technically implementing `SupportsEagle3` and despite an official `z-lab/Qwen3.5-122B-A10B-DFlash` checkpoint existing — vLLM's `supports_eagle3()` check is a plain `isinstance()` on the top-level served model object with no multimodal unwrapping, and the wrapper class doesn't inherit/delegate the required aux-hidden-state methods. **Not empirically tested** (would require reviving the torn-down 2-box RDMA cluster) — reasoned from source, not measured.
>   - **Conclusion: DFlash only works on this box's plain dense/MoE `Qwen3ForCausalLM`/`Qwen3MoeForCausalLM` family.** Every hybrid Qwen3-Next-derived model actually in production or recently deployed here (Qwen3-Next-80B, Qwen3-Coder-Next, and by inference Qwen3.5-122B) is blocked. This vLLM version's native MTP head (`qwen3_next_mtp`, already deployed on both boxes) is the correct/only spec-decode path for that family.
> - **Measured on `Qwen3-Coder-30B-A3B-Instruct` (works — plain `Qwen3MoeForCausalLM`, 48 full-attention layers, no hybrid layers):** ~28.4 tok/s on a 300-token code-gen response (10.6s wall). Real draft-acceptance samples from logs: mean accept length 3.5–6.1 tokens per verify step (max 15), accept rate 16–34%, per-position acceptance decaying from ~89% at position 1 to ~12–24% by position 15 — DFlash's mechanism is genuinely working. Headline tok/s still trails spark1's FP8 MTP-2 80B (~49 tok/s) because this target is **BF16 only** (no FP8 checkpoint exists) — decode is bandwidth-bound, so BF16 pays ~2× the bytes/token vs FP8, a confound separate from DFlash itself. A same-precision non-spec baseline run would be needed to isolate DFlash's true multiplier.
> - **KV pool measured tight:** 322,928 tokens total, only **1.23× concurrency at full 262,144 (256K)** — this is a full-attention MoE (every one of 48 layers carries KV), unlike the hybrid 80B's ~9.6× headroom. Don't push `MAX_SEQS` up without checking `docker logs vllm-chat | grep "GPU KV cache size"` first.
> - **⚠️ CLIENT IMPACT — spark2 IS client-facing** (correction 2026-06-30: this doc previously assumed spark2 was sandbox-only — wrong). spark2's served model id changed from `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` to `Qwen/Qwen3-Coder-30B-A3B-Instruct` for the duration of this experiment — any client pointed at spark2:8001 expecting the production 80B id will 404 or get a different model's behavior until reverted. Which client(s) specifically point at spark2 is being tracked outside this doc (Kostadis managing directly) — **treat spark2 as production-adjacent, not a free sandbox, until that's resolved.** `dgxlib/models.yaml` NOT updated for this experiment id — verify whether it needs to be before leaving this running long-term.
> **Revert spark2 to production MTP-2 Qwen3-Next-80B:** `ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`.
> **Re-run this experiment:** `ssh spark2 'bash ~/spin-up-vllm-qwen3-coder-30b-a3b-dflash.sh'` (weights now cached, ~9 min instead of ~25 min fresh pull).
>
> ---
>
> **▶ PREV (2026-06-29): BOTH boxes swapped THROUGHPUT (plain, seqs 16) → LATENCY (MTP-2 spec decode, seqs 3) via `spin-up-vllm-qwen3-next-80b-mtp.sh`.**
> Rationale: the +47% throughput A/B (2026-06-23) was for the concurrent batch document-conversion job; the current use is **low-concurrency background serving** (run it as a background task, do other things on the box), where MTP single-stream speedup wins. Both chat slots now: `Qwen3-Next-80B-A3B-Instruct-FP8`, **`--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`**, **`--max-num-seqs 3`**, **util 0.80**, 256K, fp8 KV, `--max-num-batched-tokens 40960`, hermes, vLLM 0.22.0, image `vllm/vllm-openai:v0.22.0-aarch64`.
> - **Measured:** draft acceptance **~95–99%** (mean accept length ~2.95 of 3), ~49 tok/s single-stream decode. Host headroom: spark1 **~14 GB**, spark2 **~8–10 GB** (fresh boot).
> - **spark2 raised 0.70 → 0.80:** MTP's KV floor needs it — at 0.70 the 256K KV check fails to allocate even one request (`estimated maximum model length 15456`). 0.80 gives ~12 GiB KV (~3.5× full-256K seqs).
> - **spark2 rebooted 2026-06-29** (host-RAM wedge: embed sidecar + MTP-0.80 crept to ~6 GB host over 2 days; on reboot only `vllm-embed` auto-restarted, `vllm-chat` relaunched manually). Restart: `ssh spark2 'MAX_SEQS=3 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`.
> - **spark2 kernel panic 2026-06-30** (hard crash, no OOM in kern.log — silent between 15:00 and 18:03; root cause unknown, likely GPU/memory pressure under unified memory). Box manually rebooted at 18:03.
> - **vllm-embed removed 2026-06-30:** `vllm-embed` (port 8000, util 0.05, always-on) replaced by Ollama `qwen3-embedding:0.6b` on port 11434 (lazy-load, unloads after 5 min idle). This frees the ~6 GB GPU reservation permanently and gives spark2 headroom matching spark1 (~14 GB host available with just vllm-chat). Embed endpoint changed: `spark2:8000` → `spark2:11434`, model id `Qwen/Qwen3-Embedding-0.6B` → `qwen3-embedding:0.6b`.
> - **`dgxlib/models.yaml` unchanged:** served id is identical and MTP changes how fast it decodes, not how it's *called* (thinking default + timeouts unchanged).
>
> **Revert both boxes to THROUGHPUT (plain, seqs 16):** `ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b.sh'` and `ssh spark2 'GPU_UTIL=0.70 bash ~/spin-up-vllm-qwen3-next-80b.sh'`.

> **▶ PREV (2026-06-25): spark1 `vllm-chat` REVERTED `inclusionAI/Ling-lite` → `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`. Both boxes are back on the production 80B; the Ling experiment is torn down.**
> The Ling-lite throughput experiment (banner below) concluded; spark1 was
> restored to the production model via `ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b.sh'`.
> - **Verified:** `/v1/models` = `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`,
>   `max_model_len 262144`; smoke returns "OK"; vLLM 0.22.0, util 0.80, seqs 16,
>   fp8 KV, hermes tools — the documented spark1 throughput config. Warm restart
>   (~10 min: cached weights, shard load + torch.compile).
> - **Clients pinned to the 80B id work against spark1 again** (MemPalace,
>   llm_wiki, CampaignGenerator, opencode — no repoint needed).
> - **Kept for re-runs:** `spin-up-vllm-ling-lite.sh` (the experiment script,
>   32K caveat baked in), the `inclusionAI/Ling-lite` `dgxlib/models.yaml` entry
>   (inert while not served), and the `ling-ab-slice/` test slice + A/B commands.
> **Re-run the Ling experiment:** `ssh spark 'bash ~/spin-up-vllm-ling-lite.sh'`.
>
> ---
>
> **▶ PREV (2026-06-25, superseded by the revert above): spark1 `vllm-chat` SWAPPED Qwen3-Next-80B → `inclusionAI/Ling-lite` — a fewer-active-MoE EXPERIMENT for the pdf-translators batch job.**
> Ling-lite is a 16.8B-total / **~2.75B-active** full-attention MoE (vs the 80B's
> ~3B active), serving on spark1:8001 via `spin-up-vllm-ling-lite.sh`. **BF16
> weights** (~32 GB; no FP8 checkpoint exists for this 1.0 build), **fp8 KV**,
> **util 0.75**, **seqs 16**, **NO tool-call parser** (plain chat — bailing_moe
> has no vLLM tool parser, and the render job doesn't need one), image
> `vllm/vllm-openai:v0.22.0-aarch64`.
> - **Served at 32K context (`--max-model-len 32768`), NOT 128K.** The model
>   card advertises 128K but this default-branch checkpoint's config caps
>   `max_position_embeddings=32768`; the 128K needs YaRN rope-scaling that isn't
>   enabled. Forcing it (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`) on a RoPE model
>   yields NaN/OOB past 32K — don't.
> - **Verified:** arch `BailingMoeForCausalLM` resolved on the aarch64 image;
>   `Application startup complete`; coherent generation (*"The ocean is a vast,
>   interconnected body of saltwater that covers more than 70% of the Earth's
>   surface."*), `reasoning:null`, **no `<think>` leak** (Ling is the non-thinking
>   line; Ring is its reasoning sibling). KV pool **11.37M tokens → 347× @ 32K**
>   (tiny weights leave a huge pool; KV is NOT the bound here). Host RAM healthy
>   at **24 GB available**.
> - **Why:** decode+prefill scale with *active* params; 2.75B < 3B, so Ling is a
>   touch faster per token. The bet is throughput; the risk is quality
>   (~Qwen2.5-7B-Instruct tier, well below the 80B). **Safe to try only because
>   the pdf-translator has a HARD validator gate** (`validate_adventure.py`) —
>   judge it on validated-docs/hour, not tok/s.
> - **⚠️ CLIENT IMPACT:** spark1's served id changed, so clients pinned to
>   `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` (MemPalace `llm_model`, opencode `dgx`
>   default, llm_wiki provider, CampaignGenerator `DGX_DEFAULT_MODEL` — see §6)
>   now 404 against spark1 until repointed or pointed at spark2 (still the 80B).
>   Tool-calling clients (opencode) won't work against Ling (no parser).
> - **⚠️ BATCH-JOB IMPACT:** `batch_convert.py` defaults assume spark1 is the
>   large-context box (`--spark1-ctx 262144`, `--prompt-cap 40000 +
>   --output-cap 80000 = 120K`). Ling at 32K can't hold that — you MUST run
>   Ling with caps summing under 32768, or point the job at spark2 for big docs.
>   `dgxlib/models.yaml` updated (new `inclusionAI/Ling-lite` entry, 32K note).
> **Revert spark1 to the production 80B:** `ssh spark 'bash ~/spin-up-vllm-qwen3-next-80b.sh'`.
>
> ---
>
> **▶ PREV (2026-06-24): spark2 `vllm-chat` dropped to `--gpu-memory-utilization 0.70` (was 0.80) for host-RAM headroom. Config otherwise unchanged (throughput: plain, seqs 16, 256K, fp8 KV, hermes, vLLM 0.22.0).**
> Context: after spark2 rebooted (only the `--restart unless-stopped` embed
> sidecar came back; the chat slot doesn't auto-restart), `vllm-chat` was
> brought back at 0.80 and idle host RAM was only **9 GB available / 1 GB free**
> — at the documented "drop to seqs 8 or util 0.78" edge, because spark2 also
> carries the `vllm-embed` sidecar. Re-run at **0.70** via
> `MAX_LEN=262144 GPU_UTIL=0.70 MAX_SEQS=16 bash ~/spin-up-vllm-qwen3-next-80b.sh`.
> - **Measured @ 0.70:** `speculative_config=None`, seqs 16, fp8 KV,
>   max_model_len 262144. **GPU KV pool 723,010 tok → 2.76× @ 256K** (was 1.84M
>   / 7.0× at 0.80). Host RAM recovered **9 GB → 23 GB available** (~13 GB
>   reservation freed). Verified serving the batch doc-conversion job at
>   16 running reqs, ~187 tok/s aggregate.
> - **Tradeoff:** 0.70 trades KV concurrency (7.0× → 2.76× at full 256K) for
>   host headroom. Fine for the batch job (chapters are far short of 256K, and
>   bandwidth — not KV — bounds decode here); if many concurrent requests each
>   want huge contexts, vLLM will queue past the 2.76× full-length ceiling.
> - **spark1 stays at 0.80** (no embed sidecar → more host headroom; the two
>   boxes' chat util now differs by design).
> - **No client config change**; `dgxlib/models.yaml` unchanged.
> **Revert spark2 to 0.80:** `MAX_LEN=262144 GPU_UTIL=0.80 MAX_SEQS=16 bash ~/spin-up-vllm-qwen3-next-80b.sh`.
>
> ---
>
> **▶ PREV (2026-06-23, latest): BOTH boxes on the THROUGHPUT config (plain, NO spec decode, seqs 16). A/B concluded — throughput config won (+47%); spark2 moved to match spark1 to serve a batch document-conversion job.**
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

**⚠⚠ SUPERSEDED 2026-09-10 by the top LIVE banner: spark1 now serves `qwen3.8-flash-next` SINGLE-BOX and spark2 serves `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` (MTP-2 + APC, seqs 8). The cross-box DSpark pair described in the 2026-08-04 note below is GONE. Of the two-single-box Qwen3-Next world described from here down, the spark2 half is LIVE AGAIN and accurate; the spark1 half is a revert target only.**

**⚠ Superseded 2026-08-04 (itself now superseded — see above): BOTH boxes were one cross-box `deepseek-ai/DeepSeek-V4-Flash-0731` endpoint (spark1 head :8001, spark2 headless worker with NO API; upgraded 2026-08-04 from the `DeepSeek-V4-Flash-DSpark` preview that first replaced this config on 2026-07-30).**

**⚠ Superseded by the LIVE banner at the top of this doc (2026-07-04, later): spark2 is now the MTP-8 BATCH box (MTP-2 spec decode + APC, seqs 8), NOT the plain-16 throughput or DFlash configs described below. spark1 is unaffected — still MTP-2 + APC latency/production (seqs 3).**

Snapshot of the previous steady state (**restore target** as of 2026-07-30) on
**both** DGX Sparks as of
2026-07-04, later (single-box Qwen3-Next-80B-A3B-Instruct-FP8 on each, both @ 256K
on vLLM 0.22.0, both at **util 0.80**, both fp8 KV, chunked prefill 40K, hermes tools). **The two boxes diverge by design:**
**spark1 = LATENCY config** (MTP-2 spec decode `qwen3_next_mtp`,
num_speculative_tokens=2, seqs 3, + APC prefix caching); **spark2 = MTP-8 BATCH
config** (MTP-2 spec decode + APC, seqs 8 — measured fastest on the Cognee
batch load). spark2 embed sidecar replaced: `vllm-embed` (port 8000, always-on)
→ Ollama `qwen3-embedding:0.6b` (port 11434, lazy-load). spark2 host headroom
now matches spark1 (~13-14 GB with just vllm-chat). The 2026-06-25 `inclusionAI/Ling-lite` experiment
on spark1 was reverted — see top LIVE banner.
Use this as a "rebuild from scratch" reference if either box wipes, or as
inventory when debugging. **Verified live 2026-07-12: both boxes match this
snapshot exactly (model id, image, flags, restart policy) — no drift.**

> **Two-box layout — CORRECTION 2026-06-30: spark2 IS client-facing, not sandbox-only.** This section previously claimed spark2 was purely experimental (opencode sandboxing / side-by-side comparison) — that's wrong. `spark1` backs production LLM clients (MemPalace, llm_wiki, CampaignGenerator, opencode) and runs `vllm-embed` (port 8000), `vllm-chat` (port 8001), and Ollama (port 11434, mostly idle). spark2 also has at least one real client pointed at it — which one(s) is being tracked outside this doc — so treat model swaps on spark2 (like the DFlash experiment in the top LIVE banner) as having real client impact, not as a free sandbox. spark2 still doubles as the box for models incompatible with spark1's clients (e.g. reasoning models that emit `<think>` traces llm_wiki can't strip).

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

> **⚠ 2026-10-10 — CURRENT STATE (supersedes both tables below). spark2 runs Decision-2.0-Nox-4B on 8005 (NPC prototype); its 8001 `qwen38-flash` is STOPPED; the 8002 `clef` row is STOPPED; Kai 8004 stopped.**
>
> | box | port | service | what |
> |---|---:|---|---|
> | spark1 | 8001 | **`qwen38-flash`** (docker) | **`qwen3.8-flash-next`** (Qwen3.8-Flash-Next; RadixArk NVFP4 routed experts + blockwise-fp8 side layers = `MODE=hybrid`), **single-box TP=1**, **262K** (`--max-model-len 262144`), util **0.80**, seqs 8, **bf16 KV** (`kv_cache_dtype=auto`), **MTP-2** (`{"method":"mtp","num_speculative_tokens":2}`), **APC on** (correct only because this image carries the Mamba block-size fix — see LIVE banner), deterministic QSA top-k (`VLLM_QSA_DET_TOPK=1`), reduced draft vocab (65,536), `MADV_RANDOM` on the mmapped table, chunked prefill 8192, PIECEWISE CUDA graphs, **`--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`**, image `qwen38-flash-dgx` (LOCAL build from `~/qwen3.8-Flash-DGX`, vLLM `0.1.dev20073+g8e685d198`). The 47.7 GiB PLE n-gram table is **mmapped from NVMe**, not resident. Weights on card **76.75 GiB**. KV pool **553,254 tok → 2.11× @ 262K**. Via `./spin-up-vllm-qwen38-flash-next.sh`. **`--restart unless-stopped` — DOES survive a reboot.** |
> | spark2 | 8002 | **`clef`** (docker) | **Clef + Clef-flash decision models** (LIVE 2026-10-02) — Jev/SystemOne API `POST /v1/systemone`, `GET /v1/models`, `GET /health`; image `clef-server` (transformers 5.10.2, torch 2.11.0+cu130, fla Triton kernels), BF16, 68.9 GiB allocated, `--restart unless-stopped`. Via `STOP_CHAT=1 ./spin-up-clef.sh`. |
> | spark2 | 8001 | — | **STOPPED 2026-10-10** (`qwen38-flash`, identical to spark1; container kept). |
> | spark2 | 8005 | **`decision2-nox-4b`** (docker) | **Decision-2.0-Nox-4B** (LIVE again 2026-10-10) — Jev/SystemOne API, ~19 GB RSS; image `decision2-runtime`. NPC prototype backend. |
> | spark1 | 11434 | Ollama (systemd) | `qwen3-embedding:0.6b` — **verified untouched** by the 2026-09-10 swap (live 1024-dim smoke), still the MemPalace embedding path |
> | spark2 | 11434 | Ollama (systemd) | `qwen3-embedding:0.6b`, lazy-load — untouched |
> | both | 8000 | — | not running on either box (confirmed 2026-09-10) |
>
> **Both boxes are independent endpoints again**, so `refresh-current-setup.sh`
> should be run WITHOUT `-C` (cluster mode was only for the cross-box DSpark head).
> The two tables below are older Qwen3-Next-era snapshots.

### spark1 (192.168.1.147) — REVERT TARGET, not current

| port | service | purpose |
|---:|---|---|
| 11434 | Ollama (systemd) | LLM serving + **currently the live embeddings path** (`nomic-embed-text`) while vllm-embed is down |
| 8000 | vllm-embed (docker) | Embeddings — `nomic-embed-text-v1.5` — **DOWN** (stopped back during the cross-box experiment, still not restored; embeddings on Ollama 11434. Could now be restored — box is single-box again — but left on Ollama for continuity) |
| 8001 | vllm-chat (docker) | Chat completions — **`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`** single-box TP=1, **256K** (`--max-model-len 262144`), **MTP-2 spec decode** (`--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`) **+ APC `--enable-prefix-caching`** (hybrid → experimental Mamba 'align' mode), chunked prefill **40K** (`--max-num-batched-tokens 40960`), hermes tools, no reasoning parser, **gpu-util 0.80**, **seqs 3**, fp8 KV, image **`vllm/vllm-openai:v0.22.0-aarch64`**. KV pool **1,002,438 tok → 3.82× @ 256K** (APC+MTP). Via `PREFIX_CACHING=1 MAX_SEQS=3 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh`. (2026-06-29: THROUGHPUT plain/seqs-16 → LATENCY MTP-2/seqs-3; 2026-07-01: + APC prefix caching.) **Auto-restarts on reboot (`--restart unless-stopped`, 2026-07-02).** Independent endpoint. |

### spark2 (192.168.1.121) — REVERT TARGET, not current

| port | service | purpose |
|---:|---|---|
| 11434 | Ollama (systemd) | Embeddings — **`qwen3-embedding:0.6b`** — 1024-dim, instruction-aware, lazy-load (unloads after 5 min idle). Replaced `vllm-embed` on 2026-06-30 to free the always-on ~6 GB GPU reservation. Endpoint: `http://192.168.1.121:11434/v1/embeddings`. |
| 8001 | vllm-chat (docker) | Chat completions — **MTP-8 BATCH box (2026-07-04, later)**: **`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`**, **MTP-2 spec decode** (`--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`) **+ APC `--enable-prefix-caching`**, **256K** (`--max-model-len 262144`), **gpu-util 0.80**, **seqs 8**, **fp8 KV**, chunked prefill **40K** (`--max-num-batched-tokens 40960`), hermes tools, no reasoning parser, image **`vllm/vllm-openai:v0.22.0-aarch64`**. KV pool measured **933,232 tok → 3.56× @ 256K**. Measured fastest on the Cognee batch load (~138 t/s vs plain-16's ~125). Via `ssh spark2 'PREFIX_CACHING=1 MAX_SEQS=8 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`. Revert to plain throughput (higher KV, 16 slots): `ssh spark2 'PREFIX_CACHING=1 bash ~/spin-up-vllm-qwen3-next-80b.sh'`. Revert to latency (MTP-2 + APC, seqs 3): `ssh spark2 'PREFIX_CACHING=1 MAX_SEQS=3 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`. **Auto-restarts on reboot (`--restart unless-stopped`).** |

(Port 8000 / `vllm-embed` removed 2026-06-30 — replaced by Ollama on 11434. Briefly ran SGLang `sglang-chat` on 2026-06-11; reverted to vLLM same day — see the note under the LIVE banner. **2026-06-30: swapped to a DFlash experiment; 2026-07-01: swapped again to MTP-2 80B + APC prefix caching — see top LIVE banner for full findings.**)

## VRAM budget (steady state)

> **⚠ 2026-09-10 — CURRENT budget. The two boxes now differ, so read the right column.**
>
> **✅ 2026-10-08: the spark2 column is CURRENT again** (qwen38-flash back on spark2; measured KV pools this boot: spark1 602,496, spark2 590,910; host available ~16 GB on each).
>
> **(superseded) ⚠ 2026-10-02: the spark2 column is HISTORICAL** — spark2 now runs Clef + Clef-flash: **68.9 GiB allocated by torch (no reservation cap), no off-card table, host available ~41-45 GB.** The spark1 column is current.
>
> | | spark1 — `qwen3.8-flash-next` | spark2 — `qwen3.8-flash-next` |
> |---|---|---|
> | Reserved cap | ~97 GB (**0.80** × ~121 GB usable) | ~97 GB (**0.80**) |
> | Weights on card | **76.75 GiB** (NVFP4 experts + fp8 side layers) | **76.75 GiB** (exact match) |
> | **Off-card** | **47.7 GiB PLE n-gram table mmapped from NVMe** — served through the page cache, NOT in the pool | **same — 47.7 GiB mmapped** |
> | KV | bf16, pool **553,254 tok → 2.11× @ 262K** | bf16, pool **576,427 tok → 2.20× @ 262K** |
> | Host available, idle | **~14-17 GB** | **~16 GB** |
>
> **⚠ On BOTH boxes the host page cache is NOT spare capacity — it is the storage
> tier for a 47.7 GiB lookup table, and prefill speed is a direct function of
> how much of it stays cached.** That is why `GPU_MEM` must stay at 0.80 and
> not go up: the recipe measured 0.85 drifting into swap after a day and 0.875
> being OOM-killed on a 300K prefill, which is the same unified-memory story as
> `feedback_gpu_util_080_default` / `gpu-reservation-and-kv-tradeoffs.md`
> arriving from a different direction. Measured effect of a cold cache:
> prefill drops from ~2,450 tok/s warm to ~1,137 tok/s.
>
> Ollama's big pulled models (`llama3.3:70b` 42.5 GB, `qwen2.5:32b` 19.9 GB)
> would be fatal if loaded on top of either box — the user confirms nothing
> drives them, but that is a behavioural guarantee, not an enforced one. The
> lazy-loaded `qwen3-embedding:0.6b` (639 MB) is fine and verified coexisting.
> The recipe warns "one big model at a time — an 8B embedding model next to it
> already starves the KV cache"; ours is an order of magnitude smaller.
>
> The two sections below are older Qwen3-Next-era snapshots (spark2's is now a
> REVERT TARGET in the literal sense — that is the config `spin-up-vllm-qwen3-next-80b-mtp.sh`
> restores).

### spark1 — REVERT TARGET, not current

| service | reserved cap | actual model size | notes |
|---|---:|---:|---|
| vllm-embed | — | — | **DOWN** — embeddings on spark2:8000 (`Qwen/Qwen3-Embedding-0.6B`); spark1 left without a local embedder for continuity |
| vllm-chat (single-box) | ~102 GB (**0.80** × ~128 GB) | ~76.5 GiB FP8 weights (incl. MTP draft head) + fp8 KV + activations @ 256K | Qwen3-Next-80B-A3B, **MTP-2 spec decode (`qwen3_next_mtp`), seqs 3** (LATENCY) **+ APC prefix caching** (Mamba 'align' mode), 256K, chunked prefill 40K (hybrid: only the periodic full-attn layers carry KV). KV pool **1,002,438 tok → 3.82× full-256K seqs** (APC+MTP). **No embed sidecar on spark1** — **~14 GB host available** with the MTP draft loaded. Draft acceptance ~96% (unaffected by APC). (2026-06-29: THROUGHPUT plain/seqs-16 → MTP-2/seqs-3; 2026-07-01: + APC.) |
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

### spark2 — REVERT TARGET, not current

**As of 2026-07-04 (later) spark2 runs MTP-2 spec decode + APC at seqs 8 (the MTP-8 batch config) — measured fastest on the Cognee batch load. See top LIVE banner.**

| service | reserved cap | actual model size | notes |
|---|---:|---:|---|
| vllm-chat (single-box) | ~102 GB (**0.80** × ~128 GB) | ~76.5 GiB FP8 weights (incl. MTP draft head) + fp8 KV + activations @ 256K | Qwen3-Next-80B-A3B, **MTP-2 spec decode (`qwen3_next_mtp`), seqs 8** (MTP-8 BATCH) **+ APC prefix caching** (Mamba 'align' mode), 256K, chunked prefill 40K. KV pool **933,232 tok → 3.56× full-256K seqs** (tighter than the plain-16 build's 6.11× — drafter + Mamba align cache eat KV). **util 0.80.** Host headroom **~13 GB** (just vllm-chat; embed is lazy-load Ollama). **This is the batch box (2026-07-04, later); spark1 stays MTP-2 + APC latency (seqs 3).** |
| Ollama (idle) | ~0 | unloads after 5 min | `qwen3-embedding:0.6b` (639 MB GGUF). Lazy-load; when active ~640 MB GPU. Replaced the always-on `vllm-embed` (was ~6 GB reservation) on 2026-06-30. |

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

> **▶ CURRENT (2026-09-10, later): BOTH boxes serve `qwen3.8-flash-next`, single-box TP=1, same image, same flags.**
> spark1: `./spin-up-vllm-qwen38-flash-next.sh` — spark2: `HOST=spark2 ./spin-up-vllm-qwen38-flash-next.sh`.
> Container `qwen38-flash` on each, port 8001, `MODE=hybrid`, `CTX=262144`,
> `GPU_MEM=0.80`, `SEQS=8`, `MTP=2`, `PREFIX_CACHE=1`, `DET_TOPK=1`,
> `DRAFT_VOCAB=1`, bf16 KV, `--enable-auto-tool-choice --tool-call-parser
> qwen3_coder --reasoning-parser qwen3`, `--restart unless-stopped`, image
> `qwen38-flash-dgx` (vLLM `0.1.dev20073+g8e685d198`). Full measurements and the
> per-box verification live in the LIVE banners at the top of this doc; the
> revert paths are there too. **The boxes are interchangeable — there is no
> latency/throughput split any more, and no second model to A/B against.**
>
> **▶ PREV (2026-07-03, Qwen3-Next era): boxes split — spark1 = MTP-2 LATENCY, spark2 = PLAIN THROUGHPUT.**
> **spark1 (latency)** — `MAX_SEQS=3 GPU_UTIL=0.80 SPEC_TOKENS=2 PREFIX_CACHING=1 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh`
> → `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`, `--max-model-len 262144`,
> `--max-num-seqs 3`, `--gpu-memory-utilization 0.80`, `--kv-cache-dtype fp8`,
> `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`,
> `--enable-prefix-caching`, `--max-num-batched-tokens 40960`,
> `--enable-auto-tool-choice --tool-call-parser hermes`,
> `--restart unless-stopped` (added 2026-07-02 — the chat slot now survives
> reboots/powercycles; before this, a reboot left it `Exited` until someone
> noticed), image `vllm/vllm-openai:v0.22.0-aarch64`. **Why these flags:** seqs=3 is the
> low-concurrency background-serving point — it keeps MTP active
> (below the auto-disable-by-batch threshold) and leaves box headroom; MTP-2
> drafts 2 tokens/step off the model's native MTP head and verifies in one
> target pass. **Measured:** draft acceptance ~95–99% (mean accept length
> ~2.95/3), ~49 tok/s single-stream decode.
> **spark2 (throughput / batch, 2026-07-03)** — `ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b.sh'`
> → same model, **plain** (`speculative_config=None`, `enable_prefix_caching=False`),
> `--max-num-seqs 16`, util 0.80, 256K, fp8 KV, hermes, `--restart unless-stopped`.
> **Measured:** KV pool 1,601,403 tok → 6.11× @ 256K. Point batch / high-concurrency
> jobs here; single-stream reads slower (~30 tok/s, no MTP) but aggregate throughput
> is higher under load (won a +47% A/B). Revert spark2 to latency:
> `ssh spark2 'PREFIX_CACHING=1 MAX_SEQS=3 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`.
> The banners below are PRIOR slot occupants, kept as runbooks.

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

## 4. spark2 vllm-chat slot (Docker container, port 8001)

> **✅ 2026-10-08: this slot is LIVE again** — `qwen38-flash` restarted with `docker start`; the CURRENT block below applies. The Clef/Decision 2.0 note that follows is historical.

> **(superseded) ⚠ 2026-10-02: this slot is EMPTY on spark2.** The box runs the Clef decision
> models on **:8002** instead (container `clef`, `spin-up-clef.sh`, `clef/`). Rebuild:
> `scp clef/Dockerfile clef/server.py spark2:~/clef/ && ssh spark2 'cd ~/clef && docker build -t clef-server .'`,
> `hf download Cloudflare/clef-flash` + `Cloudflare/clef` into `~/.cache/huggingface` (74 GB),
> then `STOP_CHAT=1 ./spin-up-clef.sh`. Everything below describes the revert target.

> **▶ CURRENT (2026-09-10, later): container `qwen38-flash`, serving
> `qwen3.8-flash-next` — an INDEPENDENT endpoint, identical to spark1's §3.**
> This is not a Ray worker and not half of a cross-box pair: it is a second,
> complete, single-box TP=1 server with the same image, the same flags and a
> byte-identical checkpoint. **Everything in §3 applies verbatim** — bring-up
> (`HOST=spark2 ./spin-up-vllm-qwen38-flash-next.sh`), flags, the prefix-caching
> block-size landmine, the `reasoning`-vs-`reasoning_content` client trap, the
> PLE-table page-cache dependency. Per-box measurements are in the LIVE banner at
> the top of this doc. Deliberately not duplicated here — one description, two
> boxes.
>
> **How spark2 was stocked** (it had no recipe, image or checkpoint): copied from
> spark1 over the 10.100.16.x cable rather than re-downloaded — see §8.
>
> **REVERT to Qwen3-Next-80B** (77 GB checkpoint still on the box, so a load, not
> a download): `ssh spark2 'docker rm -f qwen38-flash'; ssh spark2 'PREFIX_CACHING=1 MAX_SEQS=8 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`.
> The banners and run commands below are PRIOR occupants of this slot, kept as runbooks.

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
| laptop, desktop, etc. | spark1 chat (`qwen38-flash`) | `http://192.168.1.147:8001/v1/chat/completions` — model id **`qwen3.8-flash-next`** (2026-09-10) |
| laptop, desktop, etc. | spark2 Ollama embeddings | `http://192.168.1.121:11434/v1/embeddings` — `qwen3-embedding:0.6b`, lazy-load (was `spark2:8000/vllm-embed` until 2026-06-30) |
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

> **▶ LIVE (2026-09-10): spark1:8001 serves `qwen3.8-flash-next` (lowercase). CLIENT REPOINT STATUS — audited per client on the WORKSTATION, not assumed:**
>
> | client | config location | on this box? | status |
> |---|---|---|---|
> | **openclaw** | `~/.openclaw/openclaw.json` | **yes** | ✅ **FIXED 2026-09-10 and verified end-to-end** (chat + tool call). Provider renamed `deepseek-v4-local` → **`spark1-local`** (the old name said "deepseek" while serving Qwen; a box-named provider survives the next swap as a one-line id change). `openclaw models list` shows `spark1-local/qwen3.8-flash-next` as default, 262k. Backup: `~/.openclaw/openclaw.json.bak-20260910`. |
> | **CampaignGenerator** | `config/wiring.yaml` (mneme-rendered) | **no rendered file** | ⚠ **Not "broken" — UNCONFIGURED.** There is no `~/.config/hypostasis/hypostasis.yaml` and no rendered `config/wiring.yaml` on this box, so `wiring_get("dgx_model")` returns **`None`** (verified live). It never had a stale id to break. It requires `--model` or `DGX_MODEL` per invocation: `DGX_MODEL=qwen3.8-flash-next python session_doc.py ... --dgx-endpoint http://192.168.1.147:8001/v1`. The durable fix belongs in mneme (`machines.dgx.default_model` in `hypostasis.yaml`, then `hypostasis apply`) — **a different repo, deliberately not edited here.** Note `~/src/mneme/hypostasis.example.yaml` still carries the old Qwen3-Next id and a TEST-NET placeholder endpoint. |
> | **MemPalace** | `~/.mempalace/config.json` | **no** | ⚠ Not present on this box — `~/.mempalace/` holds only `hook_state/`. The doc has always said this config lives **on the laptop**. Nothing to repoint here. |
> | **opencode** | `~/.config/opencode/opencode.json` | **no** | ⚠ Not installed on this box (`~/.config/opencode/` absent). Nothing to repoint here. |
> | **llm_wiki** | `%APPDATA%\com.llmwiki.app` | **no** | ⚠ Separate **Windows** desktop, not reachable from this Linux box. Set Endpoint `http://192.168.1.147:8001/v1`, Model `qwen3.8-flash-next` in App Settings. |
>
> So the long-standing "five broken clients" TODO was partly an artefact of the
> doc tracking clients across machines: **one existed here and is now fixed**;
> one was never configured; three live on other hosts.
>
> **❌ THAT FALLBACK IS GONE (2026-09-10, later).** The line that used to sit
> here said spark2:8001 still served `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`, so
> any client pinned to the old id "works today untouched". **That is no longer
> true** — spark2 now serves `qwen3.8-flash-next` too, and a client still
> sending the Qwen3-Next id to `http://192.168.1.121:8001/v1` will **400**. The
> escape hatch is now a revert, not a second endpoint:
> `ssh spark2 'docker rm -f qwen38-flash'; ssh spark2 'PREFIX_CACHING=1 MAX_SEQS=8 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`
> (~15 min; the 77 GB checkpoint is still on the box).
>
> **✅ What this buys instead:** the repoint work above is now worth doing ONCE
> and covers both boxes — same id, same parsers, same thinking switch, on
> `192.168.1.147:8001` **and** `192.168.1.121:8001`. Any client already fixed for
> spark1 (openclaw) works against spark2 by changing only the host. The two are
> interchangeable, so spark2 is also where to send overflow when spark1 is
> queueing (it was at 8 running / 31 deferred during this swap's verification).
>
> **Unaffected either way:** anything using Ollama `:11434` for embeddings —
> verified live during this swap.
>
> **⚠ When repointing any client, also handle thinking.** Traces are ON by
> default and are billed to `completion_tokens` (86% of them on a short reply),
> and both opencode and openclaw read `reasoning_content`, which this build
> leaves null. Send `chat_template_kwargs: {"enable_thinking": false}`. openclaw
> does this natively via `"reasoning": true` + `"compat": {"thinkingFormat":
> "qwen-chat-template"}` (set here); dgxlib consumers get it from
> `thinking_default: false`.
>
> ---
>
> **▶ PREV (2026-08-04, superseded): CLIENTS ARE STILL BROKEN AGAINST spark1:8001 — AND the checkpoint upgrade just re-broke the one that was fixed.**
> spark1:8001 now serves **`deepseek-ai/DeepSeek-V4-Flash-0731`** (upgraded
> 2026-08-04 from the DSpark preview — see the top LIVE banner) — and
> **spark2:8001 serves nothing at all** (headless TP worker). Any client that
> sends an explicit model id — MemPalace `llm_model`, llm_wiki's
> custom-provider Model, CampaignGenerator's `DGX_MODEL` /
> `DGX_DEFAULT_MODEL`, the opencode `dgx` provider — **will 400** until it is
> repointed to `deepseek-ai/DeepSeek-V4-Flash-0731`. **No client configs have
> been changed as part of this doc sync.** Repointing these four remains the
> outstanding TODO from the 2026-08-03 keep decision.
> **Unaffected:** anything using Ollama `:11434` for embeddings (MemPalace's
> embedding path still works).
>
> **REGRESSION — `openclaw` was fixed 2026-07-30, now broken again by the
> 2026-08-04 checkpoint upgrade.** A **fifth** client not listed above:
> `~/.openclaw/openclaw.json`, provider `deepseek-v4-local` →
> `http://192.168.1.147:8001/v1`. Checked live 2026-08-04: it's still
> configured with model id `deepseek-ai/DeepSeek-V4-Flash-DSpark` (3
> occurrences — `primary`, the provider-model key, and the model `id`/`name`
> fields) — the id that stopped being served the moment this doc's LIVE
> banner changed to `-0731`. **Not fixed as part of this doc sync** — same
> reasoning as the four clients above (a config-file edit, not a doc edit),
> flagged here so it isn't mistaken for still-working. The July fixes
> (`contextWindow` 262144, `maxTokens` 16000, stale-provider cleanup) remain
> valid and don't need redoing — only the model-id string needs to change.
> If the **revert to the DSpark preview** is run instead, openclaw's current
> id is already correct and needs no change.

> **⚠️ PREV (2026-06-20): spark1:8001 AND spark2:8001 serve `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`** (single-box on each box).
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

**Not reachable from the Linux workstation — this must be done on the Windows
box by hand.** App Settings → OpenAI-compatible endpoint:

- Endpoint: `http://192.168.1.147:8001/v1`
- Model: **`qwen3.8-flash-next`**  ← updated 2026-09-10

Settings persist to `%APPDATA%\com.llmwiki.app` on Windows.

> **⚠ llm_wiki and reasoning traces.** llm_wiki is the client that got
> Nemotron rejected from spark1 in the first place (2026-05-18 → 05-19) because
> it cannot strip `<think>` traces. This model emits traces **by default**, so
> check this before trusting it: the slot runs `--reasoning-parser qwen3`, which
> should keep the trace out of `content` and put it in `reasoning` — but if
> llm_wiki shows think-tags or empty answers, that is the known failure mode.
> The alternative is spark2 (`http://192.168.1.121:8001/v1`,
> `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`), which llm_wiki was already happy with.

### CampaignGenerator (`~/src/CampaignGenerator`)

**Current (2026-09-10), spark1:**

```bash
DGX_MODEL=qwen3.8-flash-next python session_doc.py ... \
  --dgx-endpoint http://192.168.1.147:8001/v1
```

**Or spark2, which needs no change at all:**

```bash
DGX_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 python session_doc.py ... \
  --dgx-endpoint http://192.168.1.121:8001/v1
```

The model id is NOT hardcoded in this repo: `campaignlib/api/backends.py`
resolves `model_override or $DGX_MODEL or DGX_DEFAULT_MODEL`, where
`DGX_DEFAULT_MODEL = wiring_get("dgx_model")` reads the **mneme-rendered**
`config/wiring.yaml` (do-not-edit; rendered from `hypostasis.yaml` via
`hypostasis apply`). On this workstation neither file exists, so `dgx_model`
resolves to `None` and `DGX_MODEL`/`--model` is mandatory. To make it a
default, set `machines.dgx.default_model` in `~/.config/hypostasis/hypostasis.yaml`
and re-apply — **in the mneme repo, not here.**

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

> **⚠ 2026-09-10 — spark1 rebuild is now a DIFFERENT procedure from spark2.**
> spark1 runs `qwen3.8-flash-next` out of a locally built image and a locally
> assembled hybrid checkpoint; neither comes from a plain `docker pull` + HF
> download. Full sequence on a wiped spark1:
>
> ```bash
> # 1. recipe + image (~1 min build; 10 patches on the official vLLM image)
> ssh spark 'git clone https://github.com/blazux/qwen3.8-Flash-DGX.git ~/qwen3.8-Flash-DGX \
>            && cd ~/qwen3.8-Flash-DGX && docker build -t qwen38-flash-dgx .'
>
> # 2. checkpoint (~126 GiB, resumable, ~25-40 min on a good link).
> #    HF_TOKEN lives in ~/.bashrc BEHIND the non-interactive guard, so
> #    `ssh spark 'cmd'` cannot see it (memory `feedback_profile_export_keyword`).
> #    Pull it out inside the remote shell:
> ssh spark 'eval "$(grep -m1 "HF_TOKEN=" ~/.bashrc | sed "s/^[[:space:]]*//; s/^export //")"; \
>            export HF_TOKEN; cd ~/qwen3.8-Flash-DGX && scripts/download-weights.sh'
>
> # 3. hybrid checkpoint (~10 min, +13 GB) — fp8 side layers, +20% decode
> ssh spark 'cd ~/qwen3.8-Flash-DGX && scripts/prepare-hybrid.sh'
>
> # 4. serve (from the WORKSTATION; ~14.5 min to healthy on a cold boot)
> ./spin-up-vllm-qwen38-flash-next.sh
>
> # 5. verify
> ssh spark 'cd ~/qwen3.8-Flash-DGX && scripts/smoke-test.sh localhost:8001'
> ```
>
> Disk needed per box: ~126 GiB checkpoint + ~13 GB hybrid ≈ **140 GB**.
> The two HF caches are NOT shared between boxes.
>
> **spark2 now runs the same model (2026-09-10, later), and it was stocked from
> spark1 over the direct cable — NOT re-downloaded.** The doc used to say doing
> this on spark2 "means downloading all 126 GiB again"; it doesn't. Copying wins
> three ways: no HF pull, no second `prepare-hybrid.sh` run (the prepared
> `-fp8hybrid` snapshot comes along), and spark2 ends up running **the exact
> bytes validated on spark1** instead of a separately re-derived checkpoint.
> **Measured 2026-09-10: image 20.7 GB in 3m48s, checkpoint 139 GB in 4m51s,
> ~425 MB/s** — ssh-cipher-bound, not cable-bound.
>
> ```bash
> # Run from the WORKSTATION. `gx10-3e5c.local` is the NVIDIA-Sync ssh alias that
> # already exists on spark1 and resolves to 10.100.16.2 over the direct cable
> # (`~/.ssh/config` on the box). Going spark1 -> spark2 directly keeps 139 GB
> # off the workstation's LAN link. Reverse the aliases to stock spark1 instead.
> FAST='-o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com'
>
> # 1. recipe repo (pins the same commit; a fresh clone could drift)
> ssh spark "rsync -a --delete -e 'ssh $FAST' \$HOME/qwen3.8-Flash-DGX/ gx10-3e5c.local:\$HOME/qwen3.8-Flash-DGX/"
>
> # 2. image — transfer, do NOT rebuild. The ten patches (incl. the Mamba
> #    block-size fix that makes APC safe) are compiled in; a rebuild is a
> #    chance for them to differ.
> ssh spark "docker save qwen38-flash-dgx:latest | ssh $FAST gx10-3e5c.local 'docker load'"
>
> # 3. checkpoint. The HF cache is ROOT-OWNED (written by the download
> #    container), so the receiving side untars inside a container to get root.
> ssh spark "tar -C \$HOME/.cache/huggingface/hub -cf - models--RadixArk--Qwen3.8-Flash-Next-NVFP4 \
>   | ssh $FAST gx10-3e5c.local 'docker run --rm -i -v \$HOME/.cache/huggingface:/hf --entrypoint tar qwen38-flash-dgx -C /hf/hub -xf -'"
>
> # 3b. ⚠ REQUIRED FOLLOW-UP. Exactly one file in the tree is mode 600 root-only
> #     (`trees/<rev>.json`, xet dedup metadata). `tar` as kostadis cannot read
> #     it, skips it, and exits non-zero only at the very END — a status the
> #     pipeline above discards, so step 3 LOOKS clean. Copy it through a
> #     container:
> R='$HOME/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4'
> REV=7b719225242aacd3dbd3f9407468c2ee9a9d2594
> ssh spark "docker run --rm -v $R/trees:/t:ro --entrypoint cat qwen38-flash-dgx /t/\$REV.json \
>   | ssh $FAST gx10-3e5c.local 'docker run --rm -i -v $R/trees:/t --entrypoint sh qwen38-flash-dgx -c \"cat > /t/\$REV.json && chmod 600 /t/\$REV.json\"'"
>
> # 4. VERIFY THE BYTES, not the byte count. Must match between boxes.
> HASH='docker run --rm -v $HOME/.cache/huggingface:/hf:ro --entrypoint sh qwen38-flash-dgx -c "cd /hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4 && find . -type f | sort | xargs -P 8 -n1 md5sum | sort -k2 | md5sum"'
> ssh spark "$HASH"; ssh spark2 "$HASH"     # 2026-09-10: 671788c9611b59321593c21741e16f42 on both
>
> # 5. free the port (the spin-up script only force-removes `qwen38-flash`, so a
> #    container under any other name — e.g. the old `vllm-chat` — collides), then serve
> ssh spark2 'docker rm -f vllm-chat'
> HOST=spark2 ./spin-up-vllm-qwen38-flash-next.sh
>
> # 6. verify
> ssh spark2 'cd ~/qwen3.8-Flash-DGX && scripts/smoke-test.sh localhost:8001'
> ssh spark2 'docker logs qwen38-flash 2>&1 | grep "attention block size"'  # MUST say 1600
> ```
>
> **REVERT spark2 to Qwen3-Next-80B** (77 GB checkpoint still cached on the box —
> a load, not a download; ~15 min to healthy):
> `ssh spark2 'docker rm -f qwen38-flash'; ssh spark2 'PREFIX_CACHING=1 MAX_SEQS=8 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'`

### Post-reboot (containers exist, just stopped)

**As of 2026-07-02 this is mostly automatic:** `vllm-chat` on both boxes
carries `--restart unless-stopped`, so after a reboot/powercycle the docker
daemon relaunches it at boot with its original flags — no manual step, just
wait ~5–10 min for weight reload + compile, then verify with
`curl -sS http://<box>:8001/v1/models`. Ollama (spark2 embed) auto-starts
via systemd. The recipe below is only needed for a container that was
**manually stopped** (`docker stop` sticks despite the restart policy):

```bash
# spark1: embed first, then chat
ssh spark 'docker start vllm-embed'
until curl -sS --max-time 2 http://192.168.1.147:8000/v1/models 2>/dev/null | grep -q '"id"'; do sleep 3; done
ssh spark 'docker start vllm-chat'
until curl -sS --max-time 2 http://192.168.1.147:8001/v1/models 2>/dev/null | grep -q '"id"'; do sleep 10; done

# spark2 in parallel (independent box) — Ollama auto-starts via systemd; just start chat
# NOTE: the spark2 chat container is `qwen38-flash` as of 2026-09-10 (later), NOT `vllm-chat`.
ssh spark2 'docker start qwen38-flash'
until curl -sS --max-time 2 http://192.168.1.121:8001/v1/models 2>/dev/null | grep -q '"id"'; do sleep 10; done
```

Warm-restart cost: both boxes now run `qwen38-flash` and take **~14 min** to
healthy from a cold page cache (76.75 GiB of weights + an 80s draft-head load
+ ~2 min of profile/compile/KV-alloc) — measured 14.5 min on spark1 and 14.4 min
on spark2. The spin-up script drops the page cache first on purpose, so a
restart does not get to reuse the warm PLE table.

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

4. **Install Ollama** (for embeddings — replaces vllm-embed):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   sudo mkdir -p /etc/systemd/system/ollama.service.d
   sudo tee /etc/systemd/system/ollama.service.d/override.conf <<'EOF'
   [Service]
   Environment="OLLAMA_HOST=0.0.0.0:11434"
   Environment="OLLAMA_FLASH_ATTENTION=1"
   Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
   Environment="OLLAMA_NUM_PARALLEL=8"
   EOF
   sudo systemctl daemon-reload && sudo systemctl enable ollama && sudo systemctl start ollama
   ollama pull qwen3-embedding:0.6b
   ```
   Embed endpoint: `http://192.168.1.121:11434/v1/embeddings`, model `qwen3-embedding:0.6b`.

5. **Start vllm-chat** with the MTP-2 latency config:
   ```bash
   MAX_SEQS=3 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh
   ```

6. **Smoke-test** chat and embed:
   ```bash
   curl -sS http://192.168.1.121:8001/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"Qwen/Qwen3-Next-80B-A3B-Instruct-FP8","messages":[{"role":"user","content":"Say only OK"}],"max_tokens":20}'
   curl -sS http://192.168.1.121:11434/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"model":"qwen3-embedding:0.6b","input":"hello"}'
   ```

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
# ⚠ As of 2026-09-10 (later) the spark2 chat container is `qwen38-flash`,
#   NOT `vllm-chat`, and it serves `qwen3.8-flash-next` — same as spark1.

# Is it running?
ssh spark2 'docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'

# Watch / restart / stop / start
ssh spark2 'docker logs -f qwen38-flash'
ssh spark2 'docker restart qwen38-flash'   # ~14 min back to healthy, not 30-60s
ssh spark2 'docker stop qwen38-flash'
ssh spark2 'docker start qwen38-flash'

# What model is served?
curl -sS http://192.168.1.121:8001/v1/models

# Smoke-test. `enable_thinking:false` matters: without it the trace is ON by
# default and eats most of max_tokens (28 completion tokens for "OK", 24 of
# them reasoning), and it arrives in `reasoning`, NOT `reasoning_content`.
curl -sS http://192.168.1.121:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Say only OK"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":20}'

# Full smoke (coherence + determinism + prefix-cache hit + prefill/decode)
ssh spark2 'cd ~/qwen3.8-Flash-DGX && scripts/smoke-test.sh localhost:8001'

# The prefix-caching landmine check — cheap, and the failure is silent.
# MUST print 1600. If it prints 8, the image lacks the Mamba block-size fix
# and APC is silently corrupting state on every cache hit.
ssh spark2 'docker logs qwen38-flash 2>&1 | grep "attention block size"'

# HF cache size (spark2-only — not shared with spark1)
ssh spark2 'du -sh ~/.cache/huggingface'

# CURRENT spark2 bring-up (run from the WORKSTATION, not on the box):
HOST=spark2 ./spin-up-vllm-qwen38-flash-next.sh
# Revert spark2 to Qwen3-Next-80B (77 GB weights still cached — a load, ~15 min):
ssh spark2 'docker rm -f qwen38-flash'
ssh spark2 'PREFIX_CACHING=1 MAX_SEQS=8 GPU_UTIL=0.80 SPEC_TOKENS=2 bash ~/spin-up-vllm-qwen3-next-80b-mtp.sh'
# Older occupant scripts, kept as runbooks:
# ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b.sh'  # Qwen3-Next 80B Instruct FP8, plain, 128K, fp8 KV, hermes, seqs 8
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
