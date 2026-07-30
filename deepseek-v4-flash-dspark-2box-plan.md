# DeepSeek-V4-Flash-DSpark across 2x DGX Spark — plan

**Status as of 2026-07-30: weights + image staged on both boxes, nothing deployed.**
Steps 0 and 1 are done; the next action (Step 2) is the destructive one.
This is a plan/runbook, not an inventory. `current-setup.md` is untouched
on purpose — the Sparks are still serving Qwen3-Next-80B and this config
has never been brought up. See "Doc obligations" at the bottom for what
must change *if* we actually deploy it.

---

## 1. Why

`reference_deepseek_v4_flash_2box` (memory) captured the pre-DSpark
NVIDIA-forum recipe: V4-Flash FP8, TP=2, MTP-2 spec decode, ~44 tok/s
decode, 200K ctx. DSpark is the follow-on DeepSeek released 2026-06-27 —
**same checkpoint, plus a speculative-decoding module**. On 2x Spark the
published numbers move to ~60-67 tok/s single-stream on code, 1M context
via an NVFP4 KV path, and real concurrency (6 seqs @ 1M ≈ 182 tok/s
aggregate; 16 seqs @ 200K ≈ 315 tok/s).

The bar to beat is Qwen3.5-122B-FP8 (`project_qwen35_first_working_model`),
and the axis we actually care about is **prefill**, not decode
(`user_workflow_read_heavy`, `feedback_llm_bench_prefill_vs_decode`).

## 2. Done so far

**2026-07-29:**

- [x] Confirmed `deepseek-ai/DeepSeek-V4-Flash-DSpark` exists, is public
      and ungated: **166.9 GB / 156 GiB, 48 safetensors shards, 74 files**,
      FP4 experts + FP8 dense, 284B total / 13B active.
      (Bigger than the 149 GB in the old note — that was plain V4-Flash;
      DSpark ships the draft module on top.)
- [x] Downloaded to **spark1** into `~/.cache/huggingface/hub/` via the
      vLLM image (`--entrypoint bash` + `hf download`). 23 min @ 117 MB/s.
      0 `.incomplete` files, exit 0.
- [x] Copied to **spark2** over the cable (rsync -a to `~/dspark-staging`,
      ~367 MB/s / 2.94 Gb/s, ~7 min). 156 GiB / 48 shards present.

**2026-07-30:**

- [x] **Step 0 done** — spark2's `~/dspark-staging` moved into the HF cache
      as `models--deepseek-ai--DeepSeek-V4-Flash-DSpark`, chowned root:root,
      staging dir gone (no duplicate 156 GB left behind).
      Both boxes now verified byte-for-byte equivalent:

      | | spark1 | spark2 |
      |---|---|---|
      | Size | 156 GB | 156 GB |
      | `refs/main` | `62af8fff…` | `62af8fff…` |
      | Shards | 48 | 48 |
      | Total files | 74 | 74 |
      | `.incomplete` | 0 | 0 |
      | Owner | root:root | root:root |

      The "74 files" figure reconciles: 57 in the snapshot root plus the
      `encoding/` (incl. `encoding/tests/`) and `inference/` subdirs.
- [x] **Step 1 done** — `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` pulled on
      both boxes, identical digest
      `sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`,
      18.8 GB. Image internals (verified, not assumed):
      **vLLM `0.25.2.dev0+g752a3a504.d20260714`**, torch `2.11.0+cu130`,
      entrypoint is already `vllm serve` (so pass flags only, drop the
      `vllm serve` prefix from §4), and `python3` exists but **`python`
      does not** — matters if a wrapper script shells out.
- [ ] **NEXT: Step 2 — the destructive one.** Needs the user's go-ahead.
- [ ] Everything else.

Both boxes were still serving `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`
healthily after Steps 0 and 1 — neither step touches the running service.

## 3. Live-state facts verified today (these corrected stale notes)

| Fact | Reality on the boxes | What the notes said |
|---|---|---|
| Cable IPs | **10.100.16.1 / 10.100.16.2** on `enp1s0f0np0` | `todo_minimax_m27_two_box` says 192.168.100.1/.2 via `99-fastlink.yaml` — **that config is gone** |
| Second cable port | `enp1s0f1np1` DOWN | — |
| spark1→spark2 ssh | Works via `~/.ssh/id_ed25519_nvsync_cluster_assistant`, host alias `gx10-3e5c.local` | not recorded |
| HF cache ownership | `~/.cache/huggingface/hub` is **root:root** on both boxes | not recorded — bit us on rsync |
| sudo on spark2 | **needs a password** (no passwordless sudo) | not recorded |
| sudo on **spark1** | **also needs a password** (corrected 2026-07-30 — the note below implied spark1 was fine) | not recorded |
| Both boxes now | `vllm-chat:8001` serving `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`, vLLM `v0.22.0-aarch64`, 262144 ctx | matches `current-setup.md` |
| Disk | spark1 2.5T free, spark2 2.7T free | fine |

The cache being root-owned is the non-obvious one: **every host-side tool
that writes to `~/.cache/huggingface/hub` will fail with permission
denied**, because the vLLM containers create everything in there as root.
Do cache writes from inside a container, or stage in `$HOME` and move.
And since sudo needs a password on *both* boxes, even *reading* the cache
over a non-interactive ssh has to go through a container:

```bash
ssh spark 'docker run --rm -v /home/kostadis/.cache/huggingface:/hf \
  --entrypoint bash vllm/vllm-openai:v0.22.0-aarch64 -c "ls /hf/hub/"'
```

Cable verification followed `feedback_test_tcp_not_just_ping_rdma` —
ping *and* TCP *and* a real ssh, not just ICMP.

## 4. The recipe

Two independent public recipes converge on nearly identical flags. Prefer
the **prebuilt image** — the other path is a 3-stage Docker build.

**Image (recommended):** `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
(pull on both nodes; ships Keys' concurrency patch prebuilt)

Pulled and inspected 2026-07-30 — digest
`sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`,
18.8 GB, **vLLM `0.25.2.dev0+g752a3a504.d20260714`**, torch `2.11.0+cu130`.
Two practical notes: the image entrypoint is already `vllm serve`, so pass
the flags below *without* the `vllm serve` prefix; and it has `python3` but
no `python` on `PATH`.

**Alternative:** `vllm-dspark-runtime:dspark-nvfp4-stage-c`, built A→B→C
from `recipe/nvfp4/Dockerfile.*` (tonyd2wild), based on Rafael Caricio's
DSpark vLLM PR, Patch 3 (`e83606a`) baked in.

Note this is **not** our usual `vllm/vllm-openai:v0.22.0-aarch64` — DSpark
needs the fork. Same lesson as the jasl/vllm pin in the old note: someone
else did the GB10 build work, don't source-build.

### Identical command on both nodes

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --kv-cache-dtype nvfp4_ds_mla \
  --block-size 256 \
  --max-model-len 1048576 \
  --max-num-seqs 6 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --moe-backend flashinfer_b12x \
  --async-scheduling \
  --enable-chunked-prefill \
  --speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --generation-config vllm
```

`--distributed-executor-backend mp --nnodes 2` — **no Ray**, same win the
old note flagged vs our `reference_spark_2box_vllm` recipe.

### Env vars (not optional)

```bash
VLLM_USE_B12X_MOE=1                     # "this one env var is the entire speed difference"
VLLM_USE_FLASHINFER_SAMPLER=1
VLLM_USE_B12X_WO_PROJECTION=1
VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1
VLLM_DSPARK_REPLICATE_MARKOV_W1=1       # tonyd2wild variant
HF_HUB_OFFLINE=1                        # cache is warm on both nodes — set it
VLLM_HOST_IP=10.100.16.1                # head (spark1)
WORKER_VLLM_HOST_IP=10.100.16.2         # worker (spark2)
NCCL_SOCKET_IFNAME=enp1s0f0np0
TP_SOCKET_IFNAME=enp1s0f0np0
GLOO_SOCKET_IFNAME=enp1s0f0np0
MASTER_PORT=25440
```

### Landmines (from the recipe authors, not us)

- `num_speculative_tokens` must be **3**, not 5 — CUDA-graph ladder
  constraint at `max_num_seqs=6`. (The HF model card advertises 7; the
  Spark recipes use 3. Believe the Spark recipes.)
- Do **not** pass `--override-generation-config` — the 2026-07-03 garble
  fix removes it.
- Do **not** set `VLLM_USE_B12X_FP8_GEMM=1` on Stage C — DeepGEMM assertion
  failure.
- **Worker node starts first**, then head.
- Reported in the wild: CUDA illegal-memory-access crashes, gibberish
  output, and throughput collapsing to ~10 tok/s at true full-1M context.

### Expected boot signature

```
GPU KV cache size: ~1.9-2.04M tokens
Maximum concurrency for 1,048,576 tokens per request: ~1.81x
```

Per `feedback_size_context_by_kv_pool`: read the actual KV pool from the
log before trusting any context setting.

## 5. Bring-up steps

**Step 0 (safe, do anytime — no service impact):**
Move spark2's staged weights into the HF cache. Same filesystem, so the
move is instant. Must run as root → do it in a container:

```bash
ssh spark2 'docker run --rm -v /home/kostadis:/h --entrypoint bash \
  vllm/vllm-openai:v0.22.0-aarch64 -c \
  "mv /h/dspark-staging /h/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark && \
   chown -R root:root /h/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark"'
```
Then verify both boxes list 48 shards and a `refs/main` pointing at
snapshot `62af8fffb2f7030cac4de2f0169f5b8d1101b646`.

**Step 1 — pull the image on both boxes** (no service impact):
`docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1`

**Step 2 — DESTRUCTIVE, needs the user's go-ahead:**
Stop `vllm-chat` on **both** boxes. This takes down the *chat* model
MemPalace, llm_wiki, and CampaignGenerator point at (`current-setup.md` §6).
Note the containers use `--restart unless-stopped`, so a manual `docker stop`
sticks.

Confirmed 2026-07-30: `vllm-chat:8001` is the **only** running container on
either box — nothing on `:8000`, so there is no `vllm-embed` slot to stop.
(This answers the §7 open question.)

### Ollama stays up — and should

`ollama` is a **systemd service on `:11434`, not a container**, so Step 2
does not touch it. Keeping it is free and desirable:

- It is the live embeddings path (`project_embedding_qwen3_upgrade`) —
  `qwen3-embedding:0.6b` is pulled on both boxes, and MemPalace embeds
  against spark1:11434. So **MemPalace search keeps working through the
  whole DSpark experiment** even with chat down. Step 2's blast radius is
  smaller than first written.
- Nothing is resident right now: `/api/ps` = 0 models on both boxes.
  Ollama loads on demand and unloads after keep-alive, so an idle Ollama
  costs no GPU memory at DSpark launch time.

**Theoretical hazard, no live trigger.** Unified memory means an Ollama
load competes with vLLM's reservation, and spark1 still has large models
*pulled* — `llama3.3:70b` (42.5 GB), `qwen2.5:32b` (19.9 GB),
`qwen2.5:14b` (9.0 GB). Loading any of those on top of a DSpark
reservation would repeat the `feedback_gpu_util_080_default` wedge (~15 GB
host headroom starved sshd's fork; needed a physical reboot).

But per the user (2026-07-30): **llm_wiki no longer uses `qwen2.5:14b`, and
the 32b/70b models haven't been used in ages.** So nothing actually drives
them — the only live Ollama consumer is `qwen3-embedding:0.6b` for
MemPalace, which is 0.64 GB and harmless. Being pulled is not being
loaded, and `/api/ps` confirms nothing is resident.

Conclusion: **no mitigation needed before bring-up.** Don't bother capping
`OLLAMA_MAX_LOADED_MODELS` or shortening keep-alive for this experiment.
If we ever want the risk gone rather than merely dormant, the clean move is
`ollama rm llama3.3:70b qwen2.5:32b qwen2.5:14b` (frees ~71 GB of disk and
removes the failure mode outright) — but disk is not tight (2.3 TB free),
so this is hygiene, not a prerequisite.

Note: this corrects the `project_llmwiki_perf` memory, which recorded
`qwen2.5:14b` as llm_wiki's active ingest model.

**Step 3 — launch worker (spark2) first, then head (spark1).**

**Step 4 — smoke test:** `/v1/models`, then a coherence check. Gibberish is
a *known* failure mode here, so check meaning, not just HTTP 200 — same
discipline as the MTP script's coherence check.

**Step 5 — measure (see §6).**

**Revert:** `bash ~/spin-up-vllm-qwen3-next-80b.sh` on each box (or the
`-mtp` variant on spark2) puts the old world back. Weights are cached, so
it's a load, not a download.

## 6. What to measure — prefill first

The published DSpark numbers are **decode**. Every 2-box thread so far
omits cold TTFT, and the 1-box thread's numbers are alarming (~13 min at
250K cold, ~41 min at 518K). Given `user_workflow_read_heavy`, a model
that decodes at 67 tok/s but takes minutes to read a large prompt is worse
for us than Qwen3-Next today.

Measure, in this order:

1. **Cold TTFT** at 8K / 32K / 128K. This is the go/no-go number.
2. **Prefill tok/s** at the same points.
3. Warm TTFT with `--enable-prefix-caching` (the tonyd2wild variant enables
   it). `reference_apc_hybrid_qwen3_next` shows APC gave ~10x TTFT on
   Qwen3-Next — if APC works here, "load context once, live in it" becomes
   viable and the cold-prefill objection softens a lot.
4. Draft acceptance rate — the number that determines the decode speedup.
5. Decode tok/s, single-stream and at 6 concurrent.
6. Only then: subjective coding quality vs the Qwen3.5-122B bar.

## 7. Open questions

- Does `--gpu-memory-utilization 0.85` starve the host? Our own
  `feedback_gpu_util_080_default` says 0.88 wedged spark2 hard enough to
  need a reboot, and 0.80 is our default. The anemll recipe says 0.85,
  tonyd2wild says 0.80. **Start at 0.80.**
- ~~Does the embed container need to come down too?~~ **Answered
  2026-07-30: no.** `vllm-chat:8001` is the only running container on either
  box; `:8000` is empty. Embeddings live on Ollama, which stays up (see §5).
- Cold-start time: the old note said ~6 min container→serving for plain
  V4-Flash. DSpark adds a draft-module compile.
- Is there a 1-box fallback worth trying instead? Yes — `Entrpi/ds4-on-spark`,
  2-bit, 81 GiB, ~27.7 tok/s, 766K ctx, non-vLLM CUDA fork. Leaves spark1
  alone. Quality cost is real (MMLU 63.9). Not the current plan.

## 8. Doc obligations if this is deployed

Per this repo's CLAUDE.md, bringing this up live means, **in the same change**:

- `current-setup.md`: ports table, VRAM budget, §3 vllm-chat (model id, run
  command, why-these-flags, measured behaviour), §6 client config, §7
  rebuild order, and the snapshot date line.
- `dgxlib/models.yaml`: an entry keyed on the exact served id from
  `curl -sS http://192.168.1.147:8001/v1/models` — `can_think`,
  `thinking_default`, `read_timeout` (generous; DSpark drafts).
- A new `deepseek-v4-flash-dspark-observations.md` for the measurements
  (append-only, per repo convention).

As long as nothing is deployed, none of the above applies and this file is
the only artifact.

## 9. Sources

- https://forums.developer.nvidia.com/t/deepseek-v4-flash-dspark-on-2x-dgx-spark-gb10-big-single-stream-speed-boost-60-67-tok-s-1m-context-now-with-concurrency/374846
- https://forums.developer.nvidia.com/t/1x-spark-tuned-dspark-for-deepseek-v4-flash-35-tok-s-800-prefill-and-fast-multi-agent-serving/376884
- https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
- https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark
- Predecessor recipe (plain FP8 + MTP): https://forums.developer.nvidia.com/t/deepseek-v4-flash-official-fp8-running-across-2x-dgx-spark-tp-2-mtp-200k-ctx-recipe-numbers/370309
