# Gemma 4 31B Dense — serving spec (spark2 candidate)

**Status:** SPEC / not deployed. This is a runbook-in-waiting, not live
state. Nothing in `current-setup.md` or `dgxlib/models.yaml` changes until
the swap actually happens — at which point the hard rule kicks in (update
both, same change).

**Snapshot author date:** 2026-06-20.

**One-line intent:** put Gemma 4 31B *dense* on **spark2** (the experimental
box), keeping `Qwen3-Next-80B-A3B-Instruct-FP8` live on **spark1** for the
production clients. One model per box — they do **not** co-reside (the 80B at
util 0.88 leaves ~2.4 GB free; a second vLLM container OOMs).

---

## 1. Model identity (verified against the live HF config)

Pulled from `google/gemma-4-31B/config.json` on 2026-06-20:

| field | value |
|---|---|
| HF id (chat) | `google/gemma-4-31B-it` (base: `google/gemma-4-31B`) |
| params | 30.7B text + ~0.55B vision ≈ **31B total**, **dense** (not MoE) |
| `num_hidden_layers` | 60 |
| `hidden_size` | 5376 |
| `num_attention_heads` | 32 (query) |
| `num_key_value_heads` | **16** (GQA 2:1) |
| `head_dim` | 256 |
| attention pattern | **interleaved**: sliding-window local + full global, **every 6th layer global** → **10 global + 50 local** |
| `sliding_window` | 1024 |
| `rope_theta` | 10 000 (local) / 1 000 000 (global) — dual |
| `max_position_embeddings` | 262144 (**256K**) |
| `vocab_size` | 262144 (large; tied embeddings ≈ 2.8 GB of the weight budget) |
| `torch_dtype` | bfloat16 |
| modality | multimodal (text + image + video); ~550M vision encoder, 280 soft tokens/image |
| native features | configurable **thinking mode**, **function calling**, 140+ languages |
| license | Apache 2.0 |

---

## 2. Correction to the previous turn — this is NOT "full KV on every layer"

I initially warned that "dense ⇒ every layer carries full KV ⇒ long context
is expensive." **That is wrong for Gemma 4.** It's a sliding-window/global
*interleaved* model: 50 of 60 layers only ever hold a **fixed 1024-token** KV
window, and just **10 global layers** carry the full sequence. The KV story
is therefore much closer to the Qwen3-Next hybrid than to a Llama-shape dense
model — long context is *affordable*, not prohibitive. The spec below is
built on the corrected math.

(Contrast: Qwen3-Next's non-attention layers are Gated DeltaNet with *zero*
KV; Gemma's local layers carry a small *constant* 1024-token KV. Both keep
the per-token, per-sequence KV growth on a handful of full-attention layers.)

---

## 3. KV-cache math (the load-bearing numbers)

Per token, per layer: `2 (K+V) × num_kv_heads(16) × head_dim(256) × dtype`.

- **BF16 KV:** 16 384 B = **16 KiB / token / layer**
- **FP8 KV:** 8 192 B = **8 KiB / token / layer**

Split by layer type for a sequence of length *L* (with *L* > 1024):

| layers | count | tokens held | BF16 | FP8 |
|---|---:|---|---:|---:|
| local (sliding 1024) | 50 | 1024 (constant) | **0.78 GiB** | 0.39 GiB |
| global (full) | 10 | *L* | 10·*L*·16 KiB | 10·*L*·8 KiB |

Per **single sequence**:

| context *L* | BF16 KV/seq | FP8 KV/seq |
|---|---:|---:|
| 128K (131072) | ~20.8 GiB | **~10.4 GiB** |
| 256K (262144) | ~40.8 GiB | ~20.4 GiB |

The constant 0.78 GiB local floor is negligible; the 10 global layers
dominate and scale linearly with context.

---

## 4. Weight memory

| precision | weights | notes |
|---|---:|---|
| BF16 | **~62 GB (~58 GiB)** | native dtype; **recommended for quality** |
| FP8 (vLLM dynamic `--quantization fp8`) | **~31 GB** | frees ~27 GiB for KV/concurrency, but **Gemma is quantization-sensitive** — Google ships QAT GGUFs precisely because PTQ degrades it. Treat FP8 as a *capacity* knob, validate quality before trusting it |

There is no first-party FP8 checkpoint in the search results — only BF16,
GGUF, and QAT-GGUF (Q4). vLLM doesn't serve GGUF for production here, so the
realistic choices are **BF16 weights** or **vLLM on-the-fly FP8**.

---

## 5. Placement & memory budget (spark2, alongside `vllm-embed`)

spark2 already runs `vllm-embed` (`Qwen/Qwen3-Embedding-0.6B`, util 0.05,
~6 GiB, always-on). The Gemma chat slot replaces spark2's current
`Qwen3-Next-80B` `vllm-chat`. Device ≈ 128 GB nominal (~120 GiB usable).

### Recommended baseline — BF16 weights + FP8 KV @ 128K

```
GPU_UTIL 0.82  →  ~98 GiB chat cap
  weights         ~58 GiB
  activations +
  CUDA graphs     ~8 GiB
  → KV pool       ~32 GiB  →  ~3× full-128K sequences (10.4 GiB each)
embed (0.05)      ~6 GiB
OS / docker       ~8 GiB
                  ----------
total             ~112 GiB  <  ~120 GiB   (tight but fits)
```

`--max-num-seqs 4` is a safe admission cap — paged KV means idle/short
sequences cost nothing; only concurrent *full-length* requests draw down the
pool. **Heed the cross-box lesson:** on GB10 unified memory the CPU-side
scheduler heap competes with the GPU reservation, and spark2 OOMed under
concurrency at util 0.85 with the 122B. If startup or load OOMs, drop
`GPU_UTIL` 0.82 → 0.80, then trim `MAX_LEN`.

### Variants

| goal | weights | KV | MAX_LEN | MAX_SEQS | budget feel |
|---|---|---|---:|---:|---|
| **baseline (quality)** | BF16 | fp8 | 131072 | 4 | ~3 concurrent 128K, fits at 0.82 |
| long-context | BF16 | fp8 | 262144 | 1–2 | one 256K seq ≈ 20.4 GiB; pool barely holds it — raise util or go FP8 weights |
| capacity (concurrency) | FP8 | fp8 | 131072 | 8 | weights ~31 GiB frees ~27 GiB → ~6 concurrent 128K, **quality risk** |

---

## 6. Proposed spin-up script

New file `spin-up-vllm-gemma4-31b-dense.sh`, modeled on
`spin-up-vllm-qwen3-next-80b.sh` (same `lib-vllm-spinup.sh` helpers, same
container slot `vllm-chat:8001`). Key body:

```bash
GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-4-31B-it}"
GEMMA_PORT="${GEMMA_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.82}"          # 0.80 fallback if spark2 OOMs under load
MAX_LEN="${MAX_LEN:-131072}"          # 128K; 262144 native, 32768 for max concurrency
MAX_SEQS="${MAX_SEQS:-4}"             # admission cap; KV-pool bound, not compute
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
TOOL_PARSER="${TOOL_PARSER:-gemma4}"  # vLLM ships this (used by the 26B MoE script)
QUANT="${QUANT:-}"                    # set "fp8" for FP8 weights (capacity, quality risk)
CONTAINER_NAME="vllm-chat"
IMAGE="vllm/vllm-openai:latest"       # confirm tag recognizes gemma-4-31B arch

docker run -d --runtime nvidia --gpus all \
  --name "${CONTAINER_NAME}" -p "${GEMMA_PORT}:${GEMMA_PORT}" --ipc=host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  "${IMAGE}" "${GEMMA_MODEL}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs "${MAX_SEQS}" \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --dtype bfloat16 \
  ${QUANT:+--quantization "${QUANT}"} \
  --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER}" \
  --host 0.0.0.0 --port "${GEMMA_PORT}"
```

Reuse `vllm_wait_ready` (40-min budget; first run pulls ~62 GB BF16) and
`vllm_smoke_test`. Deploy via `scp … spark2:~/ && ssh spark2 'bash ~/…'`.

---

## 7. Integration considerations (the parts that can break clients)

1. **Tool calling — `--tool-call-parser gemma4`.** vLLM already ships it
   (the 26B MoE slot used it). Gate before trusting:
   `MODEL=google/gemma-4-31B-it ./test-toolcall.sh`. This matters for
   opencode / CampaignGenerator if either is pointed at spark2.

2. **Thinking mode — default it OFF.** Gemma 4 has a native, *configurable*
   thinking mode. For the render-heavy production workloads (pdf-translators,
   narration) thinking-off is correct, same call-intent logic as the Qwen
   Instruct choice. **Open question:** whether vLLM has a Gemma reasoning
   parser to split `<think>` traces into `reasoning_content`. If thinking is
   ever enabled without one, raw trace tags leak into `content` — the exact
   failure that got Nemotron rejected from llm_wiki and leaked under opencode.
   Verify before enabling thinking; keep it off by default.

3. **Multimodal weights load even for text serving.** The ~550M vision tower
   loads regardless; it's small (~1 GB) so just ignore it, or cap with
   `--limit-mm-per-prompt '{"image":0,"video":0}'` if vLLM complains. We serve
   text only.

4. **vLLM image must recognize the `gemma-4-31B` arch.** The 26B *MoE* worked
   on `:latest`; confirm the dense 31B class is recognized on whatever tag is
   current at deploy time. Failure mode: `architecture unknown` at startup →
   pull a newer tag.

5. **dgxlib entry (apply only at deploy).** Keyed on `google/gemma-4-31B-it`:
   - `can_think: true` (native thinking mode exists)
   - `thinking_default: false` (render-heavy call intent)
   - `read_timeout`: moderate — dense 31B decodes faster than the 80B on
     short output but this isn't a slow reasoning model; start at the family
     default and tune.

---

## 8. Validation plan (before declaring it live)

1. `/v1/models` returns `google/gemma-4-31B-it`, `max_model_len` == MAX_LEN.
2. Smoke chat completion ("Say only OK") returns clean `content`.
3. `test-toolcall.sh` PASS with `gemma4` parser (null content + parseable
   `tool_calls`).
4. Confirm KV pool from the vLLM startup log (`GPU KV cache size`) matches the
   §3 estimate — **size MAX_LEN against the measured pool, not this guess**
   (the standing KV-pool rule).
5. Long-context sanity if running 256K: a needle probe
   (`bench-longctx-needle.sh`) — the interleaved sliding window is the thing
   most likely to surprise on recall.
6. Prefill vs decode both measured (`bench-prefill.sh` / `bench-decode.sh`) —
   dense 31B vs the 3B-active 80B is a real prefill tradeoff worth logging in
   `model-comparisons.md`.

---

## 9. Open questions / risks

- **FP8 quality on Gemma.** Gemma's quant sensitivity is documented (QAT
  exists for a reason). If concurrency forces FP8 weights, A/B against BF16
  before trusting it for anything downstream-feeding.
- **Gemma reasoning parser** availability in vLLM — unverified (see §7.2).
- **Dense vs sparse prefill cost.** 31B dense activates all 31B/token vs the
  80B's ~3B active. On the read-heavy workloads this box favors, the 80B may
  actually *prefill* faster despite being larger — measure, don't assume.

---

## 10. NVFP4 variant — evaluated (and the path corrected)

A Gemini suggestion proposed serving this model as **NVFP4 via
TensorRT-LLM** (`trtllm-build` engine + `modelopt` quantization). The core
idea — NVFP4 weights on GB10's native FP4 tensor cores — is legitimate and
already proven on this box. The *delivery* (TRT-LLM stack switch) and several
specific claims are not. Corrected:

**What's right:**
- NVFP4 weights for a 31B dense model ≈ **~16–18 GB** (0.5 B/param + FP8
  microscale factors). Real.
- GB10 / sm_120(1) has **native FP4 tensor cores**; NVFP4 runs on **real
  CUTLASS FP4 kernels** here — *already validated on this box* via the
  Nemotron-3-Super NVFP4 experiment (`project_nemotron3_super_nvfp4`).
- For a tiny-footprint weight set, KV pool / context headroom is generous.

**What's wrong or overstated:**
1. **"Completely alleviates the 273 GB/s bandwidth ceiling" — false.** Decode
   stays bandwidth-bound; NVFP4 *lowers the denominator*, it doesn't remove
   the ceiling. The math: 273 GB/s ÷ ~16 GB/token ≈ **~17 tok/s** single-stream
   ceiling for a dense 31B. The pitch's own **"clear ~40 tok/s"** is ~2.5× over
   what the hardware permits at batch 1 — only reachable as *aggregate* across a
   batch, not per request.
2. **Decode is a downgrade vs what's already running.** The live
   `Qwen3-Next-80B-A3B` reads only **~3B active params/token** (~3–4 GB) →
   ~70–80 tok/s ceiling. A *dense* 31B at NVFP4 reads all ~16 GB/token → slower
   per-token decode than the 80B MoE, regardless of bit-width. MoE active
   sparsity beats 4-bit dense on the decode axis.
3. **Wrong axis for this user.** NVFP4's headline win is decode bandwidth; this
   box is **read-heavy / prefill-bound** (`user_workflow_read_heavy`,
   `feedback_llm_bench_prefill_vs_decode`). The genuinely interesting NVFP4 angle
   here is **FP4 prefill *compute*** on Blackwell tensor cores — which the pitch
   never measures.
4. **Stack switch cost is hidden.** vLLM → TensorRT-LLM means per-config engine
   recompiles, a different server, and — critically — **losing the `gemma4`
   vLLM tool-call parser** the clients depend on (opencode, CampaignGenerator,
   dgxlib all assume the OpenAI-compatible vLLM endpoint). TRT-LLM tool calling
   is far less mature.
5. **Several flags/APIs look hallucinated** — verify before trusting:
   `--use_fp4_vector_matmul` (not a known `trtllm-build` flag),
   `python -m modelopt.torch.quantization.convert` (real flow is
   `examples/quantization/quantize.py --qformat nvfp4`),
   `KvCacheConfig(dtype="nvfp4")` (FP4 KV cache support is dubious/lossy; FP8 KV
   is the real option), `SamplingConfig` (TRT-LLM uses `SamplingParams`).
6. **Context contradiction:** sells "massive document contexts," then caps
   `--max_seq_len 73728` (~72K), throwing away Gemma 4's 256K and its cheap
   sliding-window KV.
7. **Quality blind spot:** Gemma is quantization-sensitive (QAT GGUFs exist for
   a reason); NVFP4 PTQ is *more* aggressive than FP8. The vision tower also
   complicates `modelopt` calibration. No quality A/B is mentioned at all.

**Corrected recommendation:** if you want NVFP4 Gemma, do it **on vLLM**, not
TRT-LLM — reuse the validated Nemotron NVFP4 recipe and the entire §6/§7 serving
+ client stack stays intact. Treat it as a **calibration A/B vs the BF16
baseline**, measuring **prefill** (the axis you care about) and **quality** (the
axis Gemma is fragile on), not decode tok/s. The TRT-LLM path is worth running
*only* if feeling that stack's friction is itself the goal — name it as a
separate detour, not the default.

---

## 11. Revert / coexistence

- This spec targets **spark2**; spark1's production Qwen3-Next-80B is
  untouched.
- Revert spark2 to Qwen: `ssh spark2 'bash ~/spin-up-vllm-qwen3-next-80b.sh'`.
- `vllm-embed` on spark2:8000 stays up throughout — it's the always-on
  embeddings path; do not stop it.

---

**Sources:** model existence/features from the Gemma 4 31B launch coverage and
HF model cards; all architecture numbers in §1/§3 read directly from
`google/gemma-4-31B/config.json` (2026-06-20).
