# GPU memory reservation & KV-cache tradeoffs (unified-memory Sparks)

**Read this before changing `--gpu-memory-utilization`, `--max-model-len`, or
`--max-num-seqs` on either Spark.** It's the reasoning behind the defaults the
spin-up scripts ship with (`GPU_UTIL=0.80`, `MAX_LEN=262144`), so you don't
re-derive it (or re-break the box) every time.

Companion to `current-setup.md` (what's running) and the memory
`feedback_gpu_util_080_default` (the one-line rule). This is the long form.

---

## TL;DR — the decision rule

1. **Default `--gpu-memory-utilization` to 0.80, not 0.88.** On a unified-memory
   box the reservation steals host RAM; 0.88 starves the host and wedges sshd.
2. **You are bandwidth-bound, not KV-bound.** Reserving a bigger KV pool than
   your *real* concurrency × context buys nothing — the bus can't feed it. The
   "extra" KV at high util is dead weight that costs host RAM.
3. **Size util to hold the KV you'll actually use, then stop:**
   `util ≈ (weights + overhead + max_num_seqs × max_model_len × 11KB + ~25GB host) / 128GB`,
   capped so the host keeps ≥ ~25 GB. For the 80B at 8×256K that lands at ~0.80.

---

## 1. The unified-memory trap (why 0.88 wedged the box)

GB10 has ONE 128 GB pool shared by GPU and host (Grace+Blackwell unified
memory). `--gpu-memory-utilization U` reserves `U × 128 GB` for vLLM, and what's
left, `(1−U) × 128 GB`, is *all* the host gets — OS, Docker, any co-resident
container (the spark2 `vllm-embed` sidecar, ~6 GB), page cache, and the memory
to `fork()` new processes.

| util | vLLM reservation | host left |
|---:|---:|---:|
| 0.88 | ~113 GB | **~15 GB** |
| 0.80 | ~102 GB | ~26 GB |
| 0.75 | ~96 GB | ~32 GB |

At 0.88 the host had ~15 GB; minus the embed container and page cache, there
wasn't enough to fork. **sshd could no longer complete a login while the
already-resident embed container kept serving** — the box needed a hard reboot
(see incident below). This is host-RAM starvation, *not* a GPU OOM (a GPU OOM
crashes the container with a CUDA error; it doesn't wedge logins).

**The lever is the reservation fraction.** It is NOT a prefill/batch problem.
Lower util → more host RAM. Simple.

---

## 2. The "right number" — KV pool formula

Inside the reservation, memory splits three ways:

```
KV pool (GB) = util × 128 − weights(~80 for 80B FP8) − overhead(~5: activations + CUDA graphs)
```

| util | KV pool | ≈ tokens @ ~11 KB/tok |
|---:|---:|---:|
| 0.88 | ~28 GB | **~2.5M** (measured) |
| 0.80 | ~17 GB | ~1.5M |
| 0.75 | ~11 GB | ~1.0M |

(~11 KB/token is the measured all-full-attn-layers KV for Qwen3-Next at fp8:
2.53M tokens in ~28 GB at 0.88. Qwen3-Next is hybrid — only the periodic
full-attention layers carry KV, the Gated DeltaNet layers carry none — so this
is already cheap per token. A dense model would be far higher; re-measure.)

Three bounds decide the number:

- **Host floor (hard ceiling on util):** keep ≥ ~25 GB host → `util ≤ ~0.80`.
- **KV admission need (soft floor):** pool must hold ≥ one `max_model_len`
  sequence to boot, ideally `max_num_seqs × max_model_len` to avoid preemption.
- **Bandwidth-useful KV (the real cap):** see §3 — usually *below* the admission
  need, which is why the admission ceiling is mostly fictional.

---

## 3. Why bandwidth, not memory, is the binding constraint

Decode reads, per step: the weights once (~3.5 GB, **shared** across the whole
batch) **plus every sequence's KV** (~11 KB/token, **per sequence**). The bus is
273 GB/s. Pin a minimum acceptable decode rate and that caps how much *live* KV
you can feed:

| decode floor | KV bandwidth/step | usable live KV | e.g. |
|---:|---:|---:|---|
| 50 tok/s | ~2.0 GB | **~180K tok** | 4×45K, 2×90K |
| 30 tok/s | ~5.6 GB | **~510K tok** | 4×128K, 8×64K |
| 20 tok/s | ~10 GB | **~920K tok** | 8×115K |
| ~9 tok/s | ~28 GB | ~2.5M tok | the *entire* 0.88 pool |

So the full 2.5M-token pool at 0.88 is only "usable" if you'll decode at ~9
tok/s. Anything above your latency floor is KV you reserved but can never read
fast enough to use — pure waste that stole host RAM. **`max_num_seqs 8 @ 256K`
(2M tokens) is itself optimistic**; the bus can't decode that many full-context
streams at a usable rate, so the admission ceiling overstates real capacity.

---

## 4. Worked example — the current 80B config

- weights ~80 GB, overhead ~5 GB.
- At **util 0.80**: pool ~17 GB ≈ ~1.5M tokens; host ~26 GB.
- 256K request = 262K tokens → boots trivially (1.5M ≫ 262K), and holds
  **~5 concurrent full-256K streams** before paging — already more than
  bandwidth can usefully decode. `max_num_seqs 8` (spark1) / `4` (spark2 MTP)
  is just an admission cap; idle slots cost nothing (paged KV).
- **256K vs 128K costs nothing you were using:** doubling `max_model_len` only
  doubles KV *for sequences that actually grow that long*, and you can't run
  many of those concurrently anyway. So we run the native 256K.

---

## 5. Diagnosing host starvation (the wedge signature)

When util is too high you get a *specific* pattern — learn it:

- ✅ `ping` fine, low latency (kernel alive)
- ✅ an already-running container (e.g. `vllm-embed:8000`) still answers
- ❌ the new chat port refuses connection (never bound)
- ❌ `ssh` accepts the TCP socket but **times out "during banner exchange"**
  (sshd can't fork its child)

That combination = host RAM starvation → **lower the reservation** (or reboot if
already wedged; the container has no `--restart` policy so it stays dead and
frees memory after a reboot). A GPU OOM looks different: the container exits with
a CUDA out-of-memory error and the box stays responsive.

---

## 6. Explicitly deferred: prefix caching

A *bigger* KV pool genuinely helps in one case — **automatic prefix caching**
(more cached prefixes retained → higher hit rate on repeated long contexts, which
the read-heavy workloads have). That's the one reason you might want util above
the bandwidth-derived floor. **Deferred for now** (2026-06-20) — sizing stays
purely bandwidth/concurrency-driven. Revisit if/when APC is turned on; the MTP
build has it off anyway (spec decode disables it).

---

## 7. Incident log

- **2026-06-20:** Bringing up Qwen3-Next-80B + MTP on spark2 at `GPU_UTIL 0.88`
  starved the host during CUDA-graph capture; sshd banner-timed-out while the
  embed container kept serving; **full reboot required**. Refixed at 0.80. Same
  fix `current-setup.md` had already recorded for the cross-box 122B (0.85→0.80).
  Both Sparks then standardized to **0.80 + 256K**.
