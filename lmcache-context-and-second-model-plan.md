# LMCache, bigger context, and a second model on the DSpark 2-box

**Status: ANALYSIS ONLY — nothing deployed, nothing restarted.**
Written 2026-08-04 against the live `deepseek-ai/DeepSeek-V4-Flash-0731`
cross-box endpoint (see the LIVE banner in `current-setup.md`).

The question that prompted this: *can LMCache increase DeepSeek's context,
and can I also have a second model?*

Short answer: **LMCache does not increase the context window** — and you
don't need it to, because there is unused headroom in the KV pool today.
LMCache's real value here is different and arguably better: it is what
makes a large context *affordable to use more than once*. The second
model is a separate problem that **competes for the same bytes**.

The finding that turned out to matter most is in §3: the KV pool's
"concurrency multiple" is really a **cache-residency limit on how many
agents can keep a prefix**, and spending it on context length converts this
from a 3-agent box into a 1-agent box **with no error, no warning, and no
log line** — just a twelve-minute stall on the second agent's second turn.

---

## 1. What was measured (live, 2026-08-04)

All of this came off the running containers — no restart, no guessing.

| Quantity | Value | Source |
|---|---|---|
| Containers | `vllm-dspark` up 18h on both boxes, image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | `docker ps` |
| Host RAM | 121 GiB total; **8 GiB available** (spark1) / **10 GiB** (spark2); 2 GiB swap used | `free -g` |
| Free NVMe | **2.2 TB** (spark1) / **2.4 TB** (spark2) of 3.6 TB | `df -h` |
| Model weights | **79.21 GiB** per box | `model_runner.py:302` |
| Available KV cache memory | **14.22 GiB** per rank | `gpu_worker.py:538` |
| GPU KV pool | **869,357 tokens** | `kv_cache_utils.py:2146` |
| KV layout | *"Using DeepSeek V4 **padded nvfp4_ds_mla** KV cache format."* | `attention.py:87` |
| NVMe sequential read, O_DIRECT | **10.1 GB/s** (3.7 GB in 0.367 s) | `dd bs=8M iflag=direct` |
| vLLM | `0.25.2.dev0+g752a3a504.d20260714`, torch `2.11.0+cu130` | `pip list` |
| `nvcc` in image | **13.0.88, present** | `nvcc --version` |
| `lmcache` installed? | **No.** (`nixl-cu13 1.3.1` *is* installed) | `pip list` |
| `lmcache` on PyPI | 0.5.2 latest, **sdist only — no aarch64 wheel**; 34 `.cu`/`.cpp` sources | `pip download --no-deps` |

Registered KV connectors in this fork (`KVConnectorFactory._registry`):

```
DecodeBenchConnector, ExampleConnector, ExampleHiddenStatesConnector,
FlexKVConnectorV1, HF3FSKVConnector, LMCacheConnectorV1, LMCacheMPConnector,
MoRIIOConnector, MooncakeConnector, MooncakeStoreConnector, MultiConnector,
NixlConnector, NixlPullConnector, NixlPushConnector, OffloadingConnector,
SimpleCPUOffloadConnector
```

From the checkpoint's `config.json`:

| Field | Value |
|---|---|
| `max_position_embeddings` | **1,048,576** |
| `rope_scaling` | YaRN, `factor: 16`, `original_max_position_embeddings: 65536` |
| `num_hidden_layers` | 43 |
| `index_topk` | 512 (DeepSeek sparse attention) |
| `dspark_block_size` | 5 |

Live engine args worth noting: `max_num_batched_tokens: 8192`,
`async_scheduling: True`, `block_size: 256`, `enable_prefix_caching: True`.

### The derived constant everything hangs off

```
14.22 GiB / 869,357 tokens = 17,564 bytes/token ≈ 17.2 KiB/token
```

That is the padded NVFP4 MLA KV cost per token, per rank. Every number
below is this constant times a context length.

---

## 2. Why LMCache cannot raise `--max-model-len`

A request that is actively decoding must have **every one of its KV tokens
resident in the GPU paged pool**. DeepSeek's sparse attention does not
change this: `index_topk: 512` means each query attends to 512 selected
tokens, but the selector chooses them *from the entire context*, so the
entire context must be there to select from.

LMCache is a **reuse cache between requests** — it stores the KV of
finished prefixes so a later request can load instead of re-prefill. It is
not a paging system that streams a live request's KV in and out per layer.

So the ceiling on `--max-model-len` is pure arithmetic and LMCache is not a
term in it:

| `--max-model-len` | KV for ONE request | Fits in 14.22 GiB? | Concurrency multiple |
|---|---|---|---|
| 256K (today) | 4.29 GiB | yes | **3.32×** |
| **512K** | **8.57 GiB** | **yes — free today** | 1.66× |
| **768K** | **12.86 GiB** | **yes — free today** | 1.11× |
| 1M | 17.14 GiB | **no — short ~2.92 GiB** | needs util ≈ 0.82 |

(The 256K row reproduces the documented 3.32× exactly, which is the check
that the 17.2 KiB/token constant is right.)

**Consequence: 768K context is available right now for the price of one
env var** — `MAX_LEN=786432 ./spin-up-vllm-dspark-2box.sh` — at unchanged
util 0.80, no new software, no host-memory risk. The script already
documents `MAX_LEN` as a knob (line 125, "after checking KV pool"); this is
that check.

**Do not act on that before reading §3.** The concurrency column above is
not spare capacity — it is how many agents can hold a cached prefix, and
768K spends all of it on one.

1M needs util 0.80 → ~0.82, which takes spark1 from 8 GiB available to
~5 GiB. That is squarely in `feedback_gpu_util_080_default` wedge
territory. **Not recommended without accepting reboot risk.**

---

## 3. The failure mode this actually creates: agent ping-pong

The concurrency column in §2 is not an abstract ratio. It is **how many
agents can keep their prefix cached at once**, and it is the thing that
will bite.

The threshold is just `pool ÷ N`:

| Concurrent agents | Largest context where ALL stay cached |
|---|---|
| 1 | ~849K |
| **2** | **~424K** |
| **3** | **~283K** |
| 4 | ~212K |
| 6 (`max_num_seqs`) | ~141K |

**This retroactively explains the current config.** 256K at 3.32× is almost
exactly sized for **3 concurrent agents** — which is also the load the
2026-08-03 throughput measurement was taken at (~100+ tok/s aggregate at 3
streams). Whether that sizing was deliberate or lucky, it is why the box
works today.

### Read it the other way: today's workload is fine

The cliff below needs contexts that are **large relative to the pool**.
768K is 88% of it, so two agents cannot coexist. 100K is 11.5% of it, and
nothing bad happens:

| 6 agents, each at | Total | vs 869,357-token pool |
|---|---|---|
| **100K** | 600K | **fits — 31% headroom, ~8.7 agents would fit** |
| 141K | 847K | exactly at the edge |
| 150K | 900K | over — eviction starts |
| 200K | 1.2M | 38% over — ping-pong |

**At ~100K per agent the current box is correctly sized and there is no
re-prefill problem.** Two costs at that size are real but are *not*
eviction, and should not be mistaken for it:

- **Cold burst.** Six 100K requests arriving together = 600K tokens of
  prefill at ~1,045 tok/s ≈ **10 min aggregate** before the last one
  generates. That is compute, not cache — no cache fixes the first send.
  It is also why the 2026-07-04 Cognee batch measured ~0% APC hit rate: a
  cold simultaneous burst has nothing to hit.
- **Distinct contexts in rotation, not `max_num_seqs`.** The divisor is how
  many *different* prefixes compete, not how many run at once. Six slots
  with twelve long-lived agents cycling through them is twelve prefixes
  against 869K → ~72K each before eviction.

### Why crossing the threshold is a cliff, not a slope

Agent loops **re-send their entire context every turn**. So once the pool
can hold fewer prefixes than there are active agents:

1. Agent A runs at 768K, prefix cached.
2. Agent B runs, evicting A's prefix — the pool holds 1.11 contexts, there
   is no room for both.
3. A's *next turn* arrives, prefix gone → **full ~12.5 min cold prefill**.
4. A's prefill evicts B. B's next turn → another ~12.5 min.

They mutually evict, and **every turn pays full cold prefill** — not one
bad turn, all of them, indefinitely.

Note what does *not* happen: vLLM does not thrash mid-generation. Its FCFS
scheduler preempts from the back of the running queue, so a newly arrived B
waits for A rather than fighting it. *(Reasoned from vLLM V1 scheduler
behaviour — not measured on this deployment.)* The damage is entirely in
the **cache**, between turns, which is why it presents as a 12-minute stall
rather than an error, a warning, or anything in the logs that says
"eviction".

### The trap in raising `MAX_LEN`

Going 256K → 768K silently converts this from a 3-agent box into a
**1-agent box**. Nothing in the config says so: `max_num_seqs` is still 6,
`/v1/models` still answers, no flag changed meaning. The only symptom is
that the second agent's second turn takes twelve minutes.

**So the §2 "free win" is only free for a single-agent workload.** For
multi-agent use the honest ceiling is the table above — ~424K for two,
~283K for three — unless the eviction is caught by a disk tier, which is
§4.

---

## 4. What LMCache actually buys — the case for doing it anyway

Cold prefill on this deployment measured **flat and linear at ~1,045 tok/s**
out to 176K (no long-context cliff — see the 2026-07-30 measurements). So
raising the context has a brutal cold cost:

| Context | Cold prefill at 1,045 tok/s |
|---|---|
| 256K | ~4.2 min |
| 512K | ~8.4 min |
| 768K | **~12.5 min** |
| 1M | **~16.7 min** |

APC is already on and delivered **17.9× at 176K**. But per §3, at 768K the
pool holds **1.11 contexts** and at 1M less than one. **APC stops working
exactly where it becomes most valuable** — the cache is evicted by the very
next request, so you pay the full 12–17 min again, every turn.

This is the gap LMCache's **disk tier** fills:

| | Value |
|---|---|
| Disk capacity for KV | 2.2 TB ÷ 17.2 KiB = **~125 million tokens** |
| — expressed as contexts | **~477 × 256K**, or **~119 × 1M** |
| 1M-token context on disk | 18.4 GB → **~1.8 s at raw 10.1 GB/s** |
| Realistic through connector + H2D + reshape | est. **6–10 s** (3–5× off raw) |
| Versus recomputing it | **16.7 min** |
| Survives container restart | yes (unlike APC) |

So the honest framing is not *"LMCache gives me more context"* — it is
**"LMCache is what makes 768K–1M context usable more than once."** That is
precisely the read-heavy, load-context-once-and-live-in-it shape this box
is used for (`user_workflow_read_heavy`), and it is the direct fix for the
agent ping-pong in §3: an evicted prefix lands on disk instead of
vanishing, so agent A's next turn costs an estimated 6–10 s instead of
12.5 min, and the agent count stops mattering.

**Honest limit: LMCache does not make big agents concurrent.** Two 768K
agents still serialize — in-flight KV must be GPU-resident, and the pool
holds one. LMCache makes the *ping-pong* cheap; it does not make the
*parallelism* possible. Running two large agents genuinely at once is a
KV-pool problem, and the only lever is `gpu_memory_utilization`, with the
wedge risk in §2.

### The CPU tier is near-worthless on GB10 — do not bother with it

Unified memory means "host RAM" and "GPU memory" are the **same 121 GiB of
LPDDR5X**. Offloading KV to CPU moves the accounting, not the bytes: an
N-byte CPU-side KV buffer and an N-byte GPU-side KV buffer consume
identical physical memory, and only the GPU-side one can be attended to
directly. On a discrete-GPU box CPU offload is the main event; here it is
strictly a loss.

**Disk is the only tier on this hardware that adds real capacity.** Config
LMCache accordingly (`local_disk` on the NVMe, `local_cpu` disabled or
token-sized).

---

## 5. The second model — arithmetic, and the conflict

79.21 GiB of weights per box out of 121 GiB is the wall. Any memory for a
second model comes **straight out of the 14.22 GiB KV pool**:

| `gpu_memory_utilization` | Freed for a 2nd model | KV pool left | Max single-request context |
|---|---|---|---|
| 0.80 (today) | — | 14.22 GiB | 848K |
| 0.75 | ~6 GiB | 8.2 GiB | ~489K |
| 0.70 | ~12 GiB | 2.2 GiB | ~131K (256K breaks) |

**Budget: ~6 GiB before 256K starts hurting, ~10 GiB before it is
impossible.** That admits:

- ✅ Qwen3-4B-FP8 (~4.3 GiB + small KV)
- ❌ any 8B (~11 GiB with KV — breaks 256K)
- ❌ Qwen3-Coder-30B-A3B (~32 GiB), anything 14B+

**The two goals are in direct conflict.** Every GiB given to a second model
is a GiB unavailable for context. You cannot both raise context to 768K–1M
and free space for a resident second model.

### The cheap way out

Ollama is already running on `:11434` on both boxes with lazy-load and a
5-minute idle unload. It holds **zero static reservation**. A 4B at Q4
(~2.5 GiB) fits inside the current 8 GiB of free host RAM while resident
and gives it back afterwards. If "another model" means *available*, not
*hot*, this is the answer and it costs nothing.

### The interesting composition (if LMCache works)

If disk-backed KV makes re-prefill cheap, the GPU pool only needs to hold
**in-flight** KV rather than the reuse cache. For a 1–2 concurrent-stream
workload you could drop util, free real memory for a resident second model,
and let disk absorb the reuse. That partially reconciles the two goals —
but it trades away max context, so it is a third option, not a free lunch.

---

## 6. Blockers, ranked

**1. `nvfp4_ds_mla` padded KV layout — make-or-break.**
The boot log says *"Using DeepSeek V4 padded nvfp4_ds_mla KV cache format."*
That is a fork-specific layout. Upstream LMCache 0.5.2 handles standard
layouts and MLA, and has fp8 paths, but this padded NVFP4 MLA variant is
not something upstream has any reason to know about. Failure modes range
from a clean exception (fine) to **silent KV corruption** (bad — it would
present as subtle incoherence, not a crash). *Not tested. Reasoned from the
log line and from what upstream supports.*

Falling back to a supported KV dtype is **not** an acceptable mitigation:
fp8 would roughly double bytes/token, halving the pool to ~430K tokens —
that costs you the exact context you were trying to buy.

**2. No aarch64 wheel.** `pip download lmcache` yields
`lmcache-0.5.2.tar.gz` only, containing 34 CUDA/C++ sources
(`mem_kernels.cu`, `cachegen_kernels.cuh`, `csrc/storage_backends/fs/`,
etc.). `nvcc 13.0.88` *is* in the image, so a source build targeting sm_121
is plausible — but it must be baked into a derived image or the container
stops being reproducible. Budget a day, not an hour.

**3. `async_scheduling: True` is live.** KV connectors in vLLM frequently
require async scheduling disabled. Expect to pay that back in throughput,
which matters because this deployment is explicitly run in throughput mode
(~100+ tok/s aggregate at 3 concurrent streams).

**4. TP=2 across two boxes, `mp` backend.** Each rank needs its own store
and disk path; spark2's worker is `--headless` with no API. MLA KV is
replicated across ranks, so a naive setup writes two identical copies —
2× disk traffic for no benefit.

**5. Everything requires a full endpoint restart** (~5.5 min boot with warm
page cache), and the whole cross-box endpoint is down for the duration.
There is no second chat endpoint to fall back to while DSpark is up.

### The good news

`LMCacheConnectorV1` **is** registered in this fork, alongside vLLM-native
`OffloadingConnector` and `SimpleCPUOffloadConnector`. Those last two do
opaque block copies and are far more likely to be layout-agnostic — which
makes them a **cheap proxy test for "can KV leave the GPU at all in this
build"** before spending a day on an LMCache source build.

---

## 7. Suggested staging

**Step 1 — context bump, ~10 min, no new software. Free *only* for
single-agent use — read §3 first.**
Re-spin at `MAX_LEN=786432 ./spin-up-vllm-dspark-2box.sh`. Confirm
`GPU KV cache size` lands near 869K tokens (it should be unchanged — the
pool is fixed by util, only the concurrency multiple moves). Run a needle
test at 700K. This delivers the context increase immediately *and* produces
the ~12.5 min cold-prefill baseline that the LMCache A/B needs.

**It also silently converts a 3-agent box into a 1-agent box.** If the
current multi-agent use matters more than the context, stop at **~424K**
(2 agents cached) or **~283K** (3 agents cached) instead of 768K — see the
threshold table in §3. Going to 768K before LMCache exists is a deliberate
choice to trade concurrent agents for depth, not a pure win.

**Step 2 — go/no-go gate, ~1 hr.**
Keep 256K. Enable the native `OffloadingConnector` via `--kv-transfer-config`
and check whether nvfp4_ds_mla blocks round-trip at all. Verify coherence
on a cache hit, not just absence of a crash — silent corruption is the
failure mode that matters. **If this fails, LMCache almost certainly fails
too, and you have saved the build.**

**Step 3 — the real experiment.**
Derived image with LMCache 0.5.2 built for aarch64 / CUDA 13 / sm_121,
`local_disk` backend on the NVMe, `local_cpu` off. A/B cold vs disk-hit
TTFT at 256K and 768K, plus a coherence check on every hit. Measure the
async-scheduling cost separately so it isn't confounded with the LMCache win.

Step 2 is the honest gate. Everything after it is contingent.

---

## 8. Open items not addressed here

- `max_num_batched_tokens` is still the recipe's **8192**, not the 40960
  our Qwen3-Next builds use. vLLM warns at boot that it clips
  `max_num_scheduled_tokens` to 8180 for draft slots. This is an
  independent prefill lever, still untried, and it interacts with the
  cold-prefill numbers in §4 — worth isolating before or after, not during,
  the LMCache work.
- `RestartPolicy=no` on both DSpark containers is still open
  (`current-setup.md` §5).
- The four clients still pointed at the old Qwen id are still broken.
