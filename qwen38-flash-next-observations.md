# Qwen3.8-Flash-Next (NVFP4 + fp8 hybrid) on a single Spark — observations

Append-only experiment log. Newest section at the bottom.

Live config, revert commands and the client-repoint list live in
`current-setup.md`; how the model wants to be *called* lives in
`dgxlib/models.yaml`. This file is the measurement + reasoning record.

---

## 2026-09-10 — deployed to spark1, replacing the cross-box DSpark pair

### What ran

`qwen3.8-flash-next` — Qwen3.8-Flash-Next (~176B total: 125B main + 51B
n-gram, 6B active), RadixArk NVFP4 routed experts plus blockwise-fp8 side
layers (`MODE=hybrid`), single box, TP=1, 262K native context, util 0.80,
seqs 8, MTP-2, APC on, bf16 KV. Container `qwen38-flash` on spark1:8001,
`--restart unless-stopped`.

Brought up with `spin-up-vllm-qwen38-flash-next.sh`, a thin wrapper over the
**blazux recipe** (`github.com/blazux/qwen3.8-Flash-DGX`, commit `bd60fcb1`,
2026-09-09) cloned on the box at `~/qwen3.8-Flash-DGX`.

Taking a whole box for a single-box model meant tearing down the cross-box
DSpark pair, which consumed both. spark2 was restored to
`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` (MTP-2 + APC, seqs 8) as the fallback
and A/B partner.

### Choosing between two community recipes

Two recipes exist for this model on GB10. They are not equivalent.

| | `krisitown/qwen38-flash-next-nvfp4-dgx-spark` | `blazux/qwen3.8-Flash-DGX` (chosen) |
|---|---|---|
| Last updated | 2026-08-28 | 2026-09-08 |
| Checkpoint | kstoyanov99, 186.5 GB HF + a 51 GB local `cat` = ~238 GB disk | RadixArk, 126 GiB, ~140 GB disk |
| `gpu-memory-utilization` | **0.9** | **0.80** |
| Prefix caching | offered as an option, on an image WITHOUT the Mamba block-size fix | fixed and on by default |
| Determinism at T=0 | not addressed | fixed (deterministic QSA top-k kernel) |
| Context | 240000 | 262144 native, 500000 with YaRN (needle-validated at 414K) |
| Defaults chosen by | throughput benchmarks | a 17-scenario agentic tournament, 3 repeats, one variable at a time |
| Independently reproduced | no | yes, on another Spark (@jschmied) |

Two things decided it.

**1. The prefix-caching landmine.** krisitown's `config/vllm.env` lists
`#ENABLE_PREFIX_CACHING=true` as a suggested option. Given
`reference_apc_hybrid_qwen3_next` — APC was a ~10× TTFT win on Qwen3-Next —
we would certainly have turned it on. blazux traced what that does on an
unpatched image: vLLM's EngineCore overwrites `cache_config.block_size` with
the *smallest* KV-group block size (8 tokens at MTP=2, the QSA raw-key ring)
while the Mamba state block is 1600. Two call sites used the former as the
latter, so a prefix hit computed the state slot as `(3200-1)//8 = 399`
instead of `1`, read past the block-table row, and restored an **all-zero
Mamba state**.

No crash. No warning. Silently different answers on cache hits — and
completely invisible to a coherence smoke test, which is the same shape as
the DSpark missing-tool-parser bug (2026-07-30) and the `nano_v3` reasoning
leak. The failure class this repo keeps re-encountering is *a config that
looks healthy and is quietly wrong*, and a coherence check does not catch any
of them.

Confirmed in our own boot log that the fix is live:

```
Setting attention block size to 1600 tokens to ensure that attention page
size is >= mamba page size
Mamba cache mode is set to 'align' ... when prefix caching is enabled
```

1600, not 8. **Do not enable prefix caching for this model on any other
qwen38-flash-next image without checking for that fix first.**

**2. `GPU_MEM=0.80` arrived at independently.** blazux ships 0.80 and
documents 0.85 drifting into swap after a day and 0.875 being OOM-killed on a
300K prefill. That is the same unified-memory conclusion as
`feedback_gpu_util_080_default` (0.88 wedged spark2 into a physical reboot)
reached from a different direction — and on this model it is sharper, because
the host page cache is not spare capacity here, it is the storage tier for a
47.7 GiB lookup table.

### The one idea that makes it fit

A 126 GiB checkpoint does not fit beside a usable KV cache in 128 GB of
unified memory. 47.7 GiB of it is the PLE n-gram embedding table — a pure
lookup where a token touches only 16 rows × 160 B ≈ 2.5 KB. The recipe mmaps
it from NVMe and gathers rows on demand:

```
PLE mmap: layer 1, 128 shards, 320001536 rows x 160 B (47.7 GiB on disk),
dtype F8_E4M3, 32 workers
```

Weights on card drop to **76.75 GiB** (blazux measured 77.83 GiB — close
match), and the rest of the pool goes to KV.

The consequence to internalise: **prefill speed is now a function of page-cache
residency**, which is a serving characteristic this repo has not had before.

### Measured (spark1, single stream, greedy)

Prefill measured with unique-nonce prompts so APC cannot fake a cold number
(the method from the DSpark work).

| | measured |
|---|---|
| Boot, cold | **14.5 min** (weights 628s + draft 80s, compile 34s, KV alloc) |
| Weights on card | 76.75 GiB |
| GPU KV pool | **553,254 tok → 2.11× @ 262K** |
| Prefill, cold page cache | **~1,137 tok/s** |
| Prefill, warm, ~10.5K | 2,100 → 2,417 → 2,452 → **2,494 tok/s** (warming across runs) |
| Prefill, warm, ~46K | **2,408 / 2,410 tok/s** — flat, no long-context cliff |
| Decode, steady state | **30.6 / 37.2 tok/s** (900-token generations) |
| APC on a re-sent ~10.6K prefix | 9.38s → 1.33s = **7.1× TTFT** |
| Determinism at T=0 | first-token logprobs identical across runs |
| Host available | ~14–17 GB |

vLLM's own counters during those runs read 4,584–4,625 tok/s prompt
throughput and 32.0–34.7 tok/s generation — higher than the end-to-end
figures, which include HTTP, tokenisation and queueing.

The APC hit was proven with `vllm:prefix_cache_hits_total` (0 → 8000), not a
stopwatch. That matters here specifically: with the table on NVMe, a warm
page cache *alone* speeds up a second identical pass, so wall-clock cannot
separate the two effects.

### Versus the DSpark it replaced

| | DSpark 2-box | Qwen3.8-Flash-Next 1-box |
|---|---|---|
| Boxes consumed | **2** | **1** |
| Prefill, cold | ~1,000–1,180 tok/s | ~1,137 tok/s — **parity** |
| Prefill, warm | (n/a — no equivalent tier) | **2,400–2,500 tok/s** |
| Prefill @ 44K | 1,177 tok/s | **2,410 tok/s (2.05×)** |
| Decode, single stream | ~21–31 tok/s | **30.6–37.2 tok/s** |
| Context | 256K | 262K |
| Survives a reboot | **no** (`--restart` never added) | **yes** |

**Honest reading of the prefill win: it is conditional.** Cold-cache first
pass is parity. The ~2× only appears once the PLE table's hot set is resident,
which it will be for a box in steady use but is not right after a restart or
after another large container evicts the cache. `PREWARM=1` streams the table
once at boot (~10s) if first-request latency after a restart matters; not
enabled.

Not measured, and the honest gap in this comparison: **aggregate throughput
under concurrency.** DSpark's keep decision rested on ~100+ tok/s at 3
concurrent streams, and this box has only been measured single-stream. The
recipe cites another Spark reaching 266.8 tok/s aggregate at 48 streams, but
that was a different config (no spec decode, 8K context) on someone else's
hardware. Until that is run here, the DSpark comparison is incomplete on the
axis DSpark was actually kept for.

### Verified working

- **Tool calling**: a real `tools`-bearing request returned
  `finish_reason: "tool_calls"` with a structured array and valid JSON
  arguments `{"location": "Paris"}` — not syntax leaked as plain text (the
  wrong-parser signature). Parser `qwen3_coder`.
- **Ollama embeddings untouched**: `qwen3-embedding:0.6b` on :11434 returned
  1024-dim vectors throughout, so the MemPalace path never broke.

### Gotchas found here

- **`reasoning`, not `reasoning_content` — and thinking is ON by default.**
  This build populates `reasoning` and leaves `reasoning_content` null: the
  same shape as `nano_v3` and the DeepSeek-R1 parser
  (`todo_nano_v3_reasoning_leak`). Confirmed by reading openclaw's own bundle
  — it looks up `reasoning_content` in 5 places and never `reasoning`, so the
  trace is dropped *and* billed. Measured on a trivial "Reply with exactly:
  OK": **24 of 28 completion tokens were trace (86%)**.

  | request | completion tok | reasoning tok | content |
  |---|---|---|---|
  | default | 28 | 24 | `'\n\nOK'` |
  | `chat_template_kwargs: {"enable_thinking": false}` | **2** | **0** | `'OK'` |
  | `/no_think` in the prompt | 41 | 33 | `'OK /no_think'` |

  **`/no_think` does not work on this model** — it is echoed into the answer
  verbatim and the trace fires anyway. Worth knowing because the upstream
  recipe's `smoke-test.sh` uses `/no_think` in its decode prompt, which means
  **its published decode figures are measured with thinking ON** (and with a
  stray `/no_think` in the output). Our own decode numbers above share that
  caveat; a thinking-off decode measurement is still to be taken.

  `models.yaml` sets `can_think: true, thinking_default: false`, which emits
  the working kwarg through `extra_body`.
- **The lowercase served id defeats the registry's prefix match.**
  `qwen3.8-flash-next` does not match the `Qwen/Qwen3` entry under `match:`
  because `registry.py:93` uses case-sensitive `startswith`. Without an
  explicit entry it resolved to `default` — `idle_timeout: 120` — which would
  kill long-context calls mid-prefill, since a 262K cold prompt is minutes of
  silence before the first token. Caught with `resolve_model_config()` before
  deploying, not after.
- **`HF_TOKEN` is invisible to `ssh spark 'cmd'`.** It lives in `~/.bashrc`
  behind the `case $- in *i*) ... return` non-interactive guard, so even an
  explicit `source ~/.bashrc` does not expose it — the
  `feedback_profile_export_keyword` shape. Extract it inside the remote shell.
- **The `Unknown vLLM environment variable` warnings are benign.**
  `VLLM_PLE_MMAP*`, `VLLM_QSA_DET_LIB`, `VLLM_MTP_DRAFT_VOCAB`,
  `VLLM_FP8_PAD_M4` are read directly by the image's patches, not by vLLM's
  env registry. Unlike the DSpark case, these warnings do *not* indicate a
  flag that is being ignored.
- **Two load passes at boot is normal**, not a crash loop: the target model
  (628s) then the MTP draft (80s). The shard progress bar restarting at 0
  looks alarming; check `docker inspect -f '{{.RestartCount}}'` before
  concluding anything.

### Known, not tuned

- **Multi-client prefill stalls.** With two or more agents live, a *decoding*
  client drops to ~0.2 tok/s for a minute or two while another client prefills
  a cold long prompt. Structural to vLLM chunked prefill — every step carrying
  a prefill chunk carries exactly one token for each decoder — not a recipe
  bug and not fixable, only tradeable: `EXTRA='--long-prefill-token-threshold
  1024'` lifts the stalled client to ~1.0 tok/s at the cost of ~36% of an 8K
  TTFT. Left off for the single-main-user case.
- **`vm.swappiness` is 60** (the Spark default); the recipe recommends 10 for
  multi-agent use. Not changed.
- **`MODE=hybrid-mtp`** (NVFP4 draft experts grafted on) would give ~+22% KV
  pool for no measured decode change. Not tried.
- **YaRN to 500K** is validated upstream (needle at 414K). Not tried — 262K
  native matched what DSpark served, which kept rope scaling out of the
  bring-up.

### Verdict

**Not yet judged.** The infrastructure is verified and the single-stream
numbers are good, but subjective quality against the DSpark bar — which is
what actually decided DSpark's ADOPTED status on 2026-08-03 — has not been
assessed, and neither has concurrent throughput. spark2 holds the known-good
Qwen3-Next-80B as both fallback and A/B partner.

---

## 2026-09-10 (later) — spark2 moved onto the same model; stocked from spark1 over the cable

### What changed

spark2 swapped `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` → `qwen3.8-flash-next`.
Both Sparks now run the same model, single-box TP=1 on each, same image, same
flags, byte-identical checkpoint. Container `qwen38-flash` on both, port 8001.

This is not a new model experiment — the model was already judged (or rather,
not yet judged) that morning on spark1. What is new here is the *stocking*
procedure and what it says about the second box.

### Copying beats downloading, by a wider margin than expected

spark2 had none of the prerequisites: no recipe clone, no `qwen38-flash-dgx`
image, no checkpoint. The documented path (§8 of `current-setup.md` at the time)
was "download all 126 GiB again" plus a second `prepare-hybrid.sh` run.

Copying spark1 → spark2 over the direct 10.100.16.x cable instead:

| leg | size | time | rate |
|---|---:|---:|---:|
| recipe repo | ~MB | instant | — |
| docker image `qwen38-flash-dgx` | 20.7 GB | 3m48s | ~91 MB/s |
| HF checkpoint (incl. prepared `-fp8hybrid`) | 139 GB | 4m51s | ~478 MB/s |

Sustained ~425 MB/s measured on the wire mid-transfer. **That is ssh-cipher
bound, not cable bound** — the cable benchmarks at ~110 Gb/s/port (~13 GB/s), so
we used about 3% of it. `aes128-gcm@openssh.com` on a single stream is the
ceiling here. Unencrypted `nc` or parallel streams would go faster; 5 minutes
did not justify the complexity.

Three reasons this is the right default, in order of importance:

1. **Identity.** spark2 runs the bytes that were validated on spark1 that
   morning, not a separately re-derived checkpoint. `prepare-hybrid.sh` rewrites
   ~15 GiB of side layers to blockwise fp8; running it twice is two chances to
   differ.
2. **The image, especially.** Ten patches are compiled into
   `qwen38-flash-dgx`, one of which is the Mamba block-size fix that makes
   prefix caching *not silently corrupt state*. `docker save | docker load`
   moves the artifact. `docker build` on the second box is a chance for the
   upstream recipe, base image or a pinned dependency to have moved.
3. **Time.** ~9 min of transfer against ~35-50 min of download + prepare.

### The gotcha: one file in 139 GB needs root, and tar hides it

`hub/models--…/trees/<rev>.json` is mode **600, root-owned** — the HF cache is
written by the download *container*, so it lands as root, and this one file is
not world-readable. `tar` running as `kostadis` cannot open it.

What makes it a trap rather than an error:

- `tar` skips the file, keeps going, and reports failure only at the **end**.
- In `tar … | ssh … 'docker run … tar -x'` without `pipefail`, the pipeline's
  exit status is *ssh's*, not tar's. The wrapper script exited **0**.
- The stderr line scrolls past in the middle of a 5-minute transfer.

So the copy looks clean and is short one file. It was caught by comparing
`du -sb` between boxes: a **94,865-byte** gap, exactly this file's size. Copied
afterwards through a container (docker = root read).

In this case it was cosmetic — `trees/` is xet dedup metadata and is never read
when vLLM is pointed at an explicit snapshot path with `HF_HUB_OFFLINE=1`. The
failure *mode* is not cosmetic: a partial copy that reports success.

**Verification that actually settles it:** parallel `md5sum` over all 428 files
on both boxes, sorted, hashed —
`671788c9611b59321593c21741e16f42` on spark1 and on spark2. A matching `du` was
not treated as sufficient, and shouldn't be.

### spark2 measurements vs spark1

Same profile (`MODE=hybrid`, 262K, util 0.80, seqs 8, MTP-2, APC on, `DET_TOPK=1`,
`DRAFT_VOCAB=1`, bf16 KV). Prefill first, per `user_workflow_read_heavy`;
unique-nonce prompts so APC cannot fake a cold number.

| | spark1 (morning) | spark2 (evening) |
|---|---|---|
| Weights on card | 76.75 GiB | **76.75 GiB** (exact) |
| KV pool @ 262K | 553,254 tok → 2.11× | **576,427 tok → 2.20×** |
| Boot to healthy | 14.5 min | **14.4 min** |
| Cold prefill | ~1,137 tok/s | **1,239 tok/s** |
| Warm prefill ~8-10K | 2,417 / 2,452 / 2,494 | **2,256 / 2,457** |
| Warm prefill ~36-46K | 2,408 / 2,410 | **2,442 / 2,450** |
| Decode (900 tok, greedy) | 30.6 / 37.2 tok/s | **32.0 / 43.4 tok/s** |
| APC TTFT win | 9.38s → 1.33s = 7.1× | **8.61s → 1.14s = 7.6×** |
| Determinism (`DET_TOPK=1`) | identical logprobs | **identical logprobs** |

Everything is within run-to-run noise. The small KV-pool edge on spark2
(+4.2%) is host-memory variance at profile time, not a config difference.

Warm prefill is flat from 8K to 36K on this box too — no long-context cliff.

### Re-verified on the new box rather than assumed

- **Prefix-caching landmine.** APC is only safe on an image carrying the Mamba
  block-size fix. spark2's own boot log: `Setting attention block size to 1600
  tokens…` — **1600, not 8** — plus `Mamba cache mode is set to 'align'`. The
  check costs one `grep` and the failure it catches is invisible to a coherence
  smoke test. Worth running on every box and every image rebuild.
- **Tool calling.** Real `tools`-bearing request → `finish_reason: "tool_calls"`,
  structured array, `{"location": "Paris"}`, `content` empty. No syntax leaked
  as text.
- **Reasoning-field shape.** Same trap as spark1: thinking ON by default (28
  completion tokens for "Reply with exactly: OK"), trace in **`reasoning`**,
  `reasoning_content` **null** — the shape opencode and openclaw silently drop.
  `chat_template_kwargs: {"enable_thinking": false}` → 2 tokens, no reasoning.
- **Ollama :11434 on spark2** — live 1024-dim vector after the swap; the
  MemPalace embedding path is unaffected.

### What was NOT measured, and why

**Cross-box output equality** — same greedy prompt to both boxes, compare text.
Attempted and abandoned as uninformative: spark1 was saturated at the time (8
running, 31 deferred) while spark2 was idle, and vLLM's numerics depend on batch
composition, so a divergence would not have implied a bad copy. The md5 tree
match is stronger evidence and it is exact. Worth doing properly when both boxes
are quiet, as a check on the *determinism kernel*, not on the copy.

### What this cost

**The A/B partner.** Until this swap spark2 held the known-good Qwen3-Next-80B —
a different local model to compare against, and a working endpoint for any
client still pinned to the old id. Both are gone. Any client sending
`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` to `192.168.1.121:8001` now 400s.

The revert was verified *before* the swap, not assumed: the 77 GB Qwen3-Next
checkpoint and `~/spin-up-vllm-qwen3-next-80b-mtp.sh` are both still on spark2,
so restoring it is a load (~15 min), not a download.

### Incidental finding

During verification spark1 was at **8 running / 31 deferred** requests — the
multi-client contention the morning's entry filed under "known, not tuned"
(`--long-prefill-token-threshold`). A second identical endpoint is somewhere for
that overflow to go, which is an argument for this swap that has nothing to do
with the model.

### Verdict

Infra verified on both boxes and matching. The swap changed how many boxes serve
the model, not the model — so the open question from the morning entry is
unchanged: **subjective quality against the DSpark bar is still not judged**, and
there is now no second local model to judge it against.
