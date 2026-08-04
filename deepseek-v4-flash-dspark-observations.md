# DeepSeek-V4-Flash-DSpark on 2x DGX Spark — observations

Append-only experiment log, per this repo's convention. Plan/runbook:
`deepseek-v4-flash-dspark-2box-plan.md`. Spin-up:
`spin-up-vllm-dspark-2box.sh`.

---

## 2026-07-30 — first bring-up (TP=2, `mp` backend, no Ray)

### Config actually served

| | |
|---|---|
| Served id | `deepseek-ai/DeepSeek-V4-Flash-DSpark` |
| Image | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (`sha256:a8394849…`, 18.8 GB) |
| vLLM | `0.25.2.dev0+g752a3a504.d20260714` (torch `2.11.0+cu130`) |
| Layout | spark1 = head (`--node-rank 0`, API on :8001); spark2 = worker (`--node-rank 1 --headless`) |
| Parallelism | TP=2, `--distributed-executor-backend mp`, `--nnodes 2`, no Ray |
| Rendezvous | `--master-addr 10.100.16.1 --master-port 25440` over the DAC cable, TCP sockets (no RoCE) |
| Context | 262144 (**not** the recipe's 1M — see below) |
| util / seqs | 0.80 / 6 |
| KV | `nvfp4_ds_mla`, `--block-size 256` |
| MoE | `flashinfer_b12x` |
| Spec decode | `{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}` |
| APC | **on by default in this build** (`enable_prefix_caching=True`, not passed explicitly) |
| Quant | `quantization=deepseek_v4_fp8`, `scale_fmt=ue8m0` → UE8M0 DeepGEMM enabled |
| Arch resolved | `DeepseekV4ForCausalLM` + draft `DeepSeekV4MTPModel` (96 params) |

Boot: **~5.5 min** container start → `Application startup complete`
(00:34:32 → ~00:40). Much faster than the ~17 min Qwen3-Next reloads —
the weights were warm in page cache.

### Boot signature — the numbers that matter

```
GPU KV cache size: 859,040 tokens
Maximum concurrency for 262,144 tokens per request: 3.28x
```

**The recipe's expectation was ~1.9–2.04M tokens. We got 859K — under half.**
Consequence, and this vindicates starting at 256K instead of 1M:

> 859,040 KV tokens ÷ 1,048,576 = **0.82×**. A 1M context does **not fit** at
> util 0.80. Following the recipe's `--max-model-len 1048576` verbatim would
> have been a boot failure, not a slow start.

Per `feedback_size_context_by_kv_pool`, that's the whole point of reading the
pool before trusting a published context setting. If 1M is actually wanted it
needs util ≥0.85 — which collides with `feedback_gpu_util_080_default` (see
the host-memory risk below), so 1M on 2 boxes looks like it needs the
0.85-and-accept-the-wedge-risk trade.

### Measured — prefill first (`user_workflow_read_heavy`)

Measured with unique-nonce prompts so APC could not silently turn a "cold"
number into a cache hit; the warm probe re-sends a byte-identical prompt.

Note the target labels undershot: the filler produced ~1.85 tokens/word, so
the "8K/32K/128K" probes were really **11K / 44K / 176K** tokens. Actual
counts below are what matter.

| Probe | Prompt tokens | Cold TTFT | Cold prefill tok/s | Warm TTFT | APC speedup |
|---|---|---|---|---|---|
| ~8K | 11,100 | 11.04 s | 1,005 | 8.46 s | 1.3× |
| ~32K | 43,999 | 37.39 s | 1,177 | 6.53 s | **5.7×** |
| ~128K | 176,428 | 168.83 s | 1,045 | **9.45 s** | **17.9×** |

**Two genuinely good results here.**

**1. Cold prefill is LINEAR, ~1,000–1,180 tok/s, flat across 11K→176K.**
No quadratic blowup. This directly contradicts the alarming 1-box thread
(~13 min at 250K): at a flat 1,045 tok/s, 250K extrapolates to **~4 min, not
13**. The 2-box config does not have the long-context prefill cliff the
single-box thread reported. It also beats the 1-box tuned thread's "~800
prefill" claim.

**2. APC scales with context, and at long context it's transformative.**
17.9× at 176K — a 169-second cold read becomes 9.45 seconds warm. That is
`reference_apc_hybrid_qwen3_next`'s "load context once, live in it" pattern
working even better here than the ~10× measured on Qwen3-Next. For a
read-heavy workflow that re-sends a large stable context, this is the single
most important number on the page, and it substantially softens the
cold-prefill objection from the plan's §6.

The ~8K warm probe improved only ~1.3×, out of line with the other two. Not
explaining that away — it looks anomalous and should be re-run before anyone
leans on the small-context warm figure. The trend across 44K and 176K is
unambiguous, though.

### Measured — decode

| | |
|---|---|
| Single-stream decode | **30.3 tok/s** (600-token generation, warm) |
| Draft acceptance | **~63%** avg; mean accept length **2.89** of 4 |
| Per-position acceptance | 0.857 / 0.619 / 0.419 |

**Decode is ~half the advertised 60–67 tok/s.** The published figure looks
optimistic for this configuration, and the acceptance rate is the likely
reason: 63% is far below the **95–99%** our own Qwen3-Next native MTP-2
achieves (`todo_speculative_decoding`). Position-3 acceptance collapses to
0.42, so the third draft token is mostly wasted compute — `num_speculative_tokens=2`
may well beat 3 here, which is a cheap A/B.

### Tuning lever found (not yet tried)

vLLM warns twice at boot:

```
max_num_scheduled_tokens is set to 8180 based on the speculative decoding
settings. This may lead to suboptimal performance. Consider increasing
max_num_batched_tokens ...
```

The recipe's `--max-num-batched-tokens 8192` is being clipped to 8180 to
make room for draft-token slots. For a read-heavy/prefill-bound workload
that is a small chunked-prefill window — our Qwen3-Next builds run **40960**.
Raising it is the most promising single knob for the prefill axis we
actually care about, and it was not explored on the first bring-up.

### Host memory — real risk, flagged

| | spark1 | spark2 |
|---|---|---|
| Idle right after boot | ~14 GB available | ~15 GB available |
| Under 44K prefill | ~11 GB available | ~12 GB available |
| Under 176K prefill | **~9 GB available** | ~10 GB available |
| At rest *after* the 176K run | **~9 GB available** | ~10 GB available |

Note the last row: headroom did **not** recover after the long-context run
finished. It stays at ~9-10 GB, so 14-15 GB was a fresh-boot figure, not the
steady state. Treat ~9-10 GB as the number to plan against.

This is at or below the headroom that previously **wedged spark2 badly
enough to need a physical reboot** (`feedback_gpu_util_080_default`: 0.88
left ~15 GB and starved sshd's fork). Both boxes stayed responsive to ssh
throughout this session, but util 0.80 with a 156 GB model across 2 boxes is
*not* the comfortable 0.80 we're used to with an 80 GB single-box model.
**Do not raise util to 0.85 for the 1M context without accepting reboot
risk.** If DSpark becomes a keeper, consider 0.75.

### Corrections to the published recipes

Every flag was validated against the image *before* the service was taken
down (`vllm serve --help=all` + reading the installed vllm source). Three
errors found — details and proof in the header of
`spin-up-vllm-dspark-2box.sh`:

1. **"Identical command on both nodes" is wrong.** `--node-rank` must
   differ, and the worker **must** pass `--headless` — the multi-node TP
   worker branch lives inside `run_headless()`, reachable only when
   `api_server_count < 1`, which only `--headless` sets.
2. Node discovery is via `--master-addr` / `--master-port` **CLI flags**,
   not a `MASTER_PORT` env var.
3. **Five of the nine "not optional" env vars do not exist in this image**
   (zero matches anywhere under the vllm package): `WORKER_VLLM_HOST_IP`,
   `VLLM_USE_B12X_WO_PROJECTION`, `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK`,
   `VLLM_DSPARK_REPLICATE_MARKOV_W1`, `VLLM_USE_B12X_FP8_GEMM`. They belong
   to the other recipe's Stage-C build. Corollary: the "never set
   `VLLM_USE_B12X_FP8_GEMM=1`" warning is moot here — nothing reads it.

Confirmed valid in this build: `--kv-cache-dtype nvfp4_ds_mla`,
`--moe-backend flashinfer_b12x` (help text: "for SM12x (RTX Pro 6000 / DGX
Spark)"), spec method `dspark` with `draft_sample_method` in
`Literal["greedy","probabilistic"]`.

One self-inflicted failure worth recording: the `--speculative-config` JSON
must be **single-quoted** for the remote shell. Unquoted, the remote shell
brace-expands `{"method":"dspark","num_speculative_tokens":3,...}` on its
commas and vLLM receives the fragment `method:dspark`:

```
error: argument --speculative-config/-sc: Value method:dspark cannot be
converted to <function loads ...>
```

### Coherence

PASS. No gibberish — the known failure mode did not appear.

```
"The capital of France is Paris.
Now counting from 1 to 10:
1, 2, 3, 4, 5, 6, 7, 8, 9, 10."
```

`reasoning: null`, `finish_reason: stop`. Note the response carries a
`reasoning` field (not `reasoning_content`) — the same shape that silently
drops traces in opencode (`todo_nano_v3_reasoning_leak`). It is null here
because nothing requested thinking; worth re-checking if a thinking mode is
ever enabled.

### Verdict so far — mixed, and the decision hinges on quality

Against the incumbent (spark1's MTP-2 + APC Qwen3-Next-80B):

**In DSpark's favour:**
- **Cold prefill flat at ~1,000–1,180 tok/s out to 176K** — no long-context
  cliff, and better than the published 1-box numbers.
- **APC gives 17.9× at 176K** (169 s → 9.45 s). For read-heavy work that
  re-sends a big stable context, this is the strongest result of the session.

**Against:**
- **Decode 30.3 tok/s** — ~half the advertised 60–67, and below spark1's
  ~49 tok/s MTP Qwen3-Next.
- **Draft acceptance ~63%** vs 95–99% for Qwen3-Next's native MTP.
- **KV pool 859K / 3.28× at 256K** — no context advantage over spark1's
  single-box 933K / 3.56×, and it costs **both boxes** to get there.
- **Consumes the whole fleet.** No second endpoint remains for batch work,
  which is what spark2 existed for.
- **Host headroom 9-12 GB under load** — in historically dangerous territory.

The honest summary: **two boxes for one endpoint that decodes slower than one
box running Qwen3-Next, in exchange for better long-context prefill and a
much better APC payoff.** Whether that trade is worth it depends entirely on
subjective coding quality, which has **not been tested yet**. That is the
remaining open question and the only thing that could justify the fleet cost.
Nothing measured here forces a revert, and nothing measured here justifies
keeping it.

### Operational gotcha: no restart policy

`vllm-dspark` runs with **`RestartPolicy=no`** on both boxes (verified via
`docker inspect`), unlike `vllm-chat` which we fixed to
`--restart unless-stopped` after the 2026-07-02 reboot incident. This is
deliberate for an experiment — a reboot should come back to a clean slate
rather than auto-starting an unproven 2-box config that might starve the host
— but it means **a power cycle leaves both boxes with no chat endpoint at
all**, not even the old Qwen. If DSpark is kept, add the restart policy.

### Next

1. Re-run the anomalous warm-8K APC probe.
2. `--max-num-batched-tokens` 8192 → 40960: does cold prefill improve?
3. `num_speculative_tokens` 3 → 2, given position-3 acceptance of 0.42.
4. Subjective coding quality vs Qwen3.5-122B / Qwen3-Next.
5. Only if it clears that bar: consider util 0.75 for host-memory safety,
   and decide whether 1M context is worth util 0.85.

---

## 2026-08-01 — pipeline parallelism investigated, ruled out (conflicts with DSpark spec decode)

Motivation: TP=2's KV pool is bounded **per box** (859K tok, above) because
DeepSeek's MLA cache isn't head-sharded across TP ranks the way ordinary
multi-head attention KV is — both ranks must hold the same context length
simultaneously, so the cluster's usable context is capped by one box, not the
sum of two. Pipeline parallelism (PP) splits by *layer* instead, so each rank
would only cache its own layers — a plausible route to a genuinely bigger
effective pool without touching `GPU_UTIL`. Checked against the actual
installed source before writing any launch script, not assumed from the
general vLLM docs or the NVIDIA forum recipe (which is TP=2 only — no PP
mentions anywhere in that thread).

### Finding: PP and DSpark's speculative decode are mutually exclusive in this image

Checked directly against `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`'s installed
vLLM (`0.25.2.dev0+g752a3a504`):

- Served checkpoint's `architectures` (from the live HF `config.json`) is
  `DeepseekV4ForCausalLM` — the **target** model. It properly implements
  `SupportsPP` with standard `get_pp_group()` / `make_layers()` partitioning
  (`vllm/models/deepseek_v4/nvidia/model.py:1356-1357`).
- DSpark's **draft/speculator** model — registered as `"DSparkDraftModel"` →
  `DSparkDeepseekV4ForCausalLM` (`vllm/models/deepseek_v4/nvidia/dspark.py:267`)
  — has `nn.Module` as its *only* base class. No `SupportsPP`.
- `SpeculativeConfig.create_draft_parallel_config()` copies the target's
  `pipeline_parallel_size` straight onto the draft model's own
  `ParallelConfig` (`vllm/config/speculative.py:~1104`).
- `ModelConfig.verify_with_parallel_config()` gates on exactly this:
  `if pipeline_parallel_size > 1 and not registry.is_pp_supported_model(...):
  raise NotImplementedError("Pipeline parallelism is not supported for this
  model. Supported models implement the SupportsPP interface.")`
  (`vllm/config/model.py:1223`).

Net: `--pipeline-parallel-size 2` with `--speculative-config method=dspark`
active hits that `NotImplementedError` at engine init — a fast, loud failure,
not silent corruption, but a hard wall. **The only way to run PP=2 at all is
`SPEC=0`**, which drops DSpark's draft/verify acceleration entirely and falls
back to plain `DeepseekV4ForCausalLM`.

### Why this closes the door, for now

DSpark's entire reason for existing on this deployment is the
speculative-decode speedup. A PP=2 config that requires disabling it is a
different experiment — "is real layer-partitioned KV capacity worth trading
away decode acceleration for" — with its own unknowns (`flashinfer_b12x` MoE
backend behavior under PP untested, async-scheduling/chunked-prefill
interaction untested, zero community precedent). Not pursued right now.

**Not proven impossible, only ruled out for image `0.1.1`'s current draft
model.** If DSpark ever ships a PP-aware draft model upstream
(`DSparkDeepseekV4ForCausalLM` gaining `SupportsPP`), re-check against the new
source before assuming this still holds.

---

## 2026-08-03 — decision: KEEP. Verdict flipped from "mixed, quality untested" to adopted

The 2026-07-30 verdict was MIXED, blocked specifically on "subjective coding
quality vs the Qwen bar is not yet tested — that's the only thing that could
justify the fleet cost." That's now resolved from real usage, not a
controlled A/B: quality is a real, noticeable improvement over the
Qwen3-Next-80B baseline.

The other open question was throughput. The 2026-07-30 single-stream decode
number (30.3 tok/s, ~half the advertised 60-67) undersold how this
deployment is actually run: in throughput mode against the 6-slot cross-box
endpoint, concurrent streams compound to **~100+ tok/s aggregate at 3
concurrent streams**. Single-stream still tops at 30-40 tok/s — that hasn't
changed — but it isn't the number that matters for this usage pattern.

**Decision: DeepSeek-V4-Flash-DSpark stays up permanently, both boxes,
throughput mode.** `current-setup.md` updated same-change (banner, verdict,
ports table) to drop the "EXPERIMENT / not yet a keeper" framing.

**Not done as part of this decision, still open:**
- Repoint the four still-broken clients (MemPalace, llm_wiki,
  CampaignGenerator, opencode) to `deepseek-ai/DeepSeek-V4-Flash-DSpark` —
  `openclaw` was already repointed 2026-07-30 and is unaffected.
- Add `--restart unless-stopped` to both `docker run` invocations in
  `spin-up-vllm-dspark-2box.sh` — verified absent from both HEAD and WORKER
  commands as of this date, so neither container survives a reboot.
- Consider dropping `GPU_UTIL` 0.80 → 0.75 for host-memory safety margin
  (9-10 GB available at 176K prefill is close to the level that once wedged
  spark2 at util 0.88).

---

## 2026-08-04 — upgraded DSpark preview -> DeepSeek-V4-Flash-0731, executed

DeepSeek's official V4-Flash release (`deepseek-ai/DeepSeek-V4-Flash-0731`,
2026-07-31) landed the day after the 2026-08-03 keep decision above. Feasibility
was confirmed via a dedicated research pass (live-read the installed vLLM
source inside the running preview container, not just the published community
recipes, which disagreed with each other on image lineage, env vars, and the
draft-weight loader) before migrating. That research corrected one of its own
early conclusions on a second pass: the reasoning-effort wrapper concern was
initially overstated (thought it silently upgraded every request to expensive
"high" effort; the actual code only mis-maps an *explicit* `reasoning_effort=
"low"`, which nothing currently sends) — caught before acting on the wrong
premise, not after.

### What was done

1. Downloaded `deepseek-ai/DeepSeek-V4-Flash-0731` (166.9GB/156GiB, 48 shards,
   74 files) to spark1 via the dspark image's `hf download` (`--entrypoint
   bash`), same pattern as the original preview download. 74/74 files, 0
   `.incomplete`, clean exit.
2. Copied to spark2 over the direct cable: plain user-level `rsync -a` via the
   pre-existing `gx10-3e5c.local` SSH alias (spark1's own known-working
   spark1->spark2 key), not a container — the source directory turned out to
   be world-readable (`drwxr-xr-x`/`-rw-r--r--`, root-owned) so no permission
   workaround was needed on the read side. One non-essential file failed to
   copy (`trees/<rev>.json`, HF's internal cache-indexing metadata, `rw-------`
   root-only) — confirmed irrelevant to serving (vLLM loads from
   `snapshots/`/`blobs/`, not `trees/`) by verifying blob count (74), shard
   count (48), and total blob bytes (166,898,661,074) were byte-identical
   between boxes before proceeding. Moved into spark2's HF cache proper and
   `chown -R root:root` via a throwaway container (same pattern as the
   original bring-up's Step 0 — the cache directory itself is root-owned and
   requires a container to write into).
3. Edited `spin-up-vllm-dspark-2box.sh`: `MODEL` default -> `-0731`,
   `SPEC_TOKENS` default 3 -> 5 (matches `dspark_block_size` in `config.json`),
   added `VLLM_USE_BREAKABLE_CUDAGRAPH=0`. Also fixed a latent bug the upgrade
   would otherwise have tripped: the boot-wait loop's `/v1/models` check did
   `grep -q "DSpark"`, a literal substring that would never match `-0731` —
   changed to `grep -qF "${MODEL##*/}"` so the script works for any future
   checkpoint swap, not just this one. Added a post-boot "Skipping unknown"
   weight-mapping check (0 expected on both ranks) as a permanent addition to
   the boot signature, not a one-off manual check — this is exactly the class
   of bug a different DSpark image lineage hit on this same 0731 checkpoint
   per community reports, and it's cheap to keep checking on every future
   bring-up.
4. Ran the script. Clean swap: worker up, head up, endpoint live within the
   normal boot window. `docker logs` confirmed `VLLM_USE_BREAKABLE_CUDAGRAPH`
   was NOT flagged as an unknown env var (unlike genuinely-unrecognized
   `VLLM_BUILD_COMMIT`/`VLLM_BUILD_PIPELINE`/`VLLM_BUILD_URL`/`VLLM_IMAGE_TAG`
   noise in the same log) — it's a real, recognized flag in this image.

### Verification results

- **Weight-mapping: 0 "Skipping unknown" warnings on both ranks.** This was
  the one item flagged as "unverified without a live boot" in the 2026-08-03
  feasibility report — a different DSpark image lineage (not ours) reportedly
  drops draft-model tensors silently on checkpoint swaps. Confirmed clean on
  the actual 0731 tensors, not just inferred from the preview's clean load.
- **KV pool: 869,357 tok -> 3.32x @ 256K.** Preview was 845,284-859,040 tok /
  3.22-3.28x — matches within the run-to-run variance already documented for
  this deployment, as expected for byte-identical architecture/quant format.
- **Coherence: PASS.** Same "capital of France + count to 10" smoke test.
- **Tool calling: PASS.** A real `tools`-bearing request returned a structured
  `tool_calls` array (`finish_reason: "tool_calls"`, valid JSON args), not
  syntax leaked as text — confirms `deepseek_v4` parser still applies cleanly.
- **Single-stream decode: 21.0 / 31.2 / 23.5 tok/s across 3 different 600-token
  prompts (non-streaming, so no chunk-undercounting).** Noisy — likely
  reflects per-prompt draft-acceptance variance, same dynamic documented for
  the preview. Comparable to the preview's single-point 30.3 tok/s baseline,
  not a controlled A/B. **Open follow-up:** a same-prompt, multi-run A/B
  against the preview (still on disk, cheap to re-boot) would settle whether
  `SPEC_TOKENS=5` vs the preview's `3` helps, hurts, or is a wash — untested
  either way by this migration.
- Host memory at test time: spark1 14GB / spark2 11GB available — consistent
  with the preview's documented headroom, no new risk observed.

### What this migration did NOT touch

- The four still-broken clients (MemPalace, llm_wiki, CampaignGenerator,
  opencode) — unchanged, still pointed at pre-DSpark ids, still 400.
- `openclaw`, which HAD been fixed 2026-07-30 to point at the DSpark preview
  id — **this upgrade silently re-broke it**, since its configured id
  (`deepseek-ai/DeepSeek-V4-Flash-DSpark`) is no longer served. Flagged in
  `current-setup.md` §7 as a regression, not fixed as part of this change
  (same reasoning as the four clients: a config-file edit is a distinct
  action from the infra swap, left for explicit go-ahead).
- `--restart unless-stopped` — still absent on both containers, still a
  reboot risk, unchanged from the 2026-08-03 TODO.
- `GPU_UTIL` 0.80 -> 0.75 host-memory-safety consideration — unchanged,
  still open.
