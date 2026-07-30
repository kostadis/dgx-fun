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
