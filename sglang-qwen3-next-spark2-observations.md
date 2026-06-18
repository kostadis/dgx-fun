# SGLang serving Qwen3-Next-80B-A3B-Instruct-FP8 on spark2 — observations

Append-only experiment log. spark2 (192.168.1.69) chat slot, port 8001,
container `sglang-chat`. The experiment: serve the **exact same model**
spark2 already ran under vLLM, but on **SGLang**, to feel the friction of
a second serving engine on GB10 and to set up a real A/B. Suboptimal by
design — calibration, not optimisation.

Setup/runbook facts live in `current-setup.md` §4. The vLLM side of the
A/B is `qwen3-next-80b-observations.md`. Don't rewrite entries — add dated
sections.

---

## 2026-06-11 — getting it to boot: two GB10-specific gates

Goal: replace vLLM with SGLang in the spark2 chat slot, same model
(`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`), same flags mapped 1:1. Took
three boots to clear two distinct Blackwell/GB10 (sm_121) gates. Both are
now baked into `spin-up-sglang-qwen3-next-80b.sh`.

### Gate 1 — FlashInfer is not allowed for hybrid GDN on Blackwell

SGLang auto-selects the FlashInfer attention backend, loads all 8 weight
shards (~9 min), then **asserts itself out**:

```
AssertionError: triton or trtllm_mha backend are the only supported
backends on Blackwell GPUs for hybrid GDN models, use --attention-backend
triton or --attention-backend trtllm_mha to specify the backend.
```

Fix: pin `--attention-backend triton`. **Important correction to a prior
assumption:** the vLLM script header worried about Qwen3-Next falling
through to a "slow Triton fallback kernel" as a *perf ceiling*. On SGLang
+ GB10 that framing is wrong — Triton is not a fallback, it's the **only
supported attention path** for this architecture on this GPU. So any
SGLang-vs-vLLM comparison here is really "SGLang-on-Triton vs vLLM."

### Gate 2 — DeepGEMM "Unknown recipe" in FP8-MoE CUDA-graph capture (0.5.9 only)

With Triton pinned, SGLang **0.5.9** (`scitrera/dgx-spark-sglang:0.5.9-t5`,
the NVIDIA-forum-recommended stable Spark build) gets further — past
weight load, into CUDA-graph capture — then dies:

```
Capture cuda graph failed: Assertion error
(.../repo-deepgemm-src/.../heuristics/.../layout.hpp:56): Unknown recipe
```

This is DeepGEMM (SGLang's FP8 GEMM library) lacking a kernel *recipe* for
this MoE GEMM shape on GB10. SGLang's own suggested fixes (lower
mem-fraction, smaller `--cuda-graph-max-bs`, `--disable-cuda-graph`) are
generic OOM/capture advice and don't address a missing-recipe assertion —
the only real options are disable-cuda-graph (SGLang flags "huge perf
loss", and the call may re-fire at decode) or a newer kernel build.

Fix: **upgrade the image, don't downgrade flags.** The official
`lmsysorg/sglang:v0.5.10.post1-cu130` (SGLang 0.5.10, CUDA 13.0.1,
multi-arch arm64) has the recipe. Clean boot, no DeepGEMM error.

### Result — it serves

Image `lmsysorg/sglang:v0.5.10.post1-cu130`, container `sglang-chat`,
128K context, `fp8_e5m2` KV, `--attention-backend triton`,
`--max-running-requests 8`, `--tool-call-parser qwen25`, mem-fraction
0.88. Verified 2026-06-11:

- **Content:** clean, coherent ("what is a tokenizer?" → correct).
- **Tool calls:** `finish_reason: tool_calls`, proper `tool_calls` array
  (`get_weather({"city":"Paris"})`, `content: null`). The `qwen25` parser
  is the correct analog to vLLM's `hermes` for this model.
- **Decode:** 128 tokens in ~3.06 s wall on a tiny prompt ≈ **~42 tok/s**.

### Image notes (for next time)

- `lmsysorg/sglang:spark` (the "official Spark" tag) is ~7 months old —
  predates Qwen3-Next. Don't use it.
- `scitrera/dgx-spark-sglang:0.5.9-t5` — forum-recommended, but Gate 2
  blocks FP8 MoE. Fine for non-MoE / known-good recipes; not this model.
- `lmsysorg/sglang:v0.5.10.post1-cu130` — **the working one** for
  Qwen3-Next FP8 on GB10.
- Entrypoint is the NVIDIA wrapper (`/opt/nvidia/nvidia_entrypoint.sh`);
  launch with `python3 -m sglang.launch_server` (all our flags confirmed
  present in 0.5.10). `sglang serve` also exists but the module form is
  the robust choice.

---

## TODO — the actual A/B (not done yet)

Booting is not the experiment; it was the cost of admission. Still owed:

- **Prefill / TTFT vs vLLM** at a few context depths (1K / 32K / 128K) —
  this is the axis that matters for the user's read-heavy workflows, and
  where SGLang's **RadixAttention prefix cache** could actually beat vLLM
  on shared-context traffic. Use `bench-prefill.sh` / the needle script
  against both `192.168.1.147:8001` (vLLM Coder-Next — different model,
  caveat) and a vLLM Qwen3-Next reference.
- **Decode tok/s vs vLLM** on the identical model. Expectation:
  ~wash (both bandwidth-bound at ~3B active on GB10's ~273 GB/s). The
  ~42 tok/s single-prompt number above is a placeholder, not a benchmark.
- **Prefix-cache win, quantified.** Fire N requests sharing a large
  system prompt; measure TTFT on requests 2..N vs request 1. This is the
  one place SGLang is *expected* to pull ahead — confirm or deny.
- Note for the comparison: vLLM used `fp8` (e4m3) KV; SGLang here uses
  `fp8_e5m2` (scale-free). Not identical KV dtype — flag if it shows up in
  quality.

---

## 2026-06-11 — reverted to vLLM same day

Decision: **reverted spark2 to vLLM** right after verifying SGLang worked.
Rationale (user call, and the right one): SGLang is no serving improvement
on this box. Decode is bandwidth-bound, the marketed SGLang speedup relies
on kernels GB10 can't run (Triton-only), quality is identical (same
weights), and operationally it's *worse* than vLLM here — slower cold
start, less-mature stack, two GB10 gates to clear. The only un-measured
escape hatch (RadixAttention prefix-cache win on read-heavy traffic) was
not worth keeping a less-mature server live to chase speculatively.

What stays as the deliverable: this log, the reproducible
`spin-up-sglang-qwen3-next-80b.sh` (both GB10 gates baked in), and the
memory entry. Re-runnable any time for the prefix-cache A/B in the TODO
above — that benchmark, not a permanent swap, is the way to settle whether
SGLang ever earns this slot. **Live state on spark2 is vLLM again**
(`vllm-chat`, `MAX_SEQS=8 bash ~/spin-up-vllm-qwen3-next-80b.sh`).
