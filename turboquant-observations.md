# TurboQuant KV-cache observations — Qwen3-Next 80B on spark1

Append-only experiment log for running **TurboQuant KV-cache
quantization** on spark1's `vllm-chat` slot. Companion to
`current-setup.md` (which documents the *live* config) and to the
spin-up script `spin-up-vllm-qwen3-next-80b-turboquant.sh`. Don't
rewrite history here — add new dated sections.

## What's under test

- **Model:** `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` (unchanged FP8
  weights; hybrid Gated-DeltaNet + periodic full-attention + MoE,
  ~3B active / ~80B total).
- **Change:** KV cache `--kv-cache-dtype fp8` → **`turboquant_k8v4`**
  (TurboQuant: FP8 keys + 4-bit values; only the full-attention layers
  carry KV, so only those are compressed — GDN layers carry none).
- **Engine:** vLLM **0.22.0** (`vllm/vllm-openai:v0.22.0-aarch64`,
  CUDA 13.0, torch 2.11), 128K context, `--max-num-batched-tokens 4096`,
  CUDA graphs on, hermes tool parser.
- **Why 0.22.0:** hybrid TurboQuant support landed in 0.21.0 (#39931),
  but the Qwen3-Next degenerate-output-under-CUDA-graph bug (#40880) was
  only fixed in 0.22.0. Ampere bug #40124 is N/A (Spark is Blackwell
  sm_121). Open #41726 (long chunked-prefill crash) guarded by
  `--max-num-batched-tokens 4096`.
- **Startup caveat:** vLLM logs *"TurboQuant is not yet compatible with
  FlashAttention >= 3 → overriding flash_attn_version to 2"* — the
  full-attention layers run on FA2, not FA3.

---

## 2026-05-30 — Phase A: deploy + long-context quality (Tier 2)

Deployed via `spin-up-vllm-qwen3-next-80b-turboquant.sh`. Startup clean
(~13 min warm-cache: shard load + torch.compile 53.8s). Engine confirms
`Using TURBOQUANT attention backend out of potential backends:
['TURBOQUANT']`. `system_fingerprint: vllm-0.22.0-1cdebd1d`.

### Correctness smoke (Tier 1 — necessary, not sufficient)
- "Capital of France?" → "The capital of France is Paris." (coherent,
  no #40880 degenerate output).
- Tool call (hermes): `content:null` + `tool_calls[get_weather]` +
  `{"location":"Paris"}` + `finish_reason:tool_calls` → **PASS**.

### Long-context needle-in-haystack (Tier 2 — the real KV-quant test)
Tool: `bench-longctx-needle.sh` — plants a unique passphrase
(`CRIMSON-FALCON-7294`) at depth fractions 0.25/0.5/0.9 in prompts of
8K→120K tokens, `temperature=0`, asks for exact recall. TTFT is
time-to-first-token (prefill-dominated at these lengths).

```
    len  depth   TTFT_s  verdict  answer
   8192   0.25     3.23     PASS  CRIMSON-FALCON-7294
   8192   0.50     3.40     PASS  CRIMSON-FALCON-7294
   8192   0.90     3.52     PASS  CRIMSON-FALCON-7294
  32768   0.25    17.16     PASS  CRIMSON-FALCON-7294
  32768   0.50    17.27     PASS  CRIMSON-FALCON-7294
  32768   0.90    17.09     PASS  CRIMSON-FALCON-7294
  65536   0.25    44.20     PASS  CRIMSON-FALCON-7294
  65536   0.50    43.45     PASS  CRIMSON-FALCON-7294
  65536   0.90    43.86     PASS  CRIMSON-FALCON-7294
 120000   0.25   109.74     PASS  CRIMSON-FALCON-7294
 120000   0.50   109.59     PASS  CRIMSON-FALCON-7294
 120000   0.90   109.50     PASS  CRIMSON-FALCON-7294
```

**Verdict: 12/12 PASS.** Recall is intact at every depth across the
full 128K window, including depth 0.9 (needle near the end — the hardest
case for a corrupted cache). The primary KV-quant failure mode (cache
corruption → wrong/garbled recall) is **not present**. Expected for
k8v4 (vLLM's near-lossless preset, +1.17% PPL) — now verified, not
assumed.

The three 120K rows completing also clears the **#41726 long
chunked-prefill crash** — did not trigger.

### Scope / caveats
- Single-needle *retrieval*. Proves the cache isn't corrupting facts;
  does NOT prove zero degradation on multi-hop reasoning or long-form
  generation quality. k8v4's +1.17% PPL implies minimal, but this test
  wouldn't catch subtle reasoning drift.
- **TTFT here is a TurboQuant baseline only, NOT an A/B.** ~110s to
  first token at 120K is steep — the FA2 + hybrid-kernel prefill
  ceiling on GB10. Whether this is slower than plain fp8 is unresolved
  until Tier 4. Standing prediction: fp8 ≈ equal or faster, since
  TurboQuant adds Hadamard rotation + FA2 on exactly this prefill path.
- Decode tok/s not yet measured (needle test is prefill-dominated).

---

## Open / TODO

- **Tier 4 A/B vs plain fp8** (the worth-it question): revert with
  `bash ~/spin-up-vllm-qwen3-next-80b.sh`, rerun `bench-longctx-needle.sh`
  + `bench-prefill.sh`, diff. Quality should match (both PASS); the
  decision is whether the memory saving justifies any speed cost. On a
  hybrid that only KV-caches a few layers, the memory win is small, so
  "fp8 was the better default" is a plausible — and still successful —
  outcome.
- Decode-throughput A/B (tok/s on a 512-token generation).
- More aggressive presets (`turboquant_4bit_nc` / `3bit_nc`) to map the
  quality cliff — expect needle FAILs to appear at long depths first.
- Real-workload soak: llm_wiki / opencode / mempalace for a day.
