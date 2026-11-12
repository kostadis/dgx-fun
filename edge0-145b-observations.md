# Edge0-145B-NVFP4 — observations

## 2026-11-12 — documentation review; candidate for a controlled Spark trial

**Verdict: worth testing for memory efficiency and coding quality, but the
published evidence does not establish an upgrade over our Qwen3.8 deployment.**
This is a review of the released documentation and publisher benchmarks, not
a local inference evaluation. No weights were downloaded, services changed,
or Edge0 requests run. Future measurements belong in new dated entries.

### What is interesting

Edge0-145B is a trained latent-MoE derivative of Qwen3.8-Flash-Next, rather
than simply a new quantization of the parent. Expert inputs are compressed
from 2560 to 1792 dimensions while attention retains full width. It has
48 layers and routes to 10 of 512 experts per layer, alongside a shared
expert. The publisher describes training natively at 4 bits.

The reported 145B total includes an 84.6B expert pool and a 51.2B per-layer
n-gram embedding lookup table (PLE); reported active parameters are 10.12B
per token. The total-parameter headline alone is a poor guide to either
quality or resident memory. See the [architecture and model profiles][models].

### Quality: promising preservation, unconfirmed coding gain

Publisher-reported scores, higher is better:

| Benchmark | Edge0-145B | Qwen3.8-Flash-Next |
|---|---:|---:|
| deepswe 1.1 | 64.6 | 58.7 |
| Terminal-Bench 2.1 | 86.5 | 83.3 |
| AA-LCR | 79.3 | 79.1 |
| IFBench | 80.6 | 81.3 |
| GPQA Diamond | 90.4 | 91.7 |
| HLE | 35.4 | 35.9 |

These are explicitly **model-level scores, not separate NVFP4 and Q3_K
evaluations**. The table suggests broadly preserved capability and possible
coding gains, but does not establish the quality of this exact NVFP4 release.
Before crediting the gains, inspect the evaluation harness, agent scaffold,
reasoning/token budgets, and reproducible run evidence. This review did not
independently reproduce the scores. [Published quality table][models].

None of these scores settles campaign-summary fidelity, invented dialogue,
NPC voice, or spelling-card exception handling. Those need our own inputs.

### Speed: use the relevant hardware row

The NVFP4 headline of 98.3 tok/s agent decode and 3165.7 tok/s 16K prefill
comes from an RTX 5090 D 24 GB. The publisher's **RTX Spark N1X 60 GiB**
row instead reports:

| Format | 4K prefill | 16K prefill | Chat decode | Agent decode |
|---|---:|---:|---:|---:|
| NVFP4 | 821.2 | 922.4 | 40.1 | 46.0 |
| Q3_K | 674.0 | 770.3 | 37.7 | 36.7 |

All values are tokens/sec. That named system is not a measurement of our
GB10 Linux configuration. The table makes NVFP4 a reasonable first Spark
candidate, even though Q3_K is faster on the publisher's 5090 D.
[Full speed table][models].

Our historical Qwen DFlash 2 experiment already measured warm short-prompt
decode around 34 tok/s prose and
52 tok/s code, with large workload and warm-up effects. Edge0's published
results do not demonstrate a speed improvement over those measurements:
prompts, hardware, runtime, quantization and timing conditions differ.
That DFlash experiment is a historical comparison, not the current service;
use [current-setup.md](current-setup.md) for the recorded deployment inventory.
Local provenance is `qwen38-dflash2-observations.md`, an existing untracked
workspace report that is not included in this documentation commit.

### Memory and deployment catches

- **Download size is about 113 GB.** The repository includes a 51.2 GB
  `ple-fp8.gguf`, 47.6 GB `experts.bin`, skeleton, projections and draft
  assets. The advertised 47.79 GiB build size is not the complete download
  or a guaranteed process-memory budget. [Released files][files].
- **Residency figures are machine-specific.** The model card reports
  23.25 GiB VRAM plus 38.27 GiB host RAM on a discrete 5090 D, with disjoint
  expert sets. Those figures are not a measured unified-memory footprint
  on our Spark. [Model card][card].
- **The supported launch path uses a custom engine.** `edge0-infer serve`
  starts `edge0-strata`, a Strata fork with latent-MoE support. Build
  requirements include CUDA 13. Generic Hugging Face library launch snippets
  are not evidence that stock vLLM or our existing llama.cpp build can load
  the released asset pack. [Engine build documentation][engine].
- **Advertised context and launch defaults differ.** The model advertises
  256K; the wrapper defaults to 131072 tokens. PLE reads default to direct
  disk I/O, with a RAM option. Concurrent requests queue, and temperature-zero
  generation is not bit-for-bit repeatable. Long-context latency and parallel
  document throughput need explicit validation. [Serving reference][serve].
- **Weights have a community license.** The publisher distinguishes the
  framework's Apache-2.0 code from Edge0-specific model terms, retains upstream
  Qwen terms, and describes separate permission requirements for certain
  commercial model services, distribution and assistant products. Consult
  the actual terms for a deployment decision. [License summary][license].

The engine documentation also records a hang in an all-resident verification
path on an ARM64 RTX Spark N1X system; that path is opt-in there. It is a
concrete reason to validate our ARM64 configuration before assuming the
consumer-card results transfer. [Engine notes][engine].

### What would make the trial worthwhile

The working hypothesis is **similar useful quality with more memory headroom**,
not an assumed decode-speed breakthrough. Test NVFP4 first against the actual
Qwen service configuration, recording the exact revisions and launch options:

1. Run matched campaign summaries and NPC drafting inputs; review omissions,
   contradictions, invented dialogue, attribution and voice.
2. Run an executable coding task and a tool-call/result round trip. Measure
   successful completion, not just plausible output or token rate.
3. Measure cold and repeated prefill, first-output latency, decode and global
   available memory on identical requests with matched thinking and output
   budgets. Separate speculative decode and cache effects.
4. Validate retrieval and synthesis at 32K, 128K and near 256K, then test
   concurrent clients. A declared context limit is not a quality result.
5. Retain the existing service for recovery and record any actual deployment
   change in `current-setup.md` and the request registry as required.

Promote only if the quality holds and the memory or latency benefit is useful
on our workload. Until then, this is a credible experiment candidate.

[card]: https://huggingface.co/Edge0/Edge0-145B-NVFP4
[files]: https://huggingface.co/Edge0/Edge0-145B-NVFP4/tree/main
[models]: https://github.com/Edge0-AI/Edge0/blob/main/docs/models/edge0-145b.md
[engine]: https://github.com/Edge0-AI/Edge0/blob/main/edge0-strata/EDGE0.md
[serve]: https://github.com/Edge0-AI/Edge0/blob/main/edge0-infer/docs/edge0-145b.md
[license]: https://github.com/Edge0-AI/Edge0#license
