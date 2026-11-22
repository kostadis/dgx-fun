# Inkling Mini: Unsloth GGUF and DSpark on DGX Spark

Completed November 21, 2026 (Pacific time). The Unsloth implementation works
alongside Clef. DSpark helps the structured prompt substantially, but disabling
drafting was fastest across this small mix of prompts. These results are
independent of the AtlasKV experiment and the earlier Cognee quality benchmarks.

> **Subsequent control diagnostic:** top-level `reasoning_effort="none"` in this
> pinned server actually renders Inkling effort **0.9**. The rates below retain
> that original behavior. They are not results for the corrected no-thinking
> configuration. [Verified cause and workaround](THINKING.md).

## Speed

Rates are median **API completion tokens / total request seconds**, including
prefill and reasoning tokens. Each cell covers two requests; the mixed median
covers all six requests in that arm.

| Inkling backend | Prose | Code | Structured prompt | Mixed median |
|---|---:|---:|---:|---:|
| Original vLLM, native checkpoint | 33.15 | 32.97 | 29.76 | 31.80 |
| Unsloth GGUF, drafting off | 38.33 | 38.33 | 34.69 | **38.27** |
| Unsloth GGUF, DSpark draft 3 | 32.27 | 35.77 | 44.15 | 35.77 |
| Unsloth GGUF, DSpark draft 7 | 25.80 | 32.20 | **54.90** | 32.20 |

Units: tokens/second. Relative to the identical GGUF without drafting, draft
three improved the structured-prompt rate by **27%** and draft seven by **58%**.
Both reduced prose throughput. Aggregate draft acceptance was **58.1%** with
draft three and **34.1%** with draft seven. Speculation counters and response
timings confirm that the drafter actually ran.

**These are not completed-answer timings.** Every GGUF speed probe exhausted
its 512-token budget in reasoning before emitting visible answer text. The
original vLLM code and structured probes did the same; its prose probes emitted
some visible text. Requesting `reasoning_effort="none"` did not eliminate
thinking. In particular, 54.90 tokens/second does not mean that complete JSON
extraction or Cognee answers finished at that rate. No complete records were
emitted in the long structured probes, so their extraction accuracy is
unassessed.

## Short completed answers and API checks

All four checks passed in every arm: arithmetic, exact four-field JSON
extraction, a JSON negation check, and a required tool call with the correct
function and arguments. These are sixteen passing checks, with one observation
per check per arm.

| Backend | Arithmetic | Four-field JSON | Negation JSON | Tool call |
|---|---:|---:|---:|---:|
| Original vLLM | 1.35 s | 4.90 s | 6.75 s | 4.04 s |
| GGUF, drafting off | 1.95 s | 4.28 s | 6.35 s | 2.06 s |
| GGUF, draft 3 | 1.76 s | 3.92 s | 6.05 s | 1.73 s |
| GGUF, draft 7 | 1.85 s | 3.70 s | 4.63 s | 1.49 s |

This supports basic OpenAI-compatible chat, schema output and tool-call
operation. It does not establish broad quality parity. The original 48 factual
questions and the agentic Phandelver task were not rerun with this GGUF.
Earlier Inkling quality scores belong to the original native vLLM deployment.

## Configuration and limits

The three GGUF arms used the same checksum-verified
`Inkling-2-Mini-UD-Q4_K_XL.gguf` target (17,629,759,456 bytes), with the
402,278,560-byte `dspark-Inkling-2-Mini-Q8_0.gguf` drafter for the speculative
arms. Model revision: `ae9327c5fbc7333116ba32eee78bb5d2bbefae4f`.
Runtime: Unsloth ARM64 CUDA 13 release `b12588-mix-c0c3de8`.
Full file hashes and launch commands are in [the measured data](results-20261121.json).

All GGUF arms used 65,536 context tokens, one active request, full GPU offload,
flash attention, FP16 K/V, eight CPU threads and disabled prompt caching.
Requests used temperature zero, top-p one and seed 604. Three synthetic prompts
were repeated twice per arm, with a separate unmeasured warmup. The fixture and
benchmark script hashes match across all arms.

The original vLLM arm retained its native checkpoint, precision and serving
configuration. It also ran during a CPU-limited model download. Comparing it
with GGUF changes several variables; the same-GGUF drafting comparison is the
cleaner experiment. Arm order was fixed, Clef shared the device, and two repeats
do not support a confidence interval. Streaming event gaps are not per-token
latency because speculative decoding can return several tokens together.

Minimum sampled host available memory was **42.25 GiB** without drafting,
**41.48 GiB** with draft three, and **41.43 GiB** with draft seven. Clef retained
the same container ID, image and start time throughout the comparisons. No
12 GiB host-memory guard tripped. Process RSS excludes GPU allocations and
should not be interpreted as the model's total unified-memory footprint.

Installation required a larger CPU-only download helper: Xet reconstruction
exceeded initial 4 GiB and 12 GiB cgroup limits before a 32 GiB helper completed
the verified transfer. One port-reuse preflight also failed before model startup
and was corrected. These installation events are excluded from inference
measurements; the original Inkling container was retained for rollback.

## Running service

The published service uses **DSpark draft three**, the guide's recommended
default, as a trial configuration. It is a compromise between the structured
gain and the prose penalty; it is not the fastest mixed configuration above.

- Base URL: **`http://192.168.1.121:8004/v1`**
- Model: **`inkling-2-mini`**; API-key placeholder: `local`
- Context: **65,536 tokens**, one active request
- User unit: `inkling-gguf-active`; guard: `inkling-gguf-active-memory`
- Clef remains healthy on port 8002; Hibana is stopped.
- Spark 1's pinned Qwen container remains running and was not restarted.

Model discovery, a real chat completion and nonzero speculation counters were
verified after publishing the endpoint. These are transient units without boot
autostart. [Client and restart instructions](README.md) and
[the operations receipt](operations-20261121.json) document the deployed state.

Sources: [Unsloth Inkling Mini guide](https://unsloth.ai/docs/models/inkling-2-mini),
[DSpark drafter instructions](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/dspark/README.md),
[pinned runtime release](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8).
The linked [vLLM Inkling Small recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Small)
is for the much larger Small model; this experiment uses Mini.
