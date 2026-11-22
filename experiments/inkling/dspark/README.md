# Inkling Mini with Unsloth DSpark

> **Stopped November 22, 2026 (Pacific):** the user selected native vLLM Inkling
> for port 8004. [Current native client and operations guide](../NATIVE.md).
> This directory retains the DSpark installation, measurements and optional
> restart instructions. Its request controls apply only when DSpark is running.

An isolated speed and API experiment following the
[Unsloth DGX Spark guide](https://unsloth.ai/docs/models/inkling-2-mini) and the
[drafter instructions](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/dspark/README.md).
All four arms completed, and the published service passed its final API and
co-resident service checks. See [results and limitations](RESULTS.md),
[measured data](results-20261121.json) and
[deployment receipt](operations-20261121.json).
AtlasKV and earlier factual/agent benchmark results remain separate.

The [completed Cognee follow-up](COGNEE.md) uses corrected no-thinking controls:
48 factual answers, 44/47 strict, 10.35 s p50; the module agent finished in
221.57 s but missed the actual Chapter 4-to-5 return hook. This is a separate
configuration from the original synthetic speed probes.

> **Thinking-control correction:** this pinned server mishandles top-level
> `reasoning_effort="none"`, rendering effort 0.9. For the verified per-request
> no-thinking workaround, use [these controls](THINKING.md). The synthetic
> results retain their original settings; the later Cognee run used corrected controls.

## Use DSpark when selected

This retained backend uses Unsloth llama-server with **DSpark draft three**,
65,536 context tokens and one active request. The following client example
applies after deliberately selecting DSpark in place of native vLLM. Clef
can remain running on port 8002 during that switch.

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.1.121:8004/v1", api_key="local")
response = client.chat.completions.create(
    model="inkling-2-mini",
    messages=[{"role": "user", "content": "What is 17 multiplied by 19?"}],
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": 0.0},
        "logit_bias": {"200008": False},
    },
    temperature=0,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

The example blocks this checkpoint's thinking-channel token. Native effort zero
alone still permits thinking. The earlier speed probes used the incompatible
top-level `none` setting and frequently never reached visible answers.

The service and its 12 GiB host-headroom guard are transient user units, with no
boot autostart. On Spark 2, after stopping a preceding GGUF candidate, restart
the saved configuration with:

```bash
python3 ~/inkling-dspark-trial/start.py --profile dspark-3 \
  --unit inkling-gguf-active --host 192.168.1.121 --port 8004 --stop-vllm
systemd-run --user --collect --unit=inkling-gguf-active-memory \
  --property=MemoryMax=128M --property=Restart=no \
  /usr/bin/python3 /home/kostadis/inkling-dspark-trial/monitor.py \
  --unit inkling-gguf-active \
  --output /home/kostadis/inkling-dspark-trial/inkling-gguf-active-memory.json
```

Stop an active GGUF deployment, or roll back to the retained native vLLM
container, using these commands **on Spark 2**:

```bash
systemctl --user stop inkling-gguf-active inkling-gguf-active-memory
# Optional rollback, after the GGUF unit has stopped:
docker start inkling2mini
```

Clef does not need to be restarted. Use
`journalctl --user -u inkling-gguf-active -n 50` for server logs and
`curl -fsS http://192.168.1.121:8004/health` for readiness.

## Experiment protocol

The target is Unsloth's recommended `Inkling-2-Mini-UD-Q4_K_XL.gguf`, with its
Q8_0 DSpark drafter. Both are pinned to repository revision
`ae9327c5fbc7333116ba32eee78bb5d2bbefae4f` and verified against published
SHA-256 values. The ARM64 CUDA 13 runtime is the official
[`b12588-mix-c0c3de8` release](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8),
archive SHA-256 `1ae89924180ec0ce8a18714737e591bf171db70f4fc5f5bd9fc160c7455004c7`.
Text-only tests omit the vision projector.

The arms are the retained original vLLM Inkling deployment, the GGUF without
drafting, and the same GGUF with DSpark draft lengths three and seven. Each
receives the same three synthetic prose/code/structured-output prompts twice,
with temperature zero, seed 604, reasoning effort `none`, and a 512-token
output budget. These bounded outputs measure speed rather than complete-task
quality. Four separate short checks cover arithmetic, exact JSON extraction,
negation, and required tool calling. Warmup is recorded separately.

The common speed statistic is API completion tokens divided by request wall
time, including prefill. SSE chunks can contain several speculative tokens;
their gaps are not token-level latency. Provider timing/counter data is also
retained. The GGUF arms disable prompt caching and share a 65,536-token context,
one active request, full GPU offload, FP16 K/V, flash attention, and eight CPU
threads. The original vLLM deployment retains its existing 65,536-token
context, eager mode and cache configuration. Its comparison with GGUF changes
both runtime and quantization; the GGUF off/on comparison isolates drafting
more closely.

All execution is on Spark 2 alongside Clef. The original `inkling2mini`
container is retained for rollback. Its completed baseline is saved before it
is stopped to free working memory for the final model transfer and backend
swap. Trial candidates initially bind
to loopback port 30008. Candidate startup enforces host headroom, bounds its
systemd cgroup, and verifies Clef's container identity/start time. No model
weights or runtime packages are changed inside Clef or the existing vLLM
container. Spark 1's Qwen service remains outside this experiment.

`download.py` supports bounded streaming HTTP and Xet downloads.
High-performance Xet helpers exceeded initial 4 GiB and 12 GiB cgroup limits;
a 32 GiB CPU-only helper completed the checksum-verified download. Those
failures are installation events, separate from model inference. Raw requests,
responses, logs and operations receipts are retained under ignored `raw/` and
`spark2:~/inkling-dspark-trial/`.

Regenerate the public aggregate from the saved local raw receipts with:

```bash
python3 experiments/inkling/dspark/analyze.py \
  --root experiments/inkling/dspark/raw/20261121 \
  --output experiments/inkling/dspark/results-20261121.json
```
