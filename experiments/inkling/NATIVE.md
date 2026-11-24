# Native vLLM Inkling service

The selected backend is the retained native vLLM Inkling 2 Mini container on
Spark 2, with OpenAI-compatible base URL **`http://192.168.1.121:8004/v1`**
and model **`inkling-2-mini`**. Clef shares Spark 2 on port 8002. The selected
configuration now has a **411,520-token context**, **FP8 KV storage** and a
**13 GiB fixed KV budget**, with one active request, eager execution, prefix
caching and no speculative decoding. The context budget includes both input
and generated output. [Context capacity results](context/RESULTS.md).

Verified November 22, 2026 at **09:40 Pacific**: a **410,496-token archive
extraction** returned all three requested codes correctly with zero reasoning
tokens. Short arithmetic, JSON and negation checks also passed. Clef stayed
unchanged, and Spark 1 Qwen was healthy. [Context receipt](context/results-20261122.json).
The [earlier 64K activation receipt](native-active-20261122.json) remains historical.
At **13:11 Pacific**, the [FP8 Cognee follow-up](fp8-cognee/RESULTS.md) was also
complete and the services remained healthy: **6.03 s factual p50, 44/47 complete
and fully grounded**, and a **180.03 s** module agent that recovered the real
handoff with a remaining belief-attribution error.
The later [182K BF16 benchmark](bf16-cognee/RESULTS.md) completed at **13:45
Pacific**: 6.10 s factual p50, 44/47 complete and 43/47 grounded, a 170.57 s
agent, and a successful 181,116-token extraction. The selected **411K FP8
service was restored afterward**, with the same container/image/command and
a restarted process; Clef remained unchanged. The BF16 candidate is stopped
and retained.
At **14:17 Pacific**, the [stock Cognee follow-up](stock-cognee/RESULTS.md) was
complete: **10.87 s p50, 46/47 complete and 44/47 fully grounded**, versus
adaptive's 6.03 s and 44/47 in both quality measures. The stock module agent
took **263.71 s** and produced a supported core handoff with minor wording
caveats. Native Inkling and Clef kept the same process starts throughout.

## Client configuration

Disable the generated thinking channel with the verified native token ban:

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.1.121:8004/v1", api_key="local")
response = client.chat.completions.create(
    model="inkling-2-mini",
    messages=[{"role": "user", "content": "What is 17 multiplied by 19?"}],
    reasoning_effort="none",
    temperature=0,
    max_tokens=4096,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
        "bad_words": ["<|content_thinking|>"],
    },
)
print(response.choices[0].message.content)
```

These controls apply **per request**. Effort `none` correctly renders zero in
this native template but still permits thinking without the token ban. The
ban suppresses the generated thinking channel; the checkpoint's internal
layer loops still execute. The earlier [Cognee thinking comparison](native-thinking/RESULTS.md)
used the original 64K BF16 setup. The subsequent [FP8 replay](fp8-cognee/RESULTS.md)
took 6.03 s factual p50 versus that off arm's 4.51 s, with 44/47 complete
answers in both; grounding improved from 43/47 to 44/47. These single runs
also differ in context limit and cache allocation.
The 182K BF16 fixed-cache follow-up measured 6.10 s, essentially the same as
411K FP8's 6.03 s in these single runs; a larger BF16 window did not restore
the old 64K timing.

For either Cognee's `questions.py` or `sequential.py`, use
`--reasoning-effort none --extra-body-file experiments/inkling/no-thinking-options.json`.
The sequential runner's `--action-format json_schema` is the verified native
agent format workaround. Use a fresh output directory for every test. The
[Unsloth/DSpark options](dspark/THINKING.md) belong to that separate runtime.

## Operations

The retained Docker container is `inkling2mini`, checkpoint revision
`b85c3cfa615fc413cc9af3379f491b963f3b586a`, image
`sha256:f95362fc89f98d28a429d7047caa9090ccab881e1c4ffd0385f643627211c6de`.
Automatic fitting uses `--max-model-len auto --kv-cache-memory-bytes 13958643712
--kv-cache-dtype fp8`. The explicit KV budget also disables the default extensible
allocator in this image. The memory guard **`inkling-native-context-memory`**
stops only Inkling if host available memory drops below 12 GiB. Context receipts
and the guard script are at `spark2:~/inkling-context-20261122/`.

Check readiness and logs on Spark 2:

```bash
curl -fsS http://192.168.1.121:8004/health
docker logs --tail 50 inkling2mini
systemctl --user status inkling-native-context-memory
```

Stop native Inkling and its guard on Spark 2:

```bash
systemctl --user stop inkling-native-context-memory
docker stop inkling2mini
```

For a manual start, first ensure the GGUF service is stopped and at least
46 GiB of host memory is available before loading native Inkling. Then start
the retained container and guard on Spark 2:

```bash
docker start inkling2mini
inkling_context_id="$(docker inspect --format '{{.Id}}' inkling2mini)"
systemd-run --user --collect --unit=inkling-native-context-memory \
  --property=MemoryMax=128M --property=Restart=no \
  /usr/bin/python3 /home/kostadis/inkling-context-20261122/monitor.py \
  --root /home/kostadis/inkling-context-20261122 \
  --container "$inkling_context_id" --profile manual
```

The original 64K container is stopped as `inkling2mini-64k-20261122`.
Use the [saved restore action](context/README.md#saved-receipts-and-operations)
to return to it, including its original guard, without restarting Clef.

Clef does not need a restart. Native Inkling and its guard have no boot
autostart. The GGUF checkpoint, drafter and runtime remain on disk, with their
services stopped; [DSpark restart instructions](dspark/README.md) are retained
for a deliberate future backend switch.
