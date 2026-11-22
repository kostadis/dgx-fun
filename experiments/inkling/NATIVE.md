# Native vLLM Inkling service

The selected backend is the retained native vLLM Inkling 2 Mini container on
Spark 2, with OpenAI-compatible base URL **`http://192.168.1.121:8004/v1`**
and model **`inkling-2-mini`**. Clef shares Spark 2 on port 8002. The native
configuration is the one used by the [paired thinking test](native-thinking/RESULTS.md):
65,536-token context, one active request, eager execution, prefix caching and
no speculative decoding.

Verified November 22, 2026 at **09:05 Pacific**: native arithmetic and JSON
smoke checks passed with zero reasoning tokens, DSpark was stopped, and Clef
and Spark 1 Qwen were healthy. [Activation receipt](native-active-20261122.json).

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
layer loops still execute. The server's launch settings are unchanged from
the measured native configuration.

For either Cognee's `questions.py` or `sequential.py`, use
`--reasoning-effort none --extra-body-file experiments/inkling/no-thinking-options.json`.
The sequential runner's `--action-format json_schema` is the verified native
agent format workaround. Use a fresh output directory for every test. The
[Unsloth/DSpark options](dspark/THINKING.md) belong to that separate runtime.

## Operations

The retained Docker container is `inkling2mini`, checkpoint revision
`b85c3cfa615fc413cc9af3379f491b963f3b586a`, image
`sha256:f95362fc89f98d28a429d7047caa9090ccab881e1c4ffd0385f643627211c6de`.
The memory guard `inkling-native-thinking-memory` stops only Inkling if host
available memory drops below 12 GiB. Activation receipts and the guard script
are at `spark2:~/inkling-native-active-20261122/`.

Check readiness and logs on Spark 2:

```bash
curl -fsS http://192.168.1.121:8004/health
docker logs --tail 50 inkling2mini
systemctl --user status inkling-native-thinking-memory
```

Stop native Inkling and its guard on Spark 2:

```bash
systemctl --user stop inkling-native-thinking-memory
docker stop inkling2mini
```

For a manual start, first ensure the GGUF service is stopped and at least
46 GiB of host memory is available before loading native Inkling. Then start
the retained container and guard on Spark 2:

```bash
docker start inkling2mini
systemd-run --user --collect --unit=inkling-native-thinking-memory \
  --setenv=INKLING_NATIVE_THINKING_ROOT=/home/kostadis/inkling-native-active-20261122 \
  --property=MemoryMax=128M --property=Restart=no \
  /usr/bin/python3 /home/kostadis/inkling-native-active-20261122/monitor.py
```

Clef does not need a restart. Native Inkling and its guard have no boot
autostart. The GGUF checkpoint, drafter and runtime remain on disk, with their
services stopped; [DSpark restart instructions](dspark/README.md) are retained
for a deliberate future backend switch.
