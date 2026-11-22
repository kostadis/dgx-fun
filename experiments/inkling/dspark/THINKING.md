# Disable the generated thinking channel

> **DSpark-specific controls.** Port 8004 now serves native vLLM Inkling at the
> user's request. Use its different [native controls](../NATIVE.md). The
> configuration below is retained for the stopped DSpark backend.

Verified November 22, 2026 UTC against the pinned Unsloth runtime and GGUF in
[the DSpark trial](README.md). The checkpoint's `<|content_thinking|>` token is
**200008**, verified through the running server's `/tokenize` endpoint.

Use these **per-request** controls when the DSpark OpenAI-compatible endpoint is running:

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.1.121:8004/v1", api_key="local")
response = client.chat.completions.create(
    model="inkling-2-mini",
    messages=[{"role": "user", "content": "What is 17 multiplied by 19?"}],
    temperature=0,
    max_tokens=4096,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": 0.0},
        "logit_bias": {"200008": False},
    },
)
print(response.choices[0].message.content)
```

Omit top-level `reasoning_effort="none"`. In this pinned server, that value
erases `reasoning_effort` from the template kwargs. Inkling's template then
falls back to **0.9**, even if nested effort zero was also supplied. The live
`/apply-template` diagnostic confirmed both cases. Supplying effort zero only
inside `chat_template_kwargs` renders **0** correctly. This corrects an earlier
interpretation that the original requests were receiving the lowest effort.

Even a correctly rendered effort zero still permits a thought. On the arithmetic
probe it took 1.45 seconds and emitted 119 thinking characters. Blocking token
200008 returned the correct answer in **0.54 seconds**, five output tokens,
and no thinking text. Exact JSON, negation and native tool-call checks also
passed without thinking. The previously stalled flood question finished in
9.89 seconds but remained factually wrong; speed does not establish grounding.

The server's `False` logit-bias value bans a token using negative infinity.
This is a llama.cpp extension and a model-specific inference workaround.
It prevents entering the generated thinking channel; the model's fixed internal
layer loops still execute. The service has not been globally changed or
restarted: requests need these controls. Clef remains running.

`reasoning_budget_tokens=0` was also tested, with automatic and explicit native
thinking delimiters. Both emitted repeated empty thinking blocks until the
256-token cap, without answering. That option is not the selected workaround.
Changing `reasoning_format` merely changes parsing and does not remove generation.

The earlier native vLLM run is a different case: an offline audit of its exact
image and cached original template renders effort `none` as **0**, correctly.
It still generated 11,497 reasoning tokens in the 48-question test and 8,589
in the completed agent run. See the [retrospective audit](native-reasoning-audit-20261122.json).
The Unsloth effort-mapping bug should not be attributed to that native run.
The completed [paired native experiment](../native-thinking/RESULTS.md) then
cut factual p50 from 10.38 to 4.51 seconds on the same native process, with
44/47 complete answers in both arms and one additional unsupported claim
without thinking. Native vLLM used `bad_words:["<|content_thinking|>"]`;
the DSpark backend uses the different controls shown above.

[Diagnostic metrics](thinking-controls-20261122.json) preserve these observations.
The earlier synthetic speed results retain their original settings and are not
relabelled as this no-thinking configuration. Full requests and responses remain
under ignored `raw/pabtso/cognee-20261120/inkling-thinking-controls-20261122/`.

For Cognee, pass the saved [options](no-thinking-options.json) to both readers
and agent decisions, and omit `--reasoning-effort none`:

```bash
trial=experiments/adaptive-chunking/raw/pabtso/cognee-20261120
/home/kostadis/cognee-local/.venv/bin/python \
  experiments/adaptive-chunking/cognee/questions.py \
  --source "$trial/adaptive-context-8" --output "$trial/inkling-no-thinking-new/questions" \
  --cases "$trial/frozen/cases.json" \
  --backend http://192.168.1.121:8004 --model inkling-2-mini \
  --extra-body-file experiments/inkling/dspark/no-thinking-options.json \
  --max-output-tokens 4096 --continue-on-truncation
```

The sequential runner accepts the same `--extra-body-file` plus
`--reader-max-output-tokens 4096`; its action cap remains 2,800 tokens. Use the
same narrative question, schema-guided actions and a fresh output directory.
Factual truncations are retained as failed generations; HTTP or retrieval errors
remain fatal. Existing runner defaults preserve the previous behavior.

Sources: [Thinking Machines' model limitations](https://huggingface.co/thinkingmachines/Inkling-2-Mini#7-bias-risks-and-limitations),
[pinned server effort handling](https://github.com/unslothai/llama.cpp/blob/b12588-mix-c0c3de8/tools/server/server-common.cpp#L1394),
[native logit-bias handling](https://github.com/unslothai/llama.cpp/blob/b12588-mix-c0c3de8/tools/server/server-schema.cpp#L353).
