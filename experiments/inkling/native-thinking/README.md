# Native Inkling thinking comparison

Paired Cognee experiment, November 22, 2026 (Pacific time). The run compares
the original native Inkling configuration at minimum effort with the same
configuration while blocking its generated thinking channel. Both arms,
source review and DSpark restoration are complete.
[Results and quality review](RESULTS.md) · [Audited metrics](results-20261122.json).
Factual p50 fell **10.38 → 4.51 s** and the 48-query sum **564.26 → 222.25 s**.
Complete answers stayed **44/47**; fully grounded complete answers fell
**44/47 → 43/47**. Agent time fell **579.03 → 164.69 s**, with fewer searches
and substantive source errors in both arms.

Both arms use the retained `inkling2mini` container and image
`sha256:f95362fc89f98d28a429d7047caa9090ccab881e1c4ffd0385f643627211c6de`,
original checkpoint revision `b85c3cfa615fc413cc9af3379f491b963f3b586a`,
65,536 context tokens, one active request, 0.26 GPU-memory utilization and eager
execution without speculation. The native engine enables prefix caching.
The container is not restarted between arms. Clef remains running throughout.

Every request keeps `reasoning_effort="none"`, which this native renderer
correctly maps to effort zero. The off arm adds:

```json
{"bad_words": ["<|content_thinking|>"]}
```

The native tokenizer maps that special token to 200008. An offline check of the
exact image verified that `bad_words` reaches native sampling parameters. A
live arithmetic preflight returned the correct answer with zero reasoning
tokens. This uses native vLLM's hard token ban, rather than the Unsloth-specific
boolean logit-bias extension.

When the **native vLLM backend** is running, an OpenAI client can request this
control with `reasoning_effort="none"` and
`extra_body={"bad_words": ["<|content_thinking|>"],
"chat_template_kwargs": {"enable_thinking": False}}`.
Port 8004 now serves **native vLLM**, selected after this experiment; see the
[current native guide](../NATIVE.md). The DSpark backend restored at the end
of the experiment was subsequently stopped. Its different
[client controls](../dspark/THINKING.md) use nested effort zero and boolean
logit bias. The native recipe above applies only to the native backend.

Each arm receives a separate KV-cache namespace through `cache_salt`, with
different namespaces for factual questions and the module agent. Identical
synthetic warmup checks precede the factual questions in both arms. This
prevents a prior arm's prefix entries from warming the next arm. It does not
control unrelated concurrent device activity. The first preflight attempted
to discover a global cache-reset API; it stopped before measured generation
and restored DSpark when that route was absent from the API listing. That
preserved attempt is excluded from measured results.

The factual test replays all 48 frozen questions sequentially on fresh copies
of the corrected adaptive index. The module agent starts from the original
Wave Echo Cave transition question, using the same research policy, schema,
`top_k=15` and eight-search maximum. It chooses a new query only after observing
the preceding result. Reader outputs are capped at 4,096 tokens in both arms;
agent actions retain the 2,800-token cap. Capped factual responses count as
failures. Schema/protocol failures remain recorded rather than successful
answers.

The Qwen-ingested graph, summaries, 759 adaptive chunks and Ollama embeddings
are reused without ingestion. Automatic feedback stays disabled and each
question/search uses a fresh session. Original inputs and historical artifacts
are protected with hashes. An exact answer and reader-message match may reuse
the corresponding previous Codex source review; novel answers require review
against the frozen gold and supplied source context.

`run.py` controls the paired experiment from the Cognee VM. `operate.py` and
`monitor.py` run on Spark 2 to switch only Inkling and protect host headroom.
The native memory guard stops only that container if available RAM drops below
12 GiB. The Cognee worker has a 12 GiB VM memory limit. DSpark draft three is
restored afterward with its original launch command; Clef identity, image and
start time must remain unchanged. `analyze.py` verifies the paired requests,
source provenance, reviews and restoration before exporting public metrics.

Full traces remain under ignored
`experiments/adaptive-chunking/raw/pabtso/cognee-20261120/inkling-native-thinking-paired-20261122/`.
The earlier preflight is in `inkling-native-thinking-20261122/`. AtlasKV,
the earlier native results and the Unsloth DSpark measurements remain separate.
