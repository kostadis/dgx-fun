# Inkling 2 Mini on Spark 2

> **Selected service, November 22, 2026 (Pacific):** port 8004 serves the
> retained **native vLLM Inkling**, model `inkling-2-mini`, alongside Clef on
> port 8002. DSpark and Hibana are stopped. [Native client and restart guide](NATIVE.md).
> Use the native per-request thinking-token ban for the measured no-thinking
> behavior; effort `none` alone still permits thinking.
> The [paired native test](native-thinking/RESULTS.md) reduced factual p50
> **10.38 → 4.51 s** on the same checkpoint and process. Complete answers stayed
> 44/47; fully grounded complete answers fell 44/47 → 43/47. DSpark was restored
> at that experiment's end, then replaced by native vLLM at the user's request.
> The [DSpark results](dspark/RESULTS.md) and [GGUF Cognee follow-up](dspark/COGNEE.md)
> remain separate historical measurements.

The original isolated installation uses the
[Thinking Machines checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
with the ARM64 image from the
[official vLLM Spark recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini).
The [release overview](https://huggingface.co/blog/inkling2mini) describes the
fixed-loop model. This trial does not change its loop count or requantize it.

The launcher creates only `inkling2mini` on Spark 2, port 8004, with its own
cache and runtime directories under `~/inkling-trial`. Existing Qwen and Clef
containers are retained. It checks available host memory and refuses to
replace a pre-existing Inkling container automatically.

The model revision is `b85c3cfa615fc413cc9af3379f491b963f3b586a`. The image is
pinned to ARM64 manifest
`sha256:ecdf0ccc663d86332459f5a50a38c30861a5079bc3924aa08fdc657ef10dd648`.
Main weights are 19,526,019,168 bytes, SHA-256
`65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`.
The bundled draft weighs 635,801,416 bytes, SHA-256
`7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`.
Both downloads were verified; speculative decoding is disabled for this trial.

Copy `start-spark.sh` to Spark 2 and execute it there. Its defaults use a
65,536-token context, one active request, a 0.26 GPU-memory fraction, eager
execution and text-only input. These deliberately differ from the dedicated
Spark recipe to fit alongside the existing Clef service. The container has a
44 GiB CPU-memory limit and no swap allowance beyond that limit; host unified
memory still needs monitoring because GPU allocations are separate.

```bash
ssh spark2 'mkdir -p ~/inkling-trial'
scp experiments/inkling/start-spark.sh spark2:~/inkling-trial/start-spark.sh
ssh spark2 'bash ~/inkling-trial/start-spark.sh'
ssh spark2 'docker logs --tail 40 inkling2mini'
```

The endpoint is OpenAI-compatible, with base URL
`http://192.168.1.121:8004/v1` and model `inkling-2-mini`. No cloud API key is
used. For the tested no-thinking behavior, supply `reasoning_effort="none"`
and the native thinking-token ban shown below. Effort alone is not a guarantee
of zero reasoning tokens. The full [native client guide](NATIVE.md) includes
Cognee options and operations instructions.

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.1.121:8004/v1", api_key="local")
response = client.chat.completions.create(
    model="inkling-2-mini",
    messages=[{"role": "user", "content": "Explain the Phandelver Pact briefly."}],
    reasoning_effort="none",
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
        "bad_words": ["<|content_thinking|>"],
    },
    temperature=0,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

The Cognee comparison reuses the saved corrected adaptive index and Ollama
embeddings. It changes both the reader and research agent to Inkling; graph
ingestion remains from Qwen. Use a fresh output directory:

```bash
trial=experiments/adaptive-chunking/raw/pabtso/cognee-20261120
/home/kostadis/cognee-local/.venv/bin/python \
  experiments/adaptive-chunking/cognee/sequential.py \
  --source "$trial/adaptive-context-8" --output "$trial/inkling-new/adaptive" \
  --question-file "$trial/sonnet-sequential-20261121/question.json" \
  --provider openai --backend http://192.168.1.121:8004 \
  --model inkling-2-mini --reasoning-effort none \
  --extra-body-file experiments/inkling/no-thinking-options.json \
  --action-format json_schema --top-k 15 --max-searches 8
```

Installation, first-load kernel preparation and smoke-test time are separate
from the measured research loop. The device is shared and eager execution is
enabled, so its latency is a configuration observation rather than a measure
of the model's best achievable decode speed. This trial covers text retrieval;
it does not test image input, native tool calls, or speculative decoding.

The first same-policy [Cognee trial](../adaptive-chunking/cognee/inkling-sequential-results.json)
ended at the action-schema guard after two searches and six model calls. It
used 30,429 tokens over 137.94 seconds until failure, including 1,946 reasoning
tokens within its 3,447 output tokens. All HTTP requests succeeded and stopped
naturally. Inkling selected a new question inside `args: [{query: ...}]` but
omitted the required top-level query, decision summary and gap list three times.
No final answer was produced; elapsed time is not successful-answer latency.
The text and JSON smoke tests passed. Native tool calling and final narrative
quality were not assessed by this failed control-loop run.

Inkling remains up after the trial. Clef's container identity and start time
were unchanged, and both Clef and Spark 1 Qwen remained healthy. Observed host
memory available fell from about 54 GiB before installation to 21 GiB afterward;
swap use increased from about 0.65 to 2.2 GiB across installation and testing.
These are shared-host observations, not isolated model-footprint measurements.
The full local trace is under ignored
`experiments/adaptive-chunking/raw/pabtso/cognee-20261120/inkling-sequential-20261121/`.
Use `docker stop inkling2mini` and `docker start inkling2mini` on Spark 2 to
stop or restart only this model; its checkpoint cache is retained.

## Schema-guided action workaround

Use `--action-format json_schema` with the sequential runner. Plain
`json_object` mode guarantees JSON syntax but does not enforce the harness's
field names or types. The schema requires the flat `action`, `query`,
`decision_summary`, `remaining_gaps` and `answer` fields, and forbids extra
properties such as `args`. The original semantic validator remains in place.
This is a per-request setting; no server restart is needed.
[vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/).

```bash
trial=experiments/adaptive-chunking/raw/pabtso/cognee-20261120
/home/kostadis/cognee-local/.venv/bin/python \
  experiments/adaptive-chunking/cognee/sequential.py \
  --source "$trial/adaptive-context-8" --output "$trial/inkling-schema-new/adaptive" \
  --question-file "$trial/sonnet-sequential-20261121/question.json" \
  --provider openai --backend http://192.168.1.121:8004 \
  --model inkling-2-mini --reasoning-effort none \
  --action-format json_schema --top-k 15 --max-searches 8
```

The exact previously rejected decision prompt was replayed with only its
response format changed. The schema-guided reply passed the original action
validator, selected a new query, and supplied the missing rationale and gap
fields. It took 7.38 seconds and 5,497 tokens including 45 reasoning tokens.
That diagnostic is separate from the fresh research run. Schema compliance
does not establish correct retrieval, grounding or a good stopping decision.

The [full schema-guided rerun](../adaptive-chunking/cognee/inkling-schema-sequential-results.json)
then completed autonomously after **seven searches and fourteen model calls**,
with **zero protocol corrections**. It took 499.32 seconds including worker
startup, or 494.64 seconds for the research loop, and used 104,978 tokens
(90,787 input, 14,191 output, including 8,589 reasoning tokens). The source
union reached 31 passages. Search 7 recovered the actual Chapter 4 ending,
and the final answer correctly used the Nundro homecoming, mine stake and
return-to-Phandalin hook into Chapter 5. All 38 contextual source appearances
and all seven agent source unions passed the request audit; graph counts and
protected original artifacts remained unchanged.

Source review still found overgeneralized Chapter 6 geography and ambiguous
wording about a goblin-body note: the passage is GM narration, not a physical
message. The exact player-knowledge qualifier was absent from the retrieved
union. The final answer was preserved unchanged. This demonstrates a working
format workaround, with separate grounding caveats. The baseline failure
remains recorded, and prefix-cache state/server load were not matched for a
speed comparison. Full answers and traces are under ignored
`raw/pabtso/cognee-20261120/inkling-schema-sequential-20261121/`.

## Original factual-question test

The [48-question replay](../adaptive-chunking/cognee/inkling-question-results.json)
uses the same corrected adaptive Cognee index, with one retrieval and answer
per question. All 48 actual reader messages matched Qwen's exactly. Inkling
took **12.10 s p50 / 18.60 s p95**, versus **4.68 s / 6.16 s** for Qwen.
The full query sum was **10 min 36 s**, versus **3 min 47 s**. Strict source
review scored 44/47; a lenient reading of one misleading sentence gives 45/47,
equal to Qwen. No new complete answer was recovered. The two known retrieval
gaps remained, and all six item questions passed.

No action schema or agent was involved. Inkling reported 11,497 reasoning
tokens despite `reasoning_effort:"none"`. These timings compare the deployed
configurations: Qwen has MTP-2 and prefix caching, whereas this Inkling service
uses eager execution without speculation alongside Clef. The
[question-only instructions](../adaptive-chunking/cognee/README.md#factual-questions-with-a-different-reader-model)
replay the frozen questions on a fresh index copy without reingestion.

A [retrospective reasoning audit](dspark/native-reasoning-audit-20261122.json)
found that reasoning made up **78.35%** of native Inkling's factual output and
**60.52%** of its completed agent output. Its exact pinned vLLM image correctly
renders effort `none` as zero; it did not have the subsequent Unsloth mapping
bug. Those historical native timings include generated thinking. The subsequent
[same-runtime paired test](native-thinking/RESULTS.md) measured p50 **10.38 s
with thinking / 4.51 s without**, and 48-query sums **564.26 / 222.25 s**.
Both scored 44/47 complete; the off arm added one unsupported claim, leaving
43/47 fully grounded complete answers versus 44/47. The faster agent chose
less search work, and both fresh agent answers had source errors.

A subsequent [Hibana fast comparison](../../phandelver-cognee-chunking-results.md#hibana-fast-original-questions-and-module-synthesis-2026-11-21)
completed the same 48 questions at 3.41 s p50 / 4.32 s p95, with 44/47 complete
answers and an added Agatha contradiction. Its five-search narrative agent
took 84.51 s versus Inkling's 499.32 s for seven searches; the Hibana answer
still confused GM information with a party-discovered transition. Search work
and answer quality differed. All three factual readers received identical
messages; the full report compares their deployed configurations and quality.
