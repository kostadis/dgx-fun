# dgxlib

A small library that **owns DGX Spark per-model behavior** — the request knobs
each served model wants (thinking on/off, read timeout, max_tokens), plus
served-model-id discovery.

The point: swapping the model on the Spark should be a **one-line edit to
[`models.yaml`](models.yaml)**, not a code patch in every caller. The registry
lives next to the `spin-up-vllm-*.sh` scripts and `current-setup.md` that already
document the Spark, so you edit it in the same breath as a model swap.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the design and rationale.

## Install

Editable-installed into the shared venv (`~/.venvs/main`):

```bash
pip install -e ~/src/dgx
```

Both consumers — mytools (`pdf_enricher.py`) and CampaignGenerator
(`campaignlib/api/backends.py`) — import it from that venv.

## The registry (`models.yaml`)

Per-model request behavior, keyed by the exact model id vLLM advertises
(`GET /v1/models`). Thinking is **`(model capability) × (call intent)`**: the
registry stores capability and a default; the *call site* decides intent.

```yaml
default:        { can_think: false, thinking_default: false, read_timeout: 300, max_tokens: 16384 }
models:
  "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8":
    can_think: true            # reasoning-capable
    thinking_default: false    # off unless the call asks
    read_timeout: 600
  "Qwen/Qwen2.5-14B-Instruct-AWQ":
    can_think: false           # no thinking template → forced off
match:                         # prefix fallbacks (longest wins) for unlisted ids
  "Qwen/Qwen3": { can_think: true, thinking_default: false, read_timeout: 600 }
```

Resolution order: exact `models` entry → longest `match` prefix → `default`.

**Swapping a model?** Add/edit its row here in the *same change* that runs the
spin-up script and updates `current-setup.md`. That's the whole payoff.

## Usage

### As a behavior layer (CampaignGenerator-style)

Resolve config where the model id and the per-call intent are known, and apply it
in your own transport:

```python
import dgxlib

cfg = dgxlib.resolve_model_config(
    "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8", thinking=True)
# cfg.extra_body  -> {"chat_template_kwargs": {"enable_thinking": True}}
# cfg.read_timeout, cfg.max_tokens

served = dgxlib.discover_model("http://192.168.1.147:8001/v1")  # read id from /v1/models
```

`thinking`: `None` uses the model's `thinking_default`; `True`/`False` overrides
it — honored only when the model is `can_think`, otherwise forced off.

### As a client (mytools-style)

The whole client, registry already applied inside `call_api`:

```python
import dgxlib as llm

model = llm.discover_model(llm.DEFAULT_ENDPOINT)
client = llm.make_client()
text = llm.call_api(client, system, content, model, thinking=True)
```

## Environment overrides (back-compat)

- `DGXLIB_REGISTRY` — path to an alternate `models.yaml` (else the bundled one).
- `DGX_NO_THINKING` — forces thinking off when the caller didn't specify.
- Consumers keep their own `DGX_*` overrides (e.g. CG's `DGX_READ_TIMEOUT`,
  `DGX_MODEL`) layered on top of the registry result.

## Tests

```bash
~/.venvs/main/bin/python -m pytest ~/src/dgx/tests/test_registry.py
```
