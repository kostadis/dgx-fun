# dgxlib — architecture

## Problem

Swapping the model served on the Spark used to force **code surgery in multiple
places**. The trigger was a surgical edit to add a `DGX_NO_THINKING` hack to one
caller — and the realization that every model swap meant repeating that kind of
hack. The same stopgap existed, independently, in two repos, and the same unbuilt
feature was tracked twice (mytools #48, CampaignGenerator #94).

`dgxlib` makes one library **own DGX behaviors** so a model swap is a one-line
registry edit, not a code change.

## Principle: shared behavior layer, per-transport clients

The two consumers talk to the *same* Spark endpoint but need *different
transports*, and that difference is load-bearing:

| | mytools (`rpg-lib/pdf_enricher.py`) | CampaignGenerator (`campaignlib/api/backends.py`) |
|---|---|---|
| Surface | plain functions mirroring a `claudelib` | an **anthropic-SDK facade** (`client.messages.stream(...).text_stream`) |
| Deps | stdlib `urllib` only | `openai` SDK + `httpx` |
| Streaming | yes (SSE; the read timeout is an inter-token *idle* budget) | yes (its whole pipeline depends on it) |
| Uses dgxlib as | **the whole client** | **only the behavior layer**, applied in its own client |

So the split is: **shared = the behavior layer (registry + discovery) + a plain
client; per-repo = the transport.** Forcing both onto one client surface would
either saddle mytools with `openai`+`httpx` it doesn't need, or strip CG of the
streaming anthropic shape its pipeline is built on.

## Thinking is `(model capability) × (call intent)`

Thinking is **not** a fixed property of a model. A reasoning-capable model *can*
think; whether you *want* it to is a per-call decision (quick extraction → no;
deep synthesis → yes). A non-reasoning model *can't*, so the choice doesn't
exist there.

Therefore:

- The **registry** (model-keyed) stores capability + a default: `can_think`,
  `thinking_default`.
- The **call site** supplies intent: `resolve_model_config(model, thinking=...)`.
- The resolver **composes** them: capability gates, call intent chooses, model
  default is the fallback. A `True` from the caller is ignored for a
  non-reasoning model (forced off), and the `enable_thinking` knob is emitted
  *only* for `can_think` models (a non-reasoning template would not understand
  it).

## Modules

```
dgxlib/
  registry.py    ModelConfig + resolve_model_config + load_registry  (the behavior layer)
  models.yaml    the per-model registry data (bundled default; package_data)
  discovery.py   discover_model() — read the served id from /v1/models
  client.py      DgxClient / make_client / call_api — plain stdlib client; applies the registry
  retry.py       RETRYABLE_STATUS + is_retryable_status — shared retry *policy*
  __init__.py    public surface
```

## Registry resolution

`resolve_model_config(model_id, *, thinking=None, max_tokens=None, registry_path=None)`:

1. **Lookup** the model's settings: exact `models[model_id]` → longest matching
   `match` prefix → `default`. Settings merge onto `default`.
2. **Thinking**: if not `can_think` → off; else `thinking` if the caller passed
   one, else `thinking_default`. The effective bool becomes
   `extra_body["chat_template_kwargs"]["enable_thinking"]` — emitted only when
   `can_think`.
3. **Timeout / tokens**: `read_timeout` from settings; `max_tokens` from the
   per-call override else settings.

Registry source precedence: explicit `registry_path` arg → `DGXLIB_REGISTRY` env
→ bundled `models.yaml`. Parsed registries are cached by path.

## Retry: shared policy, per-transport predicate

The retryable *predicate* is transport-specific — the stdlib client matches
`urllib` errors; CampaignGenerator matches `openai`/`httpx` ones — so it cannot
be shared. What is shared is the **policy**: which HTTP statuses are worth
retrying (`RETRYABLE_STATUS = {500, 502, 503, 529}` — transient backend errors +
overload). Each transport maps its own exceptions onto that set.

## Relationship to `current-setup.md`

`current-setup.md` is the human-readable record of what's running on the Spark;
`models.yaml` is its **machine-readable, behavior-only sibling**. When the
`vllm-chat` slot changes model, both must change in the same edit — the spin-up
script serves the model, `current-setup.md` records it, and `models.yaml` says
how to call it. Keeping them in lockstep is what makes a swap a one-liner instead
of a debugging session.

## Back-compat

The per-repo `DGX_*` env vars (`DGX_NO_THINKING`, and CG's `DGX_READ_TIMEOUT` /
`DGX_MODEL`) are retained as **overrides** layered on top of the registry result,
so existing invocations behave unchanged. They are no longer the *source* of the
config — the registry is.

## Deferred / non-goals

- **Endpoint/host resolution** (dgx-fun #19): the endpoint stays env
  (`DGX_ENDPOINT`) + a default constant. `current-setup.md` is prose, not
  machine-readable; a structured endpoint source is future work, not parsed here.
- **Per-agent profiles**: only per-model + per-call intent today. Named profiles
  ("extraction" vs "synthesis") are a plausible later generalization of the
  per-call axis, not built yet.
- Tool use and vision stay in the consumers' transports — out of scope for the
  shared layer. The plain client now streams (SSE) and owns its own retry
  predicate (`_is_retryable`: idle timeouts non-retryable, connection errors
  retryable); CG still supplies its own transport-level streaming and retry.
