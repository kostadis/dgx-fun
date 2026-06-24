"""Per-model DGX request behavior: the registry and its resolver.

Encapsulates what each model served on the Spark wants in a request, so that
swapping the served slot is a one-line edit to ``models.yaml`` rather than code
surgery in every caller. See that file for the schema.

The key idea is that *thinking* is ``(model capability) x (call intent)``:
the registry stores per-model capability and a default; the call site supplies
an optional override. :func:`resolve_model_config` composes the two.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_BUNDLED_REGISTRY = Path(__file__).resolve().parent / "models.yaml"

# Cache parsed registries by absolute path so repeated calls don't re-read disk.
_REGISTRY_CACHE: dict[str, dict] = {}


@dataclass
class ModelConfig:
    """Resolved request behavior for one (model, call-intent) pair.

    ``extra_body`` is merged into the OpenAI-compatible request body (vLLM reads
    ``chat_template_kwargs``); the timeouts and ``max_tokens`` are applied by the
    caller's transport.

    Two timeout knobs, because the streaming client measures *progress*, not
    total wall-clock:

    - ``read_timeout`` — legacy total-response budget. Retained for back-compat
      and as the default source for ``idle_timeout``.
    - ``idle_timeout`` — the streaming socket budget: the max seconds to wait for
      the *next* token (and for the first token / queue admission). A healthy
      generation streams continuously and never trips it however slow it is
      overall; a stalled or over-queued slot produces no bytes and trips it fast.
      Defaults to ``read_timeout`` when a model doesn't set it, so switching to
      streaming changes the *meaning* of the existing budget (total → inter-token)
      without changing its value. Set a smaller value to fail dead slots faster.
    """

    extra_body: dict = field(default_factory=dict)
    read_timeout: float = 300.0
    idle_timeout: float = 300.0
    max_tokens: int | None = 16384


def _registry_path(registry_path: str | None) -> Path:
    """Resolve which registry file to read.

    Precedence: explicit ``registry_path`` arg → ``DGXLIB_REGISTRY`` env var →
    the bundled ``models.yaml``. The env override lets dgx-fun keep an
    authoritative copy at the repo root later without any code change.
    """
    if registry_path:
        return Path(registry_path)
    env = os.environ.get("DGXLIB_REGISTRY")
    if env:
        return Path(env)
    return _BUNDLED_REGISTRY


def load_registry(registry_path: str | None = None) -> dict:
    """Load and cache the parsed registry mapping."""
    path = _registry_path(registry_path).resolve()
    key = str(path)
    if key not in _REGISTRY_CACHE:
        with open(path, encoding="utf-8") as fh:
            _REGISTRY_CACHE[key] = yaml.safe_load(fh) or {}
    return _REGISTRY_CACHE[key]


def clear_registry_cache() -> None:
    """Drop cached registries (test helper / after editing models.yaml)."""
    _REGISTRY_CACHE.clear()


def _lookup(model_id: str, reg: dict) -> dict:
    """Return the merged settings for ``model_id``: exact → longest prefix → default."""
    settings = dict(reg.get("default") or {})
    exact = (reg.get("models") or {}).get(model_id)
    if exact is not None:
        settings.update(exact)
        return settings
    # No exact hit: take the longest matching prefix from `match`.
    best_key = ""
    best_val: dict | None = None
    for prefix, val in (reg.get("match") or {}).items():
        if model_id.startswith(prefix) and len(prefix) > len(best_key):
            best_key, best_val = prefix, val
    if best_val is not None:
        settings.update(best_val)
    return settings


def resolve_model_config(
    model_id: str,
    *,
    thinking: bool | None = None,
    max_tokens: int | None = None,
    registry_path: str | None = None,
) -> ModelConfig:
    """Resolve request behavior for ``model_id`` and an optional call intent.

    ``thinking``: ``None`` uses the model's ``thinking_default``; ``True``/``False``
    overrides it — but only takes effect when the model's ``can_think`` is true,
    otherwise thinking is forced off (a non-reasoning slot cannot honor it).

    ``max_tokens``: ``None`` uses the model/default value; otherwise overrides it.
    """
    reg = load_registry(registry_path)
    s = _lookup(model_id, reg)

    can_think = bool(s.get("can_think", False))
    if not can_think:
        effective_thinking = False
    elif thinking is not None:
        effective_thinking = thinking
    else:
        effective_thinking = bool(s.get("thinking_default", False))

    extra_body: dict = {}
    if can_think:
        # Only emit the knob for models that understand it; a non-reasoning
        # template would reject (or ignore) chat_template_kwargs.
        extra_body["chat_template_kwargs"] = {"enable_thinking": effective_thinking}

    read_timeout = float(s.get("read_timeout", 300))
    # idle_timeout defaults to read_timeout: with streaming, the same number now
    # means "no token for this long" instead of "no full response for this long",
    # so an unset model keeps its old budget but stops killing slow-but-working
    # generations. A model sets idle_timeout explicitly to fail dead slots faster.
    idle_timeout = float(s.get("idle_timeout", read_timeout))
    return ModelConfig(
        extra_body=extra_body,
        read_timeout=read_timeout,
        idle_timeout=idle_timeout,
        max_tokens=max_tokens if max_tokens is not None else s.get("max_tokens", 16384),
    )
