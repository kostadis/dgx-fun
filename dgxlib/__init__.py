"""dgxlib — encapsulates DGX Spark per-model behavior.

The single home for *how* each model served on the Spark wants to be called, so
swapping the served slot is a one-line edit to ``models.yaml`` rather than code
surgery in every caller.

- :func:`resolve_model_config` / :class:`ModelConfig` — the per-model request
  registry (thinking, read timeout, max_tokens), with per-call overrides.
- :func:`discover_model` — read the served model id from ``/v1/models``.
- :func:`make_client` / :func:`call_api` — a plain non-streaming client that
  applies the registry; the whole client mytools needs. CampaignGenerator imports
  only the registry/discovery layer and applies it in its own transport.
- :func:`call_api_full` — same call, but returns ``(text, finish_reason)`` for
  callers that need to detect ``max_tokens`` truncation (e.g. pdf-translators).
- :data:`RETRYABLE_STATUS` / :func:`is_retryable_status` — shared retry policy.
"""

from .discovery import DEFAULT_ENDPOINT, discover_model
from .registry import (
    ModelConfig,
    resolve_model_config,
    load_registry,
    clear_registry_cache,
)
from .client import DgxClient, make_client, call_api, call_api_full, _is_retryable
from .retry import RETRYABLE_STATUS, is_retryable_status

__all__ = [
    "DEFAULT_ENDPOINT",
    "discover_model",
    "ModelConfig",
    "resolve_model_config",
    "load_registry",
    "clear_registry_cache",
    "DgxClient",
    "make_client",
    "call_api",
    "call_api_full",
    "RETRYABLE_STATUS",
    "is_retryable_status",
]
