"""Plain (stdlib, non-streaming) OpenAI-compatible client for the DGX Spark.

Mirrors the surface of a ``claudelib`` (``make_client`` / ``call_api``) so a
caller can pick a provider via dispatch. This is the *whole* client mytools
needs; CampaignGenerator instead imports only the behavior layer
(:mod:`dgxlib.registry`, :mod:`dgxlib.discovery`) and applies it inside its own
streaming anthropic-facade transport.

``call_api`` applies the per-model registry (:func:`dgxlib.resolve_model_config`)
so request knobs — thinking, read timeout, max_tokens — come from ``models.yaml``,
not inline hacks. Thinking is a *per-call* decision: pass ``thinking=True`` for a
reasoning-capable slot, leave it ``None`` to use the model's default.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .discovery import DEFAULT_ENDPOINT
from .registry import resolve_model_config


class DgxClient:
    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, timeout: float | None = None):
        self.endpoint = endpoint.rstrip("/")
        # None → use the per-model read_timeout from the registry. An explicit
        # value overrides the registry (back-compat for callers that set one).
        self.timeout = timeout


def make_client(endpoint: str = DEFAULT_ENDPOINT, timeout: float | None = None) -> DgxClient:
    return DgxClient(endpoint, timeout)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return 500 <= exc.code < 600
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return False


def call_api(
    client: DgxClient,
    system: str,
    content,
    model: str,
    max_tokens: int | None = None,
    *,
    thinking: bool | None = None,
) -> str:
    """Non-streaming chat completion. Returns ``choices[0].message.content``.

    Retries transient connection errors (the Spark warm-restart takes several
    minutes; tolerate it). Per-model request behavior — thinking, read timeout,
    max_tokens — is resolved from the registry; ``thinking``/``max_tokens`` here
    override it per call.
    """
    # Back-compat: DGX_NO_THINKING (when the caller didn't ask) forces thinking off.
    if thinking is None and os.environ.get("DGX_NO_THINKING"):
        thinking = False

    cfg = resolve_model_config(model, thinking=thinking, max_tokens=max_tokens)
    request_timeout = client.timeout if client.timeout is not None else cfg.read_timeout

    url = client.endpoint + "/chat/completions"
    payload = {
        "model": model,
        "max_tokens": cfg.max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    }
    payload.update(cfg.extra_body)  # e.g. {"chat_template_kwargs": {"enable_thinking": ...}}
    body = json.dumps(payload).encode("utf-8")

    delays = [10, 20, 40]
    for attempt, delay in enumerate([-1] + delays):
        if delay >= 0:
            print(f"\n  [DGX unavailable — waiting {delay}s before retry "
                  f"{attempt}/{len(delays)}...]", flush=True)
            time.sleep(delay)
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                resp_payload = json.loads(resp.read().decode("utf-8"))
            return resp_payload["choices"][0]["message"]["content"]
        except Exception as e:
            if _is_retryable(e) and attempt < len(delays):
                continue
            raise
