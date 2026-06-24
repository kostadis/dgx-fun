"""Plain (stdlib, streaming) OpenAI-compatible client for the DGX Spark.

Mirrors the surface of a ``claudelib`` (``make_client`` / ``call_api``) so a
caller can pick a provider via dispatch. This is the *whole* client mytools
needs; CampaignGenerator instead imports only the behavior layer
(:mod:`dgxlib.registry`, :mod:`dgxlib.discovery`) and applies it inside its own
streaming anthropic-facade transport.

``call_api`` applies the per-model registry (:func:`dgxlib.resolve_model_config`)
so request knobs — thinking, timeouts, max_tokens — come from ``models.yaml``,
not inline hacks. Thinking is a *per-call* decision: pass ``thinking=True`` for a
reasoning-capable slot, leave it ``None`` to use the model's default.

The request is **streamed** (Server-Sent Events). That turns the socket timeout
into an *inactivity* budget — the max gap between tokens (``idle_timeout`` in the
registry) — rather than a cap on the whole response. A slot that's wedged or
stuck behind a saturated queue produces no bytes and is failed within one idle
budget; a legitimately slow generation that keeps emitting tokens runs to
completion (bounded only by ``max_tokens``). The non-streaming client this
replaced had to wait the *entire* budget for both cases — so a dead chunk and a
slow-but-fine chunk were indistinguishable and both cost the full timeout.
"""

from __future__ import annotations

import json
import os
import socket
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
    # An *idle timeout* (no token streamed within the budget) is NOT retryable:
    # the request was accepted and the slot is wedged or hopelessly queued, so
    # re-sending the identical request just stalls again (3 retries = 3 more
    # idle budgets of wasted wall-clock). Fail fast and let the caller
    # skip/resume that unit of work. Connection-level failures (refused / DNS /
    # reset) ARE retryable — those are a box that's down or warm-restarting,
    # which recovers within minutes.
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return False
    if isinstance(exc, urllib.error.HTTPError):
        return 500 <= exc.code < 600
    if isinstance(exc, urllib.error.URLError):
        # urllib wraps the real cause in .reason; a wrapped timeout is still a
        # read timeout (non-retryable), anything else is a transport error.
        return not isinstance(exc.reason, (TimeoutError, socket.timeout))
    if isinstance(exc, ConnectionError):
        return True
    return False


def _describe_error(exc: BaseException, timeout: float | None) -> str:
    """Human-readable, failure-mode-specific summary for the retry log.

    The retry banner used to say only "DGX unavailable" for *every* error, so a
    read timeout (the request was sent, generation ran past the budget) and a
    connection refusal (the endpoint is down / wrong host) were
    indistinguishable. This names which one happened so the log is diagnosable.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code} {exc.reason}"
    # urllib wraps the real cause in URLError.reason; unwrap to classify it.
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, (socket.timeout, TimeoutError)):
        budget = f" after {timeout:g}s" if timeout else ""
        return (f"idle timeout{budget} — endpoint reachable but streamed no "
                f"token in time (slot wedged or queued behind a saturated box)")
    if isinstance(reason, ConnectionRefusedError):
        return "connection refused — endpoint down or wrong host/port"
    if isinstance(reason, ConnectionError):
        return f"connection error: {reason}"
    return f"{type(reason).__name__}: {reason}"


def _consume_stream(lines) -> tuple[str, str | None]:
    """Reassemble an OpenAI SSE token stream into ``(text, finish_reason)``.

    ``lines`` is any iterator of raw ``bytes`` lines (an ``http.client``
    response is one). We accumulate ``choices[0].delta.content`` across chunks
    and keep the last non-null ``finish_reason``. Blank lines and SSE comment
    lines (``: ping`` keepalives) are skipped; the stream ends at ``data: [DONE]``.

    A ``{"error": ...}`` payload mid-stream (vLLM surfaces some failures this way
    after a 200) is raised as a ``RuntimeError`` so the caller's retry/skip logic
    sees a failure rather than silently returning a truncated answer.

    Reading from ``lines`` is what blocks on the socket, so the caller's idle
    timeout fires *here* — between tokens — exactly when the slot stops emitting.
    """
    parts: list[str] = []
    finish_reason: str | None = None
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line or line.startswith(":"):
            continue  # keepalive / SSE comment
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if chunk.get("error"):
            raise RuntimeError(f"DGX stream error: {chunk['error']}")
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        piece = (choice.get("delta") or {}).get("content")
        if piece:
            parts.append(piece)
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
    return "".join(parts), finish_reason


def _norm_finish_reason(finish_reason: str | None) -> str:
    """Normalise an OpenAI ``finish_reason`` to the shared two-value vocabulary.

    vLLM reports ``"length"`` when the response was truncated by the
    ``max_tokens`` cap; everything else (``"stop"``, ``"tool_calls"``, ...)
    collapses to ``"stop"``. Mirrors claudelib's ``_norm_stop_reason`` so a
    provider-agnostic caller sees ``"max_tokens"`` for truncation from either
    backend.
    """
    return "max_tokens" if finish_reason == "length" else "stop"


def call_api_full(
    client: DgxClient,
    system: str,
    content,
    model: str,
    max_tokens: int | None = None,
    *,
    thinking: bool | None = None,
) -> tuple[str, str]:
    """Streaming chat completion. Returns ``(text, finish_reason)``.

    ``text`` is the concatenated ``choices[0].delta.content`` stream; the socket
    timeout (the model's ``idle_timeout``) is the max gap *between* tokens, so a
    wedged/over-queued slot fails within one idle budget while a slow-but-steady
    generation runs to completion. ``finish_reason`` is normalised to the shared
    vocabulary (``"max_tokens"`` for a length-truncated response, else ``"stop"``
    — see :func:`_norm_finish_reason`). Callers that need to detect truncation use
    the second element; :func:`call_api` discards it.

    Retries transient connection errors (the Spark warm-restart takes several
    minutes; tolerate it) but NOT idle timeouts (see :func:`_is_retryable`).
    Per-model request behavior — thinking, idle timeout, max_tokens — is resolved
    from the registry; ``thinking``/``max_tokens`` here override it per call.
    """
    # Back-compat: DGX_NO_THINKING (when the caller didn't ask) forces thinking off.
    if thinking is None and os.environ.get("DGX_NO_THINKING"):
        thinking = False

    cfg = resolve_model_config(model, thinking=thinking, max_tokens=max_tokens)
    # An explicit client.timeout still wins (back-compat); otherwise use the
    # per-model inactivity budget, which is what streaming makes meaningful.
    idle_timeout = client.timeout if client.timeout is not None else cfg.idle_timeout

    url = client.endpoint + "/chat/completions"
    payload = {
        "model": model,
        "max_tokens": cfg.max_tokens,
        "stream": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    }
    payload.update(cfg.extra_body)  # e.g. {"chat_template_kwargs": {"enable_thinking": ...}}
    body = json.dumps(payload).encode("utf-8")

    delays = [10, 20, 40]
    for attempt in range(len(delays) + 1):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json",
                         "Accept": "text/event-stream"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=idle_timeout) as resp:
                text, finish_reason = _consume_stream(resp)
            return text, _norm_finish_reason(finish_reason)
        except Exception as e:
            desc = _describe_error(e, idle_timeout)
            if _is_retryable(e) and attempt < len(delays):
                delay = delays[attempt]
                print(f"\n  [DGX {desc} — waiting {delay}s before retry "
                      f"{attempt + 1}/{len(delays)}...]", flush=True)
                time.sleep(delay)
                continue
            if attempt > 0:
                print(f"\n  [DGX {desc} — giving up after {attempt} "
                      f"retr{'y' if attempt == 1 else 'ies'}]", flush=True)
            raise


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

    Thin wrapper over :func:`call_api_full` that drops the finish-reason — kept
    for callers that only want the text (e.g. CampaignGenerator, pdf_enricher).
    """
    text, _ = call_api_full(client, system, content, model, max_tokens,
                            thinking=thinking)
    return text
