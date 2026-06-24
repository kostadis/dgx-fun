"""Tests for the streaming SSE reassembly and idle-timeout resolution.

The transport itself (urlopen) isn't exercised here — only the pure pieces:
the SSE parser (:func:`dgxlib.client._consume_stream`) and the registry's new
``idle_timeout`` knob that the streaming socket budget reads from.
"""

import pytest

import dgxlib
from dgxlib import registry
from dgxlib.client import _consume_stream, _norm_finish_reason


@pytest.fixture(autouse=True)
def _clear_cache():
    registry.clear_registry_cache()
    yield
    registry.clear_registry_cache()


def _sse(*objs) -> list[bytes]:
    """Render dicts as `data: {...}` SSE byte-lines, terminated by [DONE]."""
    import json
    lines = [f"data: {json.dumps(o)}".encode() for o in objs]
    lines.append(b"data: [DONE]")
    return lines


# ── SSE reassembly ───────────────────────────────────────────────────────────

def test_consume_stream_concatenates_deltas():
    lines = _sse(
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": ", world"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    )
    text, fr = _consume_stream(iter(lines))
    assert text == "Hello, world"
    assert fr == "stop"


def test_consume_stream_captures_length_finish_reason():
    lines = _sse(
        {"choices": [{"delta": {"content": "partial"}}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
    )
    text, fr = _consume_stream(iter(lines))
    assert text == "partial"
    # vLLM's "length" must normalise to the shared truncation signal.
    assert _norm_finish_reason(fr) == "max_tokens"


def test_consume_stream_skips_keepalives_and_blank_lines():
    lines = [
        b": ping",
        b"",
        b"data: " + b'{"choices": [{"delta": {"content": "ok"}}]}',
        b"",
        b"data: [DONE]",
    ]
    text, _ = _consume_stream(iter(lines))
    assert text == "ok"


def test_consume_stream_stops_at_done_ignoring_trailing():
    lines = [
        b'data: {"choices": [{"delta": {"content": "kept"}}]}',
        b"data: [DONE]",
        b'data: {"choices": [{"delta": {"content": "DROPPED"}}]}',
    ]
    text, _ = _consume_stream(iter(lines))
    assert text == "kept"


def test_consume_stream_raises_on_mid_stream_error():
    lines = [
        b'data: {"choices": [{"delta": {"content": "partial"}}]}',
        b'data: {"error": {"message": "slot crashed"}}',
    ]
    with pytest.raises(RuntimeError, match="slot crashed"):
        _consume_stream(iter(lines))


def test_consume_stream_tolerates_unparseable_chunk():
    lines = [
        b"data: not-json",
        b'data: {"choices": [{"delta": {"content": "after"}}]}',
        b"data: [DONE]",
    ]
    text, _ = _consume_stream(iter(lines))
    assert text == "after"


# ── idle_timeout resolution ──────────────────────────────────────────────────

def test_idle_timeout_explicit_for_fast_slot():
    c = dgxlib.resolve_model_config("Qwen/Qwen3-Next-80B-A3B-Instruct-FP8")
    assert c.idle_timeout == 120.0
    assert c.read_timeout == 600.0  # legacy budget unchanged


def test_idle_timeout_defaults_to_read_timeout_when_unset(tmp_path):
    import textwrap
    p = tmp_path / "models.yaml"
    p.write_text(textwrap.dedent("""
        default: { can_think: false, read_timeout: 450 }
    """), encoding="utf-8")
    c = dgxlib.resolve_model_config("anything", registry_path=str(p))
    # No idle_timeout in yaml -> falls back to read_timeout (pure streaming
    # reinterpretation, no value regression).
    assert c.idle_timeout == 450.0
    assert c.read_timeout == 450.0


def test_idle_timeout_tolerant_for_queued_box():
    c = dgxlib.resolve_model_config("Qwen/Qwen3.5-122B-A10B-FP8")
    assert c.idle_timeout == 900.0
    assert c.read_timeout == 3600.0
