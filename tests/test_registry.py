"""Tests for the dgxlib per-model behavior registry resolver."""

import textwrap

import pytest

import dgxlib
from dgxlib import registry


@pytest.fixture(autouse=True)
def _clear_cache():
    registry.clear_registry_cache()
    yield
    registry.clear_registry_cache()


def _write_registry(tmp_path, body: str):
    p = tmp_path / "models.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(p)


# ── bundled registry: capability x intent ────────────────────────────────────

def test_thinking_capable_default_off():
    c = dgxlib.resolve_model_config("Qwen/Qwen3-Next-80B-A3B-Instruct-FP8")
    assert c.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert c.read_timeout == 600.0
    assert c.max_tokens == 16384


def test_thinking_capable_call_override_on():
    c = dgxlib.resolve_model_config(
        "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8", thinking=True)
    assert c.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_non_reasoning_model_forces_thinking_off():
    # can_think: false → caller's thinking=True is ignored, no knob emitted.
    c = dgxlib.resolve_model_config("Qwen/Qwen2.5-14B-Instruct-AWQ", thinking=True)
    assert c.extra_body == {}


def test_longest_prefix_match():
    c = dgxlib.resolve_model_config("Qwen/Qwen3-Coder-Next")
    assert c.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert c.read_timeout == 600.0


def test_unknown_model_falls_to_default():
    c = dgxlib.resolve_model_config("some/unknown-model")
    assert c.extra_body == {}            # default can_think is false
    assert c.read_timeout == 300.0
    assert c.max_tokens == 16384


def test_per_call_max_tokens_override():
    c = dgxlib.resolve_model_config("some/unknown-model", max_tokens=2048)
    assert c.max_tokens == 2048


# ── registry resolution order / overrides ────────────────────────────────────

def test_exact_wins_over_prefix(tmp_path):
    path = _write_registry(tmp_path, """
        default: { can_think: false, read_timeout: 300, max_tokens: 100 }
        models:
          "fam/model-exact": { can_think: true, thinking_default: true, read_timeout: 999 }
        match:
          "fam/": { can_think: true, thinking_default: false, read_timeout: 111 }
    """)
    c = dgxlib.resolve_model_config("fam/model-exact", registry_path=path)
    assert c.read_timeout == 999.0
    assert c.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_longest_prefix_wins(tmp_path):
    path = _write_registry(tmp_path, """
        default: { can_think: false }
        match:
          "fam/": { can_think: false, read_timeout: 111 }
          "fam/big": { can_think: true, thinking_default: false, read_timeout: 222 }
    """)
    c = dgxlib.resolve_model_config("fam/big-model", registry_path=path)
    assert c.read_timeout == 222.0
    assert c.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_dgxlib_registry_env_override(tmp_path, monkeypatch):
    path = _write_registry(tmp_path, """
        default: { can_think: true, thinking_default: true, read_timeout: 42, max_tokens: 7 }
    """)
    monkeypatch.setenv("DGXLIB_REGISTRY", path)
    registry.clear_registry_cache()
    c = dgxlib.resolve_model_config("anything")
    assert c.read_timeout == 42.0
    assert c.max_tokens == 7
    assert c.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


# ── retry policy ─────────────────────────────────────────────────────────────

def test_retry_policy():
    assert dgxlib.is_retryable_status(503) is True
    assert dgxlib.is_retryable_status(529) is True
    assert dgxlib.is_retryable_status(400) is False
    assert dgxlib.is_retryable_status(None) is False
