"""Served-model-id discovery for the DGX Spark vLLM endpoint.

vLLM serves a single chat model per container; that model's id is what every
request must use or the server returns 400. The served id changes when the slot
is swapped (Gemma, Nemotron, Llama, Qwen, ...), so callers read it from
``/v1/models`` rather than hard-coding it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

# Default Spark chat endpoint. See current-setup.md for the live slot/IP; the IP
# (not a hostname) is deliberate — hostnames don't resolve in every shell (WSL2,
# containers, cron). Structured endpoint resolution is tracked in dgx-fun #19.
DEFAULT_ENDPOINT = "http://192.168.1.147:8001/v1"


def discover_model(endpoint: str = DEFAULT_ENDPOINT, timeout: float = 10.0) -> str:
    """Return the first model id advertised by ``GET /v1/models``.

    vLLM serves a single chat model per container; that model's id is what every
    request must use or the server returns 400.
    """
    url = endpoint.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach DGX endpoint at {url}: {e}. "
            f"Is vllm-chat running on the Spark?"
        ) from e
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"No models advertised at {url}")
    return models[0]["id"]
