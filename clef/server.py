"""Jev/SystemOne-compatible HTTP server for Cloudflare's Clef decision models.

Serves Cloudflare/clef and/or Cloudflare/clef-flash from one process on one GPU,
using the release's own `joint_schema_model.py` (record encoding, joint schema
head, `systemone`). This file owns only the HTTP layer: model selection, image
decoding, a GPU lock, and timing. It deliberately re-implements nothing from the
release — if the local output disagrees with Workers AI, the bug is not in here.

    POST /v1/systemone   Jev/SystemOne request body -> response body
    GET  /v1/models      served model ids
    GET  /health         200 once every model is loaded

Env:
    CLEF_MODELS      comma list of model ids to load, in order (default "clef-flash,clef")
    CLEF_MAX_LENGTH  token budget per request (default 16384 = the release default;
                     the backbone itself takes 64K)
    PORT             listen port (default 8002)
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import sys
import threading
import time

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from huggingface_hub import snapshot_download
from PIL import Image
import uvicorn


REPOS = {"clef": "Cloudflare/clef", "clef-flash": "Cloudflare/clef-flash"}
MODEL_IDS = [m.strip() for m in os.environ.get("CLEF_MODELS", "clef-flash,clef").split(",") if m.strip()]
MAX_LENGTH = int(os.environ.get("CLEF_MAX_LENGTH", "16384"))
PORT = int(os.environ.get("PORT", "8002"))

log = logging.getLogger("clef")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()
models: dict[str, tuple[object, object]] = {}
gpu_lock = threading.Lock()
systemone = None  # bound from the release's joint_schema_model.py at load time


def load_all() -> None:
    global systemone
    for model_id in MODEL_IDS:
        if model_id not in REPOS:
            raise SystemExit(f"unknown model id {model_id!r}; expected one of {sorted(REPOS)}")
        path = snapshot_download(REPOS[model_id], local_files_only=True)
        if path not in sys.path:
            sys.path.insert(0, path)
        import joint_schema_model  # identical file in both releases

        systemone = joint_schema_model.systemone
        started = time.monotonic()
        model, processor = joint_schema_model.load_release_model(path, device="cuda")
        log.info(
            "loaded %s from %s in %.1fs (cuda allocated %.1f GiB)",
            model_id, path, time.monotonic() - started, torch.cuda.memory_allocated() / 2**30,
        )
        models[model_id] = (model, processor)


def decode_image(value: object) -> Image.Image:
    """Accept a base64 string or a data: URL. Jev's wire format for images is not
    documented in the release, so this is our choice, not Cloudflare's."""
    if not isinstance(value, str):
        raise HTTPException(400, "images must be base64 strings or data: URLs")
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


def answer(body: dict) -> tuple[dict, float]:
    model, processor = models[body["model"]]
    if body.get("images"):
        body = {**body, "images": [decode_image(image) for image in body["images"]]}
    with gpu_lock:
        started = time.monotonic()
        response = systemone(model, processor, body, max_length=MAX_LENGTH)
        torch.cuda.synchronize()
        elapsed_ms = (time.monotonic() - started) * 1000
    return response, elapsed_ms


@app.post("/v1/systemone")
async def post_systemone(request: Request) -> JSONResponse:
    body = await request.json()
    if body.get("model") not in models:
        raise HTTPException(404, f"model must be one of {sorted(models)}")
    try:
        response, elapsed_ms = await asyncio.to_thread(answer, body)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    log.info(
        "%s %d input tokens %.1f ms",
        body["model"], response["usage"]["input_tokens"], elapsed_ms,
    )
    return JSONResponse(response, headers={"X-Clef-Latency-Ms": f"{elapsed_ms:.1f}"})


@app.get("/v1/models")
async def get_models() -> dict:
    return {"object": "list", "data": [{"id": model_id, "object": "model"} for model_id in models]}


@app.get("/health")
async def health() -> JSONResponse:
    ready = all(model_id in models for model_id in MODEL_IDS)
    return JSONResponse({"ready": ready, "loaded": list(models)}, status_code=200 if ready else 503)


if __name__ == "__main__":
    load_all()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
