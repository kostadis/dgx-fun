"""Real Cognee graph trial, reusing the Hibana SDK capture harness.

Run each arm in a separate Python process and storage root. Frozen adaptive
boundaries come from the earlier question-independent four-metric selection.
Book text, graph dumps, prompts and answers belong only in ignored raw/.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location("hibana_capture", REPO / "experiments/hibana/cognee/run.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def adaptive_payload(source, chunk, context_policy):
    """Attach frozen ancestry without changing the source span or boundaries."""
    start, end = chunk["start"], chunk["end"]
    raw = chunk["text"]
    if not raw or source[start:end] != raw:
        raise ValueError("Frozen boundary/source mismatch")
    if context_policy == "raw":
        prefix = ""
    elif context_policy == "breadcrumb":
        contextual = chunk["retrieval_text"]
        if not contextual.endswith(raw):
            raise ValueError("Frozen retrieval text does not preserve the raw source span")
        prefix = contextual[:-len(raw)]
        if not prefix.strip():
            raise ValueError("Missing frozen parent-heading context")
    else:
        raise ValueError("Unknown adaptive context policy")
    return {"text": prefix + raw, "raw_text": raw, "context_prefix": prefix,
            "source_start": start, "source_end": end}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    harness.write(temporary, value)
    temporary.replace(path)


def request(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


class Embeddings:
    """OpenAI shim to Ollama native embed with truncation explicitly disabled."""
    def __init__(self, root, endpoint, model):
        self.endpoint, self.model, self.root = endpoint.rstrip("/"), model, root
        tags = request(self.endpoint + "/api/tags")
        self.identity = next(m["digest"] for m in tags["models"] if m["name"] == model)
        self.info = request(self.endpoint + "/api/show", {"model": model})
        self.rows, self.lock, self.limit = [], threading.Lock(), threading.BoundedSemaphore(2)
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                started = time.monotonic()
                row = {"id": uuid.uuid4().hex, "started_unix": time.time(),
                       "upstream": proxy.endpoint, "model": model, "truncate": False}
                status = 200
                try:
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    if self.path == "/api/show":
                        payload = proxy.info
                    elif self.path == "/v1/embeddings":
                        if body["model"] != model:
                            raise ValueError("Unexpected embedding model")
                        texts = body["input"]
                        texts = [texts] if isinstance(texts, str) else texts
                        if not all(isinstance(t, str) for t in texts):
                            raise ValueError("Embedding inputs must be strings")
                        row["input_sha256"] = [digest(t) for t in texts]
                        row["input_count"] = len(texts)
                        with proxy.limit:
                            result = request(proxy.endpoint + "/api/embed",
                                             {"model": model, "input": texts, "truncate": False})
                        vectors = result["embeddings"]
                        if len(vectors) != len(texts) or any(len(v) != 1024 or
                                not all(math.isfinite(x) for x in v) or not any(v) for v in vectors):
                            raise ValueError("Invalid embedding dimensions/count/values")
                        row["dimensions"] = 1024
                        tokens = result.get("prompt_eval_count", 0)
                        row["prompt_eval_count"] = tokens
                        payload = {"object": "list", "model": model,
                                   "data": [{"object": "embedding", "index": i, "embedding": v}
                                            for i, v in enumerate(vectors)],
                                   "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}
                    else:
                        raise ValueError("Unexpected embedding path " + self.path)
                except Exception as error:
                    status = 502
                    row["error"] = repr(error)
                    payload = {"error": {"message": repr(error)}}
                row.update(status=status, seconds=time.monotonic() - started)
                save(root / "embeddings" / (row["id"] + ".json"), row)
                with proxy.lock:
                    proxy.rows.append(row)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"


def chunker_class(a, corpus, chunks, selection):
    # These imports must follow the isolated environment setup.
    from cognee.modules.chunking.Chunker import Chunker
    from cognee.modules.chunking.TextChunker import TextChunker
    from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
    from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
    from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine

    documents = {digest(d["text"]): d for d in corpus["documents"]}

    class AuditedChunker(Chunker):
        chunker_id = ("text_chunker_v1" if a.arm == "current" else
                      "frozen_adaptive_four_metric_breadcrumb_v2" if a.adaptive_context == "breadcrumb"
                      else "frozen_adaptive_four_metric_v1")

        async def read(self):
            text = "".join([part async for part in self.get_text()])
            doc = documents[digest(text)]
            async def replay():
                yield text
            provenance = {}
            if a.arm == "current":
                produced = [c async for c in TextChunker(self.document, replay, self.max_chunk_size).read()]
                if "".join(c.text for c in produced) != text:
                    raise ValueError("Stock chunker did not reconstruct document")
            else:
                method = selection[doc["id"]]["selected"]
                plan = [c for c in chunks[method] if c["doc_id"] == doc["id"]]
                produced, occurrences = [], Counter()
                for index, c in enumerate(plan):
                    payload = adaptive_payload(text, c, a.adaptive_context)
                    provenance[index] = {k: v for k, v in payload.items() if k != "text"}
                    tokens = get_embedding_engine().tokenizer.count_tokens(payload["text"])
                    if tokens > self.max_chunk_size:
                        raise ValueError("Frozen chunk exceeds effective embedding cap")
                    h = chunk_content_hash(payload["text"])
                    produced.append(DocumentChunk(
                        id=content_chunk_id(str(self.document.id), h, occurrences[h]),
                        chunker_id=self.chunker_id, text=payload["text"], chunk_size=tokens,
                        chunk_index=index, cut_type="frozen_adaptive", content_hash=h,
                        max_chunk_tokens=self.max_chunk_size, is_part_of=self.document,
                        contains=[], importance_weight=self.document.importance_weight,
                        document_id=str(self.document.id), document_name=self.document.name,
                        metadata={"index_fields": ["text"]}))
                    occurrences[h] += 1
            directory = "planned-chunks" if self.document.name.startswith("dry-run-") else "chunks"
            save(a.output / directory / (doc["id"] + ".json"),
                 {"doc_id": doc["id"], "source_sha256": digest(text),
                  "chunker_id": self.chunker_id, "budget": self.max_chunk_size,
                  "chunks": [{"id": str(c.id), "text": c.text, "tokens": c.chunk_size,
                              "index": c.chunk_index, **provenance.get(c.chunk_index, {})}
                             for c in produced]})
            for c in produced:
                yield c
    return AuditedChunker


def audit_calls(capture, rows):
    return [{"id": r["id"], "phase": r["phase"], "status": r["status"],
             "seconds": r["seconds"], "queue_seconds": r["proxy_queue_seconds"],
             "upstream_seconds": r["upstream_seconds"],
             "usage": r.get("response", {}).get("usage"),
             "finish_reasons": [c.get("finish_reason") for c in r.get("response", {}).get("choices", [])]}
            for r in rows]


async def execute(a, capture, embedding, state):
    import cognee
    import importlib.metadata
    import litellm
    from cognee.base_config import get_base_config
    from cognee.infrastructure.llm.config import LLMConfig, get_llm_config
    from cognee.infrastructure.llm.utils import resolve_chunk_size
    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
    from cognee.modules.cognify.estimator import _chunks_from_texts, estimate_chunks

    config = get_base_config()
    roots = {k: str(getattr(config, k)) for k in
             ("data_root_directory", "system_root_directory", "cache_root_directory")}
    assert all(Path(p).resolve().is_relative_to(a.output) for p in roots.values())
    assert get_llm_config().llm_endpoint == capture.base + "/native/v1"
    assert get_embedding_config().embedding_endpoint == embedding.base + "/v1"
    litellm.request_timeout = 1800
    state["roots"] = roots
    state["runtime"] = {k: importlib.metadata.version(k) for k in
                        ("cognee", "litellm", "instructor", "ladybug", "lancedb")}
    state["cognee_head"] = subprocess.check_output(
        ["git", "-C", str(Path(cognee.__file__).parents[1]), "rev-parse", "HEAD"], text=True).strip()
    corpus = json.loads(a.corpus.read_text())
    chunks = json.loads(a.chunks.read_text())
    selection = json.loads(a.selection.read_text())
    chunker = chunker_class(a, corpus, chunks, selection)
    size = await resolve_chunk_size(None)
    state["effective_chunk_budget"] = size
    state["embedding_tokenizer"] = type(get_embedding_engine().tokenizer).__module__
    documents = corpus["documents"]
    if a.smoke:
        # Exact existing source section: enough to exercise both chunkers and recall.
        documents = [d for d in documents if d["id"] == "pabtso-10"]
    estimated_chunks = await _chunks_from_texts([d["text"] for d in documents],
                                               chunker=chunker, chunk_size=size)
    estimate = estimate_chunks(estimated_chunks, operation="remember").to_dict()
    save(a.output / "estimate.json", estimate)
    state["planned_chunks"] = len(estimated_chunks)
    print("estimate", json.dumps(estimate), flush=True)
    if a.phase == "plan":
        return
    dataset = "pabtso_chunking_trial"
    for doc in documents:
        checkpoint = a.output / "remember" / (doc["id"] + ".json")
        if checkpoint.exists():
            continue
        state["stage"], state["document"] = "remember", doc["id"]
        capture.phase = "remember-" + doc["id"]
        path = a.output / "corpus" / (doc["id"] + ".txt")
        path.write_text(doc["text"])
        started, offset = time.monotonic(), len(capture.rows)
        result = await cognee.remember(str(path), dataset_name=dataset, self_improvement=False,
                                       data_per_batch=1, chunks_per_batch=4, chunker=chunker,
                                       raise_on_error=True)
        calls = audit_calls(capture, capture.rows[offset:])
        if any(r["status"] != 200 or "length" in r["finish_reasons"] for r in calls):
            raise RuntimeError("Ingestion had failed or truncated LLM calls; inspect raw captures")
        save(checkpoint, {"doc_id": doc["id"], "seconds": time.monotonic() - started,
                          "result": result, "llm_calls": calls})
        print("remembered", doc["id"], round(time.monotonic() - started, 2), flush=True)
    state["stage"] = "graph-audit"
    graph = await get_graph_engine()
    nodes, edges = await graph.get_graph_data()
    save(a.output / "graph.json", {"nodes": nodes, "edges": edges})
    state["graph"] = {"nodes": len(nodes), "edges": len(edges)}
    if not nodes or not edges:
        raise RuntimeError("Empty graph after ingestion")
    persisted = [properties for _, properties in nodes if properties.get("type") == "DocumentChunk"]
    expected = Counter(c.text for c in estimated_chunks)
    actual = Counter(c["text"] for c in persisted)
    if expected != actual:
        raise RuntimeError(f"Persisted chunk census differs from plan: {len(persisted)} vs {len(estimated_chunks)}")
    state["graph"]["document_chunks"] = len(persisted)
    if a.phase == "ingest":
        await cognee.disconnect()
        return
    cases = json.loads(a.cases.read_text())
    if a.smoke:
        # All six item questions, including the cross-heading attribution regression.
        cases = [c for c in cases if c["id"] in {f"q{i}" for i in range(41, 47)}]
    endpoint, model = capture.base + "/native/v1", "openai/" + a.model
    reader = LLMConfig(llm_provider="custom", llm_model=model, llm_endpoint=endpoint,
        llm_api_key="local-shadow-test", llm_temperature=0, llm_seed=604,
        llm_args={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        auto_rate_limit=False, llm_query_model=model, llm_query_provider="custom",
        llm_query_endpoint=endpoint, llm_query_api_key="local-shadow-test")
    for case in cases:
        checkpoint = a.output / "recall" / (case["id"] + ".json")
        if checkpoint.exists():
            continue
        state["stage"], state["question"] = "recall", case["id"]
        capture.phase = "recall-isolated-" + case["id"]
        session_id = f"trial-{a.arm}-{case['id']}-{uuid.uuid4().hex}"
        started_unix = time.time()
        started, offset = time.monotonic(), len(capture.rows)
        answers = await cognee.recall(case["question"], datasets=[dataset],
            query_type=cognee.SearchType.GRAPH_COMPLETION, auto_route=False, top_k=15,
            system_prompt="Answer using only the retrieved evidence. Preserve negation. Say unknown when the evidence does not establish the answer. Be concise (at most 120 words).",
            include_references=True, llm_config=reader, session_id=session_id)
        calls = audit_calls(capture, capture.rows[offset:])
        for row in capture.rows[offset:]:
            prompt = "\n".join(m.get("content", "") for m in row["forwarded"].get("messages", []))
            if any(other["question"] in prompt for other in cases if other["id"] != case["id"]):
                raise RuntimeError("Another benchmark question leaked into the retrieval prompt")
        if not answers or any(r["status"] != 200 or "length" in r["finish_reasons"] for r in calls):
            raise RuntimeError("Missing answer or failed/truncated recall")
        save(checkpoint, {"id": case["id"], "question": case["question"],
                          "seconds": time.monotonic() - started, "started_unix": started_unix,
                          "session_id": session_id, "history_audit_captured": bool(calls),
                          "answers": answers, "llm_calls": calls})
        print("recalled", case["id"], round(time.monotonic() - started, 2), flush=True)
    await cognee.disconnect()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = HERE.parent / "raw/pabtso"
    p.add_argument("--arm", choices=("current", "adaptive"), required=True)
    p.add_argument("--adaptive-context", choices=("breadcrumb", "raw"), default="breadcrumb",
                   help="Preserve frozen parent headings; raw reproduces the original defective trial")
    p.add_argument("--llm-concurrency", type=int, choices=range(1, 9), default=4,
                   help="Maximum concurrent captured model calls (DGX server supports eight)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", default="http://192.168.1.147:8001")
    p.add_argument("--model", default="qwen3.8-flash-next")
    p.add_argument("--embedding-endpoint", default="http://192.168.56.1:11434")
    p.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    p.add_argument("--corpus", type=Path, default=base / "corpus.json")
    p.add_argument("--cases", type=Path, default=base / "cases.json")
    p.add_argument("--chunks", type=Path, default=base / "pilot-20261120/chunks-private.json")
    p.add_argument("--selection", type=Path, default=base / "pilot-20261120/intrinsic.json")
    p.add_argument("--phase", choices=("plan", "ingest", "all"), default="all")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-code-sha", help="Explicitly audited previous runner SHA for a code-only resume")
    a = p.parse_args()
    for key in ("output", "corpus", "cases", "chunks", "selection"):
        setattr(a, key, getattr(a, key).resolve())
    if not a.output.is_relative_to(HERE.parent / "raw"):
        p.error("Output must be under the ignored experiments/adaptive-chunking/raw directory")
    a.output.mkdir(parents=True, exist_ok=a.resume)
    for directory in ("calls", "embeddings", "chunks", "planned-chunks", "corpus", "remember", "recall"):
        (a.output / directory).mkdir(exist_ok=True)
    capture = harness.Capture(a.output, a.backend.rstrip("/"), a.model)
    capture.limit = threading.BoundedSemaphore(a.llm_concurrency)
    embedding = Embeddings(a.output, a.embedding_endpoint, a.embedding_model)
    manifest = {"arm": a.arm, "backend": a.backend, "model": a.model,
                "chunk_context_policy": a.adaptive_context if a.arm == "adaptive" else "stock",
                "llm_concurrency": a.llm_concurrency,
                "embedding_endpoint": a.embedding_endpoint, "embedding_model": a.embedding_model,
                "embedding_digest": embedding.identity, "smoke": a.smoke,
                "embedding_model_info": embedding.info.get("model_info"),
                "input_sha256": {key: digest(getattr(a, key).read_bytes())
                                 for key in ("corpus", "cases", "chunks", "selection")},
                "runner_sha256": digest(Path(__file__).read_bytes()),
                "capture_sha256": digest(Path(harness.__file__).read_bytes()),
                "recall_session_policy": "fresh-uuid-per-question"}
    old = a.output / "manifest.json"
    if old.exists():
        previous = json.loads(old.read_text())
        if previous != manifest:
            ignored = {"runner_sha256", "recall_session_policy"}
            same_inputs = {k: v for k, v in previous.items() if k not in ignored} == {
                k: v for k, v in manifest.items() if k not in ignored}
            if not (a.resume and a.resume_code_sha == previous["runner_sha256"] and same_inputs):
                raise ValueError("Resume manifest mismatch; use a new root or explicitly audit the previous runner SHA")
            history = a.output / "manifest-history"
            history.mkdir(exist_ok=True)
            save(history / (previous["runner_sha256"] + ".json"), previous)
            snapshot = a.output / "runner.py.snapshot"
            if snapshot.exists():
                (history / (previous["runner_sha256"] + ".py.snapshot")).write_bytes(snapshot.read_bytes())
            save(history / "session-isolation-migration.json", {
                "previous_runner": previous["runner_sha256"], "new_runner": manifest["runner_sha256"],
                "reason": "Isolate every recall question; ingestion code and frozen inputs are unchanged.",
                "unix": time.time()})
    save(old, manifest)
    (a.output / "runner.py.snapshot").write_bytes(Path(__file__).read_bytes())
    settings = {
        "LLM_PROVIDER": "custom", "LLM_MODEL": "openai/" + a.model,
        "LLM_ENDPOINT": capture.base + "/native/v1", "LLM_API_KEY": "local-shadow-test",
        "LLM_TEMPERATURE": "0", "LLM_SEED": "604", "LLM_MAX_COMPLETION_TOKENS": "16384",
        "LLM_ARGS": json.dumps({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}),
        "EMBEDDING_PROVIDER": "openai_compatible", "EMBEDDING_MODEL": a.embedding_model,
        "EMBEDDING_ENDPOINT": embedding.base + "/v1", "EMBEDDING_API_KEY": "ollama",
        "EMBEDDING_DIMENSIONS": "1024", "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "CACHING": "true", "AUTO_FEEDBACK": "false", "AUTO_RATE_LIMIT": "false",
        "RAISE_INCREMENTAL_LOADING_ERRORS": "true", "TELEMETRY_DISABLED": "1",
        "COGNEE_TRACING_ENABLED": "false", "COGNEE_ALLOWED_LOCAL_FILE_ROOTS": str(a.output / "corpus"),
        "DATA_ROOT_DIRECTORY": str(a.output / "data"), "SYSTEM_ROOT_DIRECTORY": str(a.output / "system"),
        "CACHE_ROOT_DIRECTORY": str(a.output / "cache"), "COGNEE_LOGS_DIR": str(a.output / "logs"),
        "LOG_LEVEL": "WARNING", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OTEL_SDK_DISABLED": "true",
        "LITELLM_LOCAL_MODEL_COST_MAP": "true"}
    envfile = a.output / ".env"
    envfile.write_text("\n".join(k + "=" + json.dumps(v) for k, v in settings.items()) + "\n")
    os.environ.update(settings)
    os.environ["COGNEE_ENV_FILE"] = str(envfile)
    os.chdir(a.output)
    prior_progress = a.output / "progress.json"
    prior = json.loads(prior_progress.read_text()) if prior_progress.exists() else {}
    state = {**manifest, "phase": a.phase, "started_unix": time.time(), "stage": "startup", "passed": False,
             "initial_started_unix": prior.get("initial_started_unix", prior.get("started_unix", time.time()))}
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(15):
            state["updated_unix"] = time.time()
            state["completed_documents"] = len(list((a.output / "remember").glob("*.json")))
            state["completed_questions"] = len(list((a.output / "recall").glob("*.json")))
            state["llm_calls_this_process"] = len(capture.rows)
            state["embedding_requests_this_process"] = len(embedding.rows)
            save(a.output / "progress.json", state)
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        asyncio.run(execute(a, capture, embedding, state))
        state["passed"], state["stage"] = True, "complete"
    except Exception as error:
        state["error"], state["traceback"] = repr(error), traceback.format_exc()
        traceback.print_exc()
    finally:
        stop.set()
        state["finished_unix"] = time.time()
        save(a.output / "progress.json", state)
        save(a.output / "summary.json", {**state, "calls": audit_calls(capture, capture.rows),
                                        "embeddings": embedding.rows})
        capture.server.shutdown()
        embedding.server.shutdown()
    raise SystemExit(0 if state["passed"] else 1)


if __name__ == "__main__":
    main()
