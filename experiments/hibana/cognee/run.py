"""Run actual Cognee ingestion and graph recall in private storage.

The existing installed environment and editable source are reused read-only.
The capture proxy bounds LLM concurrency and preserves real SDK requests;
it changes only the model ID to the endpoint's advertised ID.
"""
import argparse
import asyncio
import dataclasses
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid


def serial(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write(path, data):
    path.write_text(json.dumps(data, default=serial, ensure_ascii=False, indent=2) + "\n")


class Capture:
    def __init__(self, root, backend, model):
        self.root, self.backend, self.model = root, backend, model
        self.phase = "startup"
        self.limit = threading.BoundedSemaphore(4)
        self.lock = threading.Lock()
        self.rows = []
        capture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                route = "reader" if self.path.startswith("/reader/") else "native"
                base = "http://192.168.1.147:8001" if route == "reader" else capture.backend
                model = "qwen3.8-flash-next" if route == "reader" else capture.model
                path = "/v1/" + self.path.split("/v1/", 1)[-1]
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                forwarded = {**body, "model": model}
                row = {"id": uuid.uuid4().hex, "phase": capture.phase, "route": route,
                       "started_unix": time.time(), "request": body, "forwarded": forwarded}
                start = time.monotonic()
                status = 502
                payload = b'{"error":{"message":"capture proxy failure"}}'
                with capture.limit:
                    row["proxy_queue_seconds"] = time.monotonic() - start
                    upstream_start = time.monotonic()
                    req = urllib.request.Request(base + path, data=json.dumps(forwarded).encode(),
                        headers={"Content-Type": "application/json", "Authorization": "Bearer local-shadow-test"})
                    try:
                        with urllib.request.urlopen(req, timeout=1800) as response:
                            payload, status = response.read(), response.status
                    except urllib.error.HTTPError as e:
                        payload, status = e.read(), e.code
                    except Exception as e:
                        row["error"] = repr(e)
                    row["upstream_seconds"] = time.monotonic() - upstream_start
                row.update(seconds=time.monotonic() - start, status=status)
                try:
                    row["response"] = json.loads(payload)
                except ValueError:
                    row["response_text"] = payload.decode(errors="replace")
                write(capture.root / "calls" / (row["id"] + ".json"), row)
                with capture.lock:
                    capture.rows.append(row)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"


async def execute(a, capture, result):
    import cognee
    import litellm
    from cognee.base_config import get_base_config
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.llm.config import LLMConfig, get_llm_config

    config = get_base_config()
    for key in ("data_root_directory", "system_root_directory", "cache_root_directory"):
        assert Path(getattr(config, key)).resolve().is_relative_to(a.output.resolve()), key
    assert os.environ["CACHING"] == "true"
    litellm.request_timeout = 1800
    result["effective_roots"] = {k: getattr(config, k) for k in (
        "data_root_directory", "system_root_directory", "cache_root_directory")}
    result["runtime"] = {"source": cognee.__file__, "version": cognee.__version__,
        "installed_versions": {name: importlib.metadata.version(name) for name in (
            "cognee", "litellm", "instructor", "ladybug", "lancedb")}}
    result["source_head"] = subprocess.check_output([
        "git", "-C", "/home/kostadis/src/cognee", "rev-parse", "HEAD"], text=True).strip()
    print("config", json.dumps({"arm": a.arm, "roots": result["effective_roots"],
        "llm": get_llm_config().llm_model, "endpoint": get_llm_config().llm_endpoint}), flush=True)
    if a.check_only:
        result["passed"] = True
        return
    files = sorted(a.corpus.glob("*.md"))
    if a.smoke:
        files = [a.corpus / "hedrack.md"]
    result["inputs"] = [{"name": p.name, "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    capture.phase = "remember"
    started = time.monotonic()
    remembered = await cognee.remember([str(p.resolve()) for p in files],
        dataset_name="hibana_shadow", self_improvement=False, data_per_batch=2)
    result["remember"] = {"result": remembered, "seconds": time.monotonic() - started}
    write(a.output / "remember.json", result["remember"])
    print("remember", json.dumps(result["remember"], default=serial), flush=True)
    graph = await get_graph_engine()
    nodes, edges = await graph.get_graph_data()
    write(a.output / "graph.json", {"nodes": nodes, "edges": edges})
    result["graph"] = {"nodes": len(nodes), "edges": len(edges)}
    if not nodes or not edges:
        raise RuntimeError("Remember produced an empty graph")
    cases = json.loads(a.cases.read_text())
    if a.smoke:
        cases = cases[:1]
    result["recall"] = []
    for reader in (("native", "shared") if not a.smoke else ("native",)):
        endpoint = capture.base + ("/reader/v1" if reader == "shared" else "/native/v1")
        model = "openai/qwen3.8-flash-next" if reader == "shared" else "openai/" + a.model
        reader_config = LLMConfig(llm_provider="custom", llm_model=model, llm_endpoint=endpoint,
            llm_api_key="local-shadow-test", llm_temperature=0, llm_seed=604,
            llm_args={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            auto_rate_limit=False, llm_query_model=model, llm_query_provider="custom",
            llm_query_endpoint=endpoint, llm_query_api_key="local-shadow-test")
        for case in cases:
            capture.phase = f"recall-{reader}-{case['id']}"
            started = time.monotonic()
            answers = await cognee.recall(case["question"], datasets=["hibana_shadow"],
                query_type=cognee.SearchType.GRAPH_COMPLETION, auto_route=False, top_k=15,
                system_prompt="Answer using only the retrieved evidence. Preserve negation and distinguish observed events from future possibilities. Say unknown when facts are not established. Be concise (at most 120 words).",
                include_references=True, llm_config=reader_config)
            row = {"id": case["id"], "reader": reader, "question": case["question"],
                "expected": case["expected"], "seconds": time.monotonic() - started, "answers": answers}
            result["recall"].append(row)
            write(a.output / (f"recall-{reader}-{case['id']}.json"), row)
            write(a.output / "progress.json", result)
            print("recall", reader, case["id"], round(row["seconds"], 2),
                  json.dumps(answers, default=serial, ensure_ascii=False)[:1800], flush=True)
    result["passed"] = True
    await cognee.disconnect()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("baseline", "exact", "fast"), required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    a.output, a.corpus, a.cases = a.output.resolve(), a.corpus.resolve(), a.cases.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "calls").mkdir()
    capture = Capture(a.output, a.backend.rstrip("/"), a.model)
    settings = {
        "LLM_PROVIDER": "custom", "LLM_MODEL": "openai/" + a.model,
        "LLM_ENDPOINT": capture.base + "/native/v1", "LLM_API_KEY": "local-shadow-test",
        "LLM_TEMPERATURE": "0", "LLM_SEED": "604", "LLM_MAX_COMPLETION_TOKENS": "16384",
        "LLM_ARGS": json.dumps({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}),
        "EMBEDDING_PROVIDER": "openai_compatible", "EMBEDDING_MODEL": "qwen3-embedding:0.6b",
        "EMBEDDING_ENDPOINT": "http://192.168.1.121:11434/v1", "EMBEDDING_API_KEY": "ollama",
        "EMBEDDING_DIMENSIONS": "1024", "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "CACHING": "true", "AUTO_FEEDBACK": "true", "AUTO_RATE_LIMIT": "false",
        "RAISE_INCREMENTAL_LOADING_ERRORS": "false", "TELEMETRY_DISABLED": "1",
        "COGNEE_TRACING_ENABLED": "false", "COGNEE_ALLOWED_LOCAL_FILE_ROOTS": str(a.corpus),
        "DATA_ROOT_DIRECTORY": str(a.output / "data"), "SYSTEM_ROOT_DIRECTORY": str(a.output / "system"),
        "CACHE_ROOT_DIRECTORY": str(a.output / "cache"), "COGNEE_LOGS_DIR": str(a.output / "logs"),
        "LOG_LEVEL": "WARNING", "TOKENIZERS_PARALLELISM": "false"}
    envfile = a.output / ".env"
    envfile.write_text("\n".join(k + "=" + json.dumps(v) for k, v in settings.items()) + "\n")
    os.environ.update(settings)
    os.environ["COGNEE_ENV_FILE"] = str(envfile)
    os.chdir(a.output)
    result = {"arm": a.arm, "backend": a.backend, "model": a.model, "started_unix": time.time(), "passed": False}
    try:
        asyncio.run(execute(a, capture, result))
    except Exception as e:
        result["error"] = repr(e)
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        capture.server.shutdown()
        result["finished_unix"] = time.time()
        result["llm_calls"] = [{"id": r["id"], "phase": r["phase"], "route": r["route"],
            "seconds": r["seconds"], "upstream_seconds": r["upstream_seconds"],
            "proxy_queue_seconds": r["proxy_queue_seconds"], "status": r["status"],
            "usage": r.get("response", {}).get("usage")} for r in capture.rows]
        write(a.output / "summary.json", result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
