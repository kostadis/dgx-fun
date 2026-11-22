"""Replay frozen factual questions against a cloned Cognee index, without agents.

This never ingests documents. Each question has a fresh session and one recall.
Full answers and captures must remain in the ignored raw directory.
"""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import shutil
import threading
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("question_capture", HERE / "run.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
SYSTEM_PROMPT = ("Answer using only the retrieved evidence. Preserve negation. Say unknown when "
                 "the evidence does not establish the answer. Be concise (at most 120 words).")


async def execute(a, capture, embedding, manifest, state):
    import cognee
    from cognee.base_config import get_base_config
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.llm.config import LLMConfig

    config = get_base_config()
    for name, directory in (("data_root_directory", "data"),
                            ("system_root_directory", "system"),
                            ("cache_root_directory", "cache")):
        assert Path(getattr(config, name)).resolve() == a.output / directory
    model, endpoint = "openai/" + capture.model, capture.base + "/native/v1"
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    if a.reasoning_effort is not None:
        extra_body["reasoning_effort"] = a.reasoning_effort
    reader = LLMConfig(
        llm_provider="custom", llm_model=model, llm_endpoint=endpoint,
        llm_api_key="local-shadow-test", llm_temperature=0, llm_seed=604,
        llm_args={"extra_body": extra_body}, auto_rate_limit=False,
        llm_query_model=model, llm_query_provider="custom",
        llm_query_endpoint=endpoint, llm_query_api_key="local-shadow-test")
    cases = json.loads(a.cases.read_text())
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Questions must be nonempty with unique IDs")
    started = time.monotonic()
    state["stage"] = "recall"
    for case in cases:
        state["question"] = case["id"]
        capture.phase = "recall-isolated-" + case["id"]
        session = f"questions-{case['id']}-{uuid.uuid4().hex}"
        tick, unix, offset = time.monotonic(), time.time(), len(capture.rows)
        answers = await cognee.recall(
            case["question"], datasets=["pabtso_chunking_trial"],
            query_type=cognee.SearchType.GRAPH_COMPLETION, auto_route=False,
            top_k=15, system_prompt=SYSTEM_PROMPT, include_references=True,
            llm_config=reader, session_id=session)
        seconds = time.monotonic() - tick
        rows = capture.rows[offset:]
        calls = R.audit_calls(capture, rows)
        if not answers or len(rows) != 1 or any(
                r["status"] != 200 or r["finish_reasons"] != ["stop"] for r in calls):
            raise RuntimeError("Expected exactly one successful, untruncated reader call")
        for row in rows:
            prompt = "\n".join(m.get("content", "") for m in row["forwarded"]["messages"])
            if any(c["question"] in prompt for c in cases if c["id"] != case["id"]):
                raise RuntimeError("Another benchmark question leaked into this prompt")
        serialized = [v.model_dump(mode="json") if hasattr(v, "model_dump") else v for v in answers]
        R.save(a.output / "recall" / (case["id"] + ".json"), {
            "id": case["id"], "question": case["question"], "seconds": seconds,
            "started_unix": unix, "session_id": session, "history_audit_captured": True,
            "answers": serialized, "llm_calls": calls})
        state["completed_questions"] = len(list((a.output / "recall").glob("*.json")))
        R.save(a.output / "progress.json", state)
        print("recalled", case["id"], round(seconds, 3), flush=True)
    graph = await get_graph_engine()
    nodes, edges = await graph.get_graph_data()
    counts = {"nodes": len(nodes), "edges": len(edges),
              "document_chunks": sum(p.get("type") == "DocumentChunk" for _, p in nodes)}
    if counts != json.loads((a.source / "progress.json").read_text())["graph"]:
        raise RuntimeError("Query-only run changed graph counts")
    state.update(graph_counts=counts, recall_loop_seconds=time.monotonic() - started)
    await cognee.disconnect()
    state.update(passed=True, stage="complete")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high"))
    a = p.parse_args()
    for name in ("source", "output", "cases"):
        setattr(a, name, getattr(a, name).resolve())
    if not a.output.is_relative_to(HERE.parent / "raw") or a.output.is_relative_to(a.source):
        p.error("Use a new isolated output under the ignored raw directory")
    manifest = json.loads((a.source / "manifest.json").read_text())
    if R.digest(a.cases.read_bytes()) != manifest["input_sha256"]["cases"]:
        p.error("Question file differs from the frozen ingestion trial")
    a.output.mkdir(parents=True)
    for name in ("data", "system", "corpus"):
        shutil.copytree(a.source / name, a.output / name)
    for name in ("calls", "embeddings", "recall"):
        (a.output / name).mkdir()
    capture = R.harness.Capture(a.output, a.backend.rstrip("/"), a.model)
    capture.limit = threading.BoundedSemaphore(1)
    embedding = R.Embeddings(a.output, manifest["embedding_endpoint"], manifest["embedding_model"])
    assert embedding.identity == manifest["embedding_digest"]
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    if a.reasoning_effort is not None:
        extra_body["reasoning_effort"] = a.reasoning_effort
    settings = {k: json.loads(v) for line in (a.source / ".env").read_text().splitlines()
                if line for k, v in [line.split("=", 1)]}
    settings.update(
        LLM_ENDPOINT=capture.base + "/native/v1", LLM_MODEL="openai/" + a.model,
        LLM_ARGS=json.dumps({"extra_body": extra_body}),
        EMBEDDING_ENDPOINT=embedding.base + "/v1",
        DATA_ROOT_DIRECTORY=str(a.output / "data"), SYSTEM_ROOT_DIRECTORY=str(a.output / "system"),
        CACHE_ROOT_DIRECTORY=str(a.output / "cache"), COGNEE_LOGS_DIR=str(a.output / "logs"),
        COGNEE_ALLOWED_LOCAL_FILE_ROOTS=str(a.output / "corpus"), AUTO_FEEDBACK="false")
    envfile = a.output / ".env"
    envfile.write_text("\n".join(k + "=" + json.dumps(v) for k, v in settings.items()) + "\n")
    os.environ.update(settings)
    os.environ["COGNEE_ENV_FILE"] = str(envfile)
    os.chdir(a.output)
    protocol = {
        "experiment": "Frozen factual question replay; one recall per question",
        "source": str(a.source), "source_manifest": manifest,
        "source_manifest_sha256": R.digest((a.source / "manifest.json").read_bytes()),
        "cases_sha256": R.digest(a.cases.read_bytes()),
        "script_sha256": R.digest(Path(__file__).read_bytes()),
        "capture_sha256": R.digest(Path(R.harness.__file__).read_bytes()),
        "model": a.model, "backend": capture.backend, "top_k": 15,
        "system_prompt": SYSTEM_PROMPT, "temperature": 0, "seed": 604,
        "reasoning_effort": a.reasoning_effort, "question_concurrency": 1,
        "fresh_session_per_question": True, "auto_feedback": False,
        "agent_calls": 0, "ingestion_calls": 0}
    R.save(a.output / "protocol.json", protocol)
    (a.output / "runner.py.snapshot").write_bytes(Path(__file__).read_bytes())
    state = {"stage": "startup", "started_unix": time.time(), "passed": False,
             "completed_questions": 0}
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(15):
            state["updated_unix"] = time.time()
            R.save(a.output / "progress.json", state)
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        asyncio.run(execute(a, capture, embedding, manifest, state))
    except Exception as error:
        state.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        stop.set()
        state["finished_unix"] = time.time()
        R.save(a.output / "progress.json", state)
        R.save(a.output / "summary.json", {**state, "calls": R.audit_calls(capture, capture.rows),
                                          "embeddings": embedding.rows})
        capture.server.shutdown()
        embedding.server.shutdown()
    raise SystemExit(0 if state["passed"] else 1)


if __name__ == "__main__":
    main()
