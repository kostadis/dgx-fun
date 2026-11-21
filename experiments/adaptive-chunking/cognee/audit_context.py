"""Verify frozen source lineage and that both ingestion stages received context.

Exports counts and capture IDs only. An audit is not an assessment of whether
the model interpreted the context correctly; review summaries and answers too.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from analyze import load, source_chunks


def normalize(text):
    return " ".join(text.split())


def verify_recall_context(chunks, calls, recalls):
    """Check evidence metadata against actual reader requests, not IDs alone."""
    indexed = {c["id"]: c["text"] for c in chunks}
    captured = {c["id"]: c for c in calls}
    verified = 0
    for recall in recalls:
        request_ids = [c["id"] for c in recall["llm_calls"]]
        if not request_ids or any(i not in captured for i in request_ids):
            raise ValueError("Recall lacks captured model requests")
        prompt = normalize("\n".join(m["content"] for i in request_ids for m in
                           captured[i]["forwarded"]["messages"]))
        ids = {r["artifact_id"] for a in recall["answers"] for r in
               a.get("metadata", {}).get("evidence", []) if r.get("kind") == "graph_node"
               and r.get("role") == "used_as_context" and r.get("artifact_id") in indexed}
        for chunk_id in ids:
            text = indexed[chunk_id]
            variants = [text, json.dumps(text, ensure_ascii=False)[1:-1], json.dumps(text)[1:-1]]
            if not any(normalize(v) in prompt for v in variants):
                raise ValueError(f"Cited context chunk {chunk_id} absent from actual reader request")
            verified += 1
    return {"queries_audited": len(recalls), "contextual_chunk_appearances_verified": verified}


def audit(root):
    manifest = load(root / "manifest.json")
    if manifest.get("chunk_context_policy") != "breadcrumb":
        raise ValueError("Expected a breadcrumb-preserving adaptive run")
    lineage = source_chunks(root, "adaptive")
    chunks = [c for path in (root / "chunks").glob("*.json") for c in load(path)["chunks"]]
    planned = [c for path in (root / "planned-chunks").glob("*.json") for c in load(path)["chunks"]]
    graph = load(root / "graph.json")
    persisted = [n[1]["text"] for n in graph["nodes"] if n[1].get("type") == "DocumentChunk"]
    if not chunks or len(lineage) != len(chunks) or not (
            Counter(c["text"] for c in chunks) == Counter(c["text"] for c in planned) == Counter(persisted)):
        raise ValueError("Contextual chunk plan, source census and persisted graph differ")
    stages = {"summarization": [], "graph_extraction": []}
    calls = [load(path) for path in (root / "calls").glob("*.json")]
    for row in calls:
        if not row["phase"].startswith("remember-"):
            continue
        messages = row["forwarded"]["messages"]
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        stage = ("summarization" if system.startswith("Summarize the chunk for retrieval.") else
                 "graph_extraction" if "build a knowledge graph" in system else None)
        if stage and row["status"] == 200:
            stages[stage].append((row["id"], normalize("\n".join(
                m["content"] for m in messages if m["role"] == "user"))))
    matches = {stage: {} for stage in stages}
    for c in chunks:
        if not c.get("context_prefix", "").strip():
            raise ValueError("Missing chunk context")
        text = normalize(c["text"])
        for stage, requests in stages.items():
            matches[stage][c["id"]] = [call_id for call_id, user in requests if text in user]
            if not matches[stage][c["id"]]:
                raise ValueError(f"Chunk {c['id']} missing contextual input in {stage}")
    hashes = {h for p in (root / "embeddings").glob("*.json") for h in
              load(p).get("input_sha256", [])}
    # Cognee DataPoint.get_embeddable_data strips surrounding whitespace.
    direct_embedded = sum(hashlib.sha256(c["text"].strip().encode()).hexdigest() in hashes for c in chunks)
    if direct_embedded != len(chunks):
        raise ValueError("Missing contextual chunk input at the embedding endpoint")
    recall_audit = verify_recall_context(chunks, calls, [load(p) for p in (root / "recall").glob("*.json")])
    return {"passed": True, "chunks": len(chunks), "frozen_source_spans_verified": len(lineage),
            "persisted_contextual_chunks_verified": len(persisted),
            "chunks_with_exact_contextual_embedding_input": direct_embedded,
            "chunks_received_by_stage": {s: len(m) for s, m in matches.items()},
            "recall_context": recall_audit,
            "matching_capture_ids": matches}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.root)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "matching_capture_ids"}))


if __name__ == "__main__":
    main()
