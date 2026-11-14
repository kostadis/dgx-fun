"""Condense completed Cognee receipts without treating graph size as quality."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path


def public_receipt(result):
    """Keep measurements and hashes while full answers and claims stay local."""
    out = copy.deepcopy(result)
    out["publication"] = {"full_answers_and_semantic_edges": "retained in ignored raw/",
                          "answer_hash_encoding": "SHA256 of UTF-8 text"}
    for arm in out["arms"].values():
        for answer in arm["answers"]:
            text = answer.pop("text")
            answer.pop("question", None)
            answer.pop("expected", None)
            answer["answer_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            answer["answer_characters"] = len(text)
        edges = arm.pop("entity_edges")
        arm["entity_edges_sha256"] = hashlib.sha256(
            json.dumps(edges, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return out


def analyze(root):
    summary = json.loads((root / "summary.json").read_text())
    graph = json.loads((root / "graph.json").read_text())
    nodes = dict(graph["nodes"])
    calls = [json.loads(p.read_text()) for p in (root / "calls").glob("*.json")]
    phases = defaultdict(list)
    for call in calls:
        phases["remember" if call["phase"] == "remember" else "recall"].append(call)
    timing = {}
    for phase, rows in phases.items():
        tokens = sum((r.get("response", {}).get("usage") or {}).get("completion_tokens", 0) for r in rows)
        timing[phase] = {"calls": len(rows), "output_tokens": tokens,
            "non_200": sum(r["status"] != 200 for r in rows),
            "upstream_seconds_sum": sum(r["upstream_seconds"] for r in rows),
            "proxy_queue_seconds_sum": sum(r["proxy_queue_seconds"] for r in rows)}
    entity_edges = []
    for source, target, rel, props in graph["edges"]:
        if nodes[source].get("type") == "Entity" and nodes[target].get("type") == "Entity":
            entity_edges.append({"source": nodes[source].get("name"), "relation": rel,
                "target": nodes[target].get("name"), "claim": props.get("edge_text")})
    answers = []
    for row in summary.get("recall", []):
        answers.append({"id": row["id"], "reader": row["reader"], "seconds": row["seconds"],
            "question": row["question"], "expected": row["expected"],
            "text": "\n".join(a.get("text", "") for a in row["answers"]),
            "result_count": len(row["answers"])})
    shared_calls = [r for r in calls if r["phase"].startswith("recall-shared-")]
    result = {"arm": summary["arm"], "completed": summary["passed"],
        "error": summary.get("error"), "runtime": summary.get("runtime"),
        "source_head": summary.get("source_head"), "inputs": summary.get("inputs"),
        "remember_seconds": summary.get("remember", {}).get("seconds"),
        "remember_result": summary.get("remember", {}).get("result"),
        "graph": {"nodes": len(nodes), "edges": len(graph["edges"]),
            "node_types": dict(Counter(n.get("type") for n in nodes.values())),
            "entity_count": sum(n.get("type") == "Entity" for n in nodes.values()),
            "entity_edge_count": len(entity_edges)},
        "shared_reader_routing_confirmed": bool(shared_calls) and all(r["route"] == "reader" for r in shared_calls),
        "llm": timing, "answers": answers, "entity_edges": entity_edges}
    if timing.get("remember"):
        result["ingestion_output_tokens_per_wall_second"] = timing["remember"]["output_tokens"] / result["remember_seconds"]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-private", action="store_true",
                   help="include answer and claim text; output must be inside the ignored raw directory")
    a = p.parse_args()
    if a.include_private and not a.output.resolve().is_relative_to(a.raw.resolve()):
        p.error("private output must stay inside --raw")
    result = {"arms": {}}
    for arm in ("baseline", "exact", "fast"):
        root = a.raw / arm
        if (root / "summary.json").exists():
            result["arms"][arm] = analyze(root)
    exported = result if a.include_private else public_receipt(result)
    a.output.write_text(json.dumps(exported, ensure_ascii=False, indent=2) + "\n")
    for arm, d in result["arms"].items():
        print(arm, "completed", d["completed"], "remember_seconds", d["remember_seconds"],
              "graph", d["graph"], "shared_reader", d["shared_reader_routing_confirmed"])


if __name__ == "__main__":
    main()
