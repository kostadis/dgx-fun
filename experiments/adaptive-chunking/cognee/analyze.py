"""Export allowlisted metrics; keep book text and answer review private."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import time


def load(path):
    return json.loads(path.read_text())


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def check_concurrency(manifests, allow_change):
    values = [m.get("llm_concurrency", 4) for m in manifests]
    if values[0] != values[1] and not allow_change:
        raise ValueError("Concurrency differs; explicitly allow the changed-setting comparison")
    return values


def source_chunks(root, arm):
    frozen = root.parent / "frozen"
    if not (frozen / "corpus.json").exists():
        return {}
    corpus = {d["id"]: d for d in load(frozen / "corpus.json")["documents"]}
    plans = load(frozen / "chunks.json")
    selection = load(frozen / "selection.json")
    result = {}
    for path in (root / "chunks").glob("*.json"):
        doc = load(path)
        doc_id, cursor = doc["doc_id"], 0
        plan = [c for c in plans[selection[doc_id]["selected"]] if c["doc_id"] == doc_id]
        for c in doc["chunks"]:
            if arm == "current":
                start, end = cursor, cursor + len(c["text"])
                cursor = end
            else:
                source = plan[c["index"]]
                start, end = source["start"], source["end"]
                if "raw_text" in c:
                    if (c["source_start"], c["source_end"]) != (start, end):
                        raise ValueError("Contextual chunk changed frozen source boundaries")
                    if c["text"] != c["context_prefix"] + c["raw_text"]:
                        raise ValueError("Contextual chunk does not preserve its audited raw text")
                    if c["context_prefix"] and c["text"] != source["retrieval_text"]:
                        raise ValueError("Contextual chunk differs from frozen retrieval text")
            if corpus[doc_id]["text"][start:end] != c.get("raw_text", c["text"]):
                raise ValueError("Chunk census no longer matches frozen source")
            result[c["id"]] = {"doc_id": doc_id, "start": start, "end": end}
    return result


def coverage(chunks, evidence):
    values = []
    for gold in evidence:
        cursor, count = gold["start"], 0
        intervals = sorted((max(gold["start"], c["start"]), min(gold["end"], c["end"]))
                           for c in chunks if c["doc_id"] == gold["doc_id"]
                           and c["start"] < gold["end"] and c["end"] > gold["start"])
        for start, end in intervals:
            count += max(0, end - max(cursor, start))
            cursor = max(cursor, end)
        values.append(count / (gold["end"] - gold["start"]))
    return values


def quality_summary(assessments, case_ids):
    """Export reviewed labels only; free-form notes can contain source text."""
    categories = {"correct_and_complete", "partial", "incorrect", "abstained", "source_ambiguous"}
    grounding = {"supported", "unsupported", "unassessed"}
    if set(assessments) != set(case_ids):
        raise ValueError("Answer review must cover every frozen question")
    counts, rows = {}, []
    for arm in ("current", "adaptive"):
        values, supports = Counter(), Counter()
        grounded_complete = 0
        for case_id, pair in assessments.items():
            entry = pair[arm]
            if entry["correctness"] not in categories or entry["grounding"] not in grounding:
                raise ValueError("Invalid review label")
            if (entry["correctness"] == "source_ambiguous") != (case_id == "q37"):
                raise ValueError("Only the predeclared q37 source ambiguity is excluded")
            values[entry["correctness"]] += 1
            supports[entry["grounding"]] += 1
            grounded_complete += entry["correctness"] == "correct_and_complete" and entry["grounding"] == "supported"
        counts[arm] = {"correctness": dict(values), "grounding": dict(supports),
                       "adjudicable_questions": len(assessments) - values["source_ambiguous"],
                       "grounded_complete": grounded_complete}
    for case_id, pair in assessments.items():
        rows.append({"id": case_id, **{arm: {key: pair[arm][key] for key in
                     ("correctness", "grounding")} for arm in ("current", "adaptive")}})
    return {"counts": counts, "labels": rows,
            "current_only_complete": [r["id"] for r in rows if r["current"]["correctness"] ==
                "correct_and_complete" and r["adaptive"]["correctness"] != "correct_and_complete"],
            "adaptive_only_complete": [r["id"] for r in rows if r["adaptive"]["correctness"] ==
                "correct_and_complete" and r["current"]["correctness"] != "correct_and_complete"]}


def analyze(root, cases):
    manifest = load(root / "manifest.json")
    progress = load(root / "progress.json")
    remembered = [load(p) for p in sorted((root / "remember").glob("*.json"))]
    recalls = [load(p) for p in sorted((root / "recall").glob("*.json"))]
    calls = [load(p) for p in (root / "calls").glob("*.json")]
    pilot_calls = []
    if (root / "default-session-pilot").exists():
        pilot_calls = [r for r in calls if r["phase"].startswith("recall-")
                       and not r["phase"].startswith("recall-isolated-")]
        calls = [r for r in calls if r not in pilot_calls]
    embeddings = [load(p) for p in (root / "embeddings").glob("*.json")]
    logfile = root.parent / (root.name + ".log")
    logtext = logfile.read_text() if logfile.exists() else ""
    validation_retries = len(re.findall(r"validation retry \d+/\d+", logtext))
    usage = Counter()
    for call in calls:
        for key, value in call.get("response", {}).get("usage", {}).items():
            if isinstance(value, (int, float)):
                usage[key] += value
    finish = Counter(choice.get("finish_reason") for r in calls
                     for choice in r.get("response", {}).get("choices", []))
    checks, retrieval = [], []
    indexed_chunks = source_chunks(root, manifest["arm"])
    for recall in recalls:
        case = cases[recall["id"]]
        answer = "\n".join(re.split(r"\n\s*Evidence:\s*\n", a.get("text", ""),
                                    maxsplit=1, flags=re.I)[0] for a in recall["answers"])
        groups = case["answer_checks"]
        matches = [any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", answer, re.I)
                       for term in alternatives) for alternatives in groups]
        checks.append({"id": recall["id"], "eligible": bool(groups),
                       "matches": matches, "passed": bool(groups) and all(matches)})
        if indexed_chunks:
            refs = [r for answer in recall["answers"] for r in answer.get("metadata", {}).get("evidence", [])]
            direct = {r["artifact_id"] for r in refs if r.get("kind") == "graph_node"
                      and r.get("role") == "used_as_context" and r["artifact_id"] in indexed_chunks}
            lineage = direct | {r["chunk_id"] for r in refs if r.get("chunk_id") in indexed_chunks}
            direct_coverage = coverage([indexed_chunks[c] for c in direct], case["evidence"])
            lineage_coverage = coverage([indexed_chunks[c] for c in lineage], case["evidence"])
            retrieval.append({"id": recall["id"], "raw_chunks_in_graph_context": len(direct),
                              "source_lineage_chunks": len(lineage),
                              "raw_chunk_evidence_coverage": direct_coverage,
                              "source_lineage_evidence_coverage": lineage_coverage,
                              "raw_chunk_complete": bool(direct_coverage) and min(direct_coverage) >= .999,
                              "source_lineage_complete": bool(lineage_coverage) and min(lineage_coverage) >= .999})
    latencies = [r["seconds"] for r in recalls]
    sessions = [r.get("session_id") for r in recalls]
    isolated = len({s for s in sessions if s}) == len(recalls) and all(
        r.get("history_audit_captured") for r in recalls)
    current_time = progress.get("finished_unix", time.time())
    result = {"arm": manifest["arm"], "complete": progress.get("passed", False) and
              len(remembered) == 13 and len(recalls) == len(cases) and isolated,
              "documents": len(remembered), "questions": len(recalls),
              "llm_concurrency": manifest.get("llm_concurrency", 4),
              "session_isolation_complete": isolated,
              "unique_question_sessions": len({s for s in sessions if s}),
              "queries_with_captured_history_audit": sum(bool(r.get("history_audit_captured")) for r in recalls),
              "planned_chunks": progress.get("planned_chunks"),
              "effective_chunk_budget": progress.get("effective_chunk_budget"),
              "embedding_tokenizer": progress.get("embedding_tokenizer"),
              "graph": {k: v for k, v in (progress.get("graph") or {}).items()
                        if k in ("nodes", "edges", "document_chunks")},
              "ingestion_seconds": sum(r["seconds"] for r in remembered),
              "recall_seconds": sum(latencies),
              "recall_p50_seconds": statistics.median(latencies) if latencies else None,
              "recall_p95_seconds": percentile(latencies, .95),
              "elapsed_seconds_including_pilot": current_time - progress.get(
                    "initial_started_unix", progress["started_unix"]),
              "archived_default_session_pilot_calls": len(pilot_calls),
              "llm_calls": len(calls), "llm_usage": dict(usage),
              "schema_validation_retries": validation_retries,
              "calls_with_explicit_output_token_limit": sum(any(key in r.get("forwarded", {})
                    for key in ("max_tokens", "max_completion_tokens")) for r in calls),
              "largest_completion_tokens": max((r.get("response", {}).get("usage", {}).get(
                    "completion_tokens", 0) for r in calls), default=0),
              "longest_upstream_seconds": max((r.get("upstream_seconds", 0)
                                                for r in calls), default=0),
              "http_statuses": dict(Counter(str(r["status"]) for r in calls)),
              "finish_reasons": dict(finish),
              "embedding_requests_including_pilot": len(embeddings),
              "embedding_inputs": sum(r.get("input_count", 0) for r in embeddings),
              "embedding_failed_requests": sum(r["status"] != 200 for r in embeddings),
              "fact_check_eligible": sum(c["eligible"] for c in checks),
              "fact_check_passed": sum(c["passed"] for c in checks),
              "fact_checks": checks,
              "retrieval_provenance": retrieval,
              "raw_chunk_evidence_complete": sum(r["raw_chunk_complete"] for r in retrieval),
              "source_lineage_evidence_complete": sum(r["source_lineage_complete"] for r in retrieval),
              "limitations": ["Fact checks are lexical spot checks, not answer correctness or grounding judgments.",
                               "Embedding request totals include archived default-session pilot requests where present.",
                               "Source lineage is provenance, not proof that the answer used or preserved a fact.",
                               "Whole-configuration comparison: chunk sizes and chunk counts differ."]}
    for stage in ("remember", "recall"):
        stage_calls = [r for r in calls if r["phase"].startswith(stage + "-")]
        prompts = [r["response"]["usage"]["prompt_tokens"] for r in stage_calls
                   if r.get("response", {}).get("usage")]
        result[stage + "_prompt_tokens_p50"] = statistics.median(prompts) if prompts else None
        result[stage + "_prompt_tokens_p95"] = percentile(prompts, .95)
    return result, recalls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assessments", type=Path)
    p.add_argument("--adaptive-root", type=Path,
                   help="Compare a fresh corrected adaptive index with the preserved stock index")
    p.add_argument("--baseline-runner-sha",
                   help="Explicitly audited stock runner SHA when comparing a changed runner")
    p.add_argument("--allow-concurrency-change", action="store_true",
                   help="Acknowledge that timings are not a controlled context-only comparison")
    a = p.parse_args()
    cases = {c["id"]: c for c in load(a.cases)}
    roots = {"current": a.run / "current", "adaptive": a.adaptive_root or a.run / "adaptive"}
    manifests = [load(roots[arm] / "manifest.json") for arm in roots]
    concurrency = check_concurrency(manifests, a.allow_concurrency_change)
    for key in ("input_sha256", "backend", "model", "embedding_endpoint", "embedding_model",
                "embedding_digest", "smoke", "capture_sha256", "recall_session_policy"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError("Arms are not comparable: manifest mismatch for " + key)
    if manifests[0]["runner_sha256"] != manifests[1]["runner_sha256"]:
        if not a.adaptive_root or a.baseline_runner_sha != manifests[0]["runner_sha256"]:
            raise ValueError("Changed runner requires an explicitly audited baseline runner SHA")
        if manifests[1].get("chunk_context_policy") != "breadcrumb":
            raise ValueError("Changed-runner comparison requires the breadcrumb fix")
    states = [load(roots[arm] / "progress.json") for arm in roots]
    for key in ("cognee_head", "runtime", "embedding_tokenizer", "effective_chunk_budget"):
        if states[0][key] != states[1][key]:
            raise ValueError("Arms are not comparable: runtime mismatch for " + key)
    report, private = {}, {}
    for arm in ("current", "adaptive"):
        report[arm], private[arm] = analyze(roots[arm], cases)
    report["comparison"] = {arm: {"runner_sha256": manifest["runner_sha256"],
        "llm_concurrency": manifest.get("llm_concurrency", 4),
        "chunk_context_policy": manifest.get("chunk_context_policy", "stock" if arm == "current" else "raw")}
        for arm, manifest in zip(roots, manifests)}
    report["comparison"]["concurrency_changed"] = concurrency[0] != concurrency[1]
    if a.assessments:
        report["reviewed_quality"] = quality_summary(load(a.assessments), cases)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    review = []
    for case in cases.values():
        review.append({"id": case["id"], "question": case["question"], "gold": case["evidence"],
                       "answers": {arm: next((r["answers"] for r in rows if r["id"] == case["id"]), [])
                                   for arm, rows in private.items()},
                       "assessment": {"current": None, "adaptive": None}})
    (a.output.parent / (a.output.stem + "-answer-review-private.json")).write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps({arm: {k: result[k] for k in ("complete", "documents", "questions",
          "planned_chunks", "fact_check_eligible", "fact_check_passed")}
          for arm, result in report.items() if arm in ("current", "adaptive")}))


if __name__ == "__main__":
    main()
