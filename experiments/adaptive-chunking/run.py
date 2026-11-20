"""Controlled local module retrieval: frozen source spans, fixed context budgets.

Uses the pinned upstream splitters, four intrinsic metrics and selector.
The coreference metric and paper's cloud/GPU splitters are outside this pilot.
All source text, vectors, questions and retrieved passages stay under raw/.
"""
import argparse
import ast
from collections import Counter
import hashlib
from functools import lru_cache
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import cast
import urllib.request

import numpy as np
import pandas as pd
import tiktoken

PIN = "ea87ce8e1a97888f3f179e7f1359ff7f43fb179d"
CANDIDATES = ("recursive_600", "recursive_1100", "module_headings_600")
METHODS = ("fixed_600", *CANDIDATES)


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@lru_cache(maxsize=1)
def encoding():
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text):
    return len(encoding().encode(text, disallowed_special=()))


def token_windows(text, size=600, overlap=50):
    tokens = encoding().encode(text, disallowed_special=())
    decoded, offsets = encoding().decode_with_offsets(tokens)
    assert decoded == text
    offsets.append(len(text))
    return [text[offsets[i]:offsets[min(len(tokens), i + size)]]
            for i in range(0, len(tokens), size - overlap)]


def span_coverage(spans, gold):
    """Fraction of a source evidence span covered by the union of retrieved spans."""
    start, end = gold
    if end <= start:
        raise ValueError("Empty gold evidence")
    intervals = sorted((max(start, a), min(end, b)) for a, b in spans if a < end and b > start)
    covered, cursor = 0, start
    for a, b in intervals:
        covered += max(0, b - max(cursor, a))
        cursor = max(cursor, b)
    return covered / (end - start)


def select_budget(chunks, ranking, budget):
    """Whole chunks only; account for the actual rendered context and separators."""
    selected = []
    for index in ranking[:20]:
        candidate = [*selected, chunks[index]]
        context = "\n\n".join(c["retrieval_text"] for c in candidate)
        if count_tokens(context) <= budget:
            selected = candidate
    return selected


def evidence_coverage(selected, evidence):
    return [span_coverage([(c["start"], c["end"]) for c in selected if c["doc_id"] == gold["doc_id"]],
                          (gold["start"], gold["end"])) for gold in evidence]


class Embedder:
    def __init__(self, endpoint, model, cache, batch=16):
        self.endpoint, self.model, self.batch = endpoint.rstrip("/"), model, batch
        self.db = sqlite3.connect(cache)
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        self.requests = self.input_texts = 0
        self.seconds = 0.
        self.identity = None
        with urllib.request.urlopen(self.endpoint + "/api/tags", timeout=20) as response:
            tags = json.load(response)
        for item in tags["models"]:
            if item["name"] == model:
                self.identity = item["digest"]
        if self.identity is None:
            raise ValueError("Requested embedding model is not installed")

    def encode(self, sentences, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        keys = [digest(self.endpoint + "\n" + self.identity + "\n" + text) for text in sentences]
        values, missing = {}, {}
        for key, text in zip(keys, sentences):
            row = self.db.execute("SELECT vector FROM vectors WHERE key=?", (key,)).fetchone()
            if row:
                values[key] = np.frombuffer(row[0], dtype=np.float32).copy()
            else:
                missing[key] = text
        pending = list(missing)
        for i in range(0, len(pending), self.batch):
            group = pending[i:i + self.batch]
            request = urllib.request.Request(self.endpoint + "/api/embed", headers={"Content-Type": "application/json"},
                data=json.dumps({"model": self.model, "input": [missing[k] for k in group],
                                 "truncate": False}).encode())
            start = time.monotonic()
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.load(response)
            self.seconds += time.monotonic() - start
            if len(result["embeddings"]) != len(group):
                raise ValueError("Embedding result count mismatch")
            for key, vector in zip(group, result["embeddings"]):
                arr = np.asarray(vector, dtype=np.float32)
                if not np.isfinite(arr).all() or np.linalg.norm(arr) == 0:
                    raise ValueError("Invalid embedding")
                arr /= np.linalg.norm(arr)
                values[key] = arr
                self.db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?)", (key, arr.tobytes()))
            self.db.commit()
            self.requests += 1
            self.input_texts += len(group)
        return np.asarray([values[key] for key in keys])


def upstream_modules(path):
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if head != PIN or subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True):
        raise ValueError("Upstream checkout must be clean at the recorded pin")
    sys.path.insert(0, str(path / "src"))
    from adaptive_chunking import metrics, postprocessing
    from adaptive_chunking.splitters import RecursiveSplitter
    # analysis.py imports plotting dependencies that the selector does not use.
    # Load its exact unmodified function body, rather than installing the paper stack.
    tree = ast.parse((path / "src/adaptive_chunking/paper/analysis.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "find_best_method")
    namespace = {"pd": pd, "np": np, "Any": object, "cast": cast}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "upstream_find_best_method", "exec"), namespace)
    return metrics, postprocessing, RecursiveSplitter, namespace["find_best_method"]


def prepare_chunks(corpus, upstream):
    metrics, post, splitter_cls, selector = upstream
    chunks = {method: [] for method in METHODS}
    for doc in corpus["documents"]:
        text = doc["text"]
        for method in METHODS:
            if method == "fixed_600":
                pieces = token_windows(text)
            else:
                settings = {"chunk_size": 1100 if method == "recursive_1100" else 600,
                            "chunk_overlap": 50, "merging": "small_only", "min_chunk_tokens": 100,
                            "max_tokens_strategy": "chunk_size"}
                if method == "module_headings_600":
                    settings.update(is_separator_regex=True,
                                    separators=[r"\n(?=#{1,6} )", r"\n\n", r"\n", " ", ""])
                pieces = splitter_cls(**settings).split_text(text)
            pieces = post.repair_gaps_between_chunks(pieces, text)
            assert post.check_chunk_gaps(pieces, text), (doc["id"], method)
            bounds = post.find_chunks_start_and_end(pieces, text)
            for index, (piece, (start, end)) in enumerate(zip(pieces, bounds)):
                assert text[start:end] == piece
                block = next((b for b in doc["blocks"] if b["start"] <= start < b["end"]), doc["blocks"][0])
                prefix = corpus.get("module_title", "Temple of Elemental Evil") + " > " + " > ".join(block["path"])
                chunks[method].append({"id": f'{method}:{doc["id"]}:{index}', "doc_id": doc["id"],
                    "start": start, "end": end, "text": piece, "retrieval_text": prefix + "\n" + piece,
                    "tokens": count_tokens(piece)})
    return chunks


def intrinsic(corpus, chunks, embedder, upstream, output):
    metrics, post, splitter_cls, selector = upstream
    path = output / "intrinsic.json"
    records = json.loads(path.read_text()) if path.exists() else {}
    for doc in corpus["documents"]:
        if doc["id"] in records:
            continue
        rows = {}
        split_points = [b["start"] for b in doc["blocks"]][1:]
        for method in CANDIDATES:
            pieces = [c["text"] for c in chunks[method] if c["doc_id"] == doc["id"]]
            vectors = embedder.encode(pieces)
            step = max(1, math.ceil(len(pieces) / 16))
            rows[method] = {
                "size_compliance": float(metrics.compute_size_compliance(pieces)),
                "block_integrity": float(metrics.compute_block_integrity(pieces, split_points, doc["text"])),
                "intrachunk_cohesion": metrics.compute_intrachunk_cohesion(
                    pieces, doc["text"], split_points, embedder, chunk_embeddings=vectors),
                "contextual_coherence": metrics.compute_contextual_coherence(
                    pieces, doc["text"], embedder, window_step=step, chunk_embeddings=vectors),
            }
            rows[method] = {k: None if v is None else float(v) for k, v in rows[method].items()}
        frame = pd.DataFrame(rows)
        winner, scores = selector(frame, pd.Series(1., index=frame.index))
        records[doc["id"]] = {"selected": winner, "metrics": rows,
                               "scores": dict(zip(frame.columns, scores))}
        write(path, records)
        print(json.dumps({"stage": "intrinsic", "document": doc["id"], "selected": winner,
                          "completed_documents": len(records), "embedding_requests": embedder.requests}), flush=True)
    chunks["adaptive_4metric"] = [c for doc in corpus["documents"]
        for c in chunks[records[doc["id"]]["selected"]] if c["doc_id"] == doc["id"]]
    return records


def validate_cases(cases, corpus):
    documents = {d["id"]: d for d in corpus["documents"]}
    seen = set()
    for case in cases:
        if case["id"] in seen:
            raise ValueError("Duplicate case ID")
        seen.add(case["id"])
        for evidence in case["evidence"]:
            doc = documents[evidence["doc_id"]]
            if doc["text"][evidence["start"]:evidence["end"]] != evidence["quote"]:
                raise ValueError("Gold evidence does not match frozen source")


def retrieve(cases, chunks, embedder, budgets, output):
    queries = ["Instruct: Given a Dungeons & Dragons adventure question, retrieve relevant module passages.\nQuery: "
               + case["question"] for case in cases]
    qvectors = embedder.encode(queries)
    summaries, raw = {}, {}
    for method, items in chunks.items():
        vectors = embedder.encode([c["retrieval_text"] for c in items])
        similarities = qvectors @ vectors.T
        results = []
        for case, scores in zip(cases, similarities):
            ranking = np.argsort(-scores, kind="stable").tolist()
            row = {"id": case["id"], "category": case["category"], "question": case["question"],
                   "evidence": case["evidence"], "budgets": {}}
            top5 = evidence_coverage([items[i] for i in ranking[:5]], case["evidence"])
            row["top5_complete"] = bool(top5) and all(v >= 1 - 1e-9 for v in top5)
            for budget in budgets:
                selected = select_budget(items, ranking, budget)
                coverage = evidence_coverage(selected, case["evidence"])
                row["budgets"][str(budget)] = {"selected": selected, "coverage": coverage,
                    "complete": bool(coverage) and all(value >= 1 - 1e-9 for value in coverage),
                    "tokens": count_tokens("\n\n".join(c["retrieval_text"] for c in selected))}
            results.append(row)
        raw[method] = results
        summaries[method] = {str(budget): {
            "complete_questions": sum(r["budgets"][str(budget)]["complete"] for r in results),
            "questions": len(results),
            "top5_complete_questions": sum(r["top5_complete"] for r in results),
            "mean_evidence_coverage": float(np.mean([np.mean(r["budgets"][str(budget)]["coverage"]) for r in results])),
            "mean_retrieved_tokens": float(np.mean([r["budgets"][str(budget)]["tokens"] for r in results])),
            "by_category": {category: {"questions": sum(r["category"] == category for r in results),
                "complete": sum(r["budgets"][str(budget)]["complete"] for r in results if r["category"] == category)}
                for category in sorted({r["category"] for r in results})}
        } for budget in budgets}
        print(json.dumps({"stage": "retrieval", "method": method, "results": summaries[method]}), flush=True)
    write(output / "retrieval-private.json", raw)
    return summaries, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path(__file__).with_name(".upstream"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://192.168.1.121:11434")
    parser.add_argument("--model", default="qwen3-embedding:0.6b")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1200, 2400])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_root = Path(__file__).with_name("raw").resolve()
    if not args.output.resolve().is_relative_to(raw_root):
        parser.error("Output must stay beneath this experiment's ignored raw/ directory")
    corpus, cases = json.loads(args.corpus.read_text()), json.loads(args.cases.read_text())
    validate_cases(cases, corpus)
    inputs = {"corpus_sha256": digest(args.corpus.read_text()), "cases_sha256": digest(args.cases.read_text()),
              "source_sha256": corpus["source_sha256"], "upstream_commit": PIN, "budgets": args.budgets,
              "module_title": corpus.get("module_title", "Temple of Elemental Evil"),
              "module_id": corpus.get("module_id", "t14"),
              "resolved_references": corpus.get("resolved_references", 0),
              "reference_export_sha256": corpus.get("reference_export_sha256"),
              "embedding_endpoint": args.endpoint, "embedding_model": args.model,
              "retrieval_candidate_pool": 20,
              "candidate_methods": CANDIDATES, "selection_metrics": ["size_compliance", "block_integrity",
                  "intrachunk_cohesion", "contextual_coherence"],
              "coherence_max_windows_per_document": 16, "selection_weights": "equal; missing metrics ignored"}
    upstream = upstream_modules(args.upstream.resolve())
    embedder = Embedder(args.endpoint, args.model, args.output / "embeddings.sqlite3")
    inputs["embedding_digest"] = embedder.identity
    inputs["runner_sha256"] = digest(Path(__file__).read_text())
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(json.dumps(inputs)):
        raise ValueError("Refusing to resume with changed source, questions or configuration")
    write(manifest, inputs)
    started = time.monotonic()
    chunks = prepare_chunks(corpus, upstream)
    write(args.output / "chunks-private.json", chunks)
    intrinsic_scores = intrinsic(corpus, chunks, embedder, upstream, args.output)
    summaries, raw = retrieve(cases, chunks, embedder, args.budgets, args.output)
    result = {**inputs, "embedding_digest": embedder.identity,
              "module_sections": len(corpus["documents"]), "questions": len(cases),
              "source_tokens_o200k": sum(count_tokens(d["text"]) for d in corpus["documents"]),
              "selection_counts": dict(Counter(r["selected"] for r in intrinsic_scores.values())),
              "chunk_counts": {k: len(v) for k, v in chunks.items()}, "retrieval": summaries,
              "elapsed_seconds": time.monotonic() - started, "embedding_seconds_this_run": embedder.seconds,
              "embedding_requests_this_run": embedder.requests, "embedding_texts_this_run": embedder.input_texts}
    write(args.output / "results.json", result)
    print(json.dumps({"stage": "complete", "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
