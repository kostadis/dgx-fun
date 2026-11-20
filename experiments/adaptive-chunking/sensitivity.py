"""Post-hoc budget sensitivity using the first run's exact cached vectors.

No new model calls, chunking choices or question changes. Primary budgets stay
1,200 and 2,400; these additional budgets check whole-chunk packing effects.
"""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np

from run import count_tokens, digest, evidence_coverage, select_budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.run / "results.json").read_text())
    if digest(args.cases.read_text()) != meta["cases_sha256"]:
        raise ValueError("Question fixture changed")
    cases = json.loads(args.cases.read_text())
    chunks = json.loads((args.run / "chunks-private.json").read_text())
    intrinsic = json.loads((args.run / "intrinsic.json").read_text())
    chunks["adaptive_4metric"] = [c for doc_id, scores in intrinsic.items()
        for c in chunks[scores["selected"]] if c["doc_id"] == doc_id]
    cache = sqlite3.connect("file:" + str((args.run / "embeddings.sqlite3").resolve()) + "?mode=ro", uri=True)

    def vectors(texts):
        rows = []
        for text in texts:
            key = digest(meta["embedding_endpoint"] + "\n" + meta["embedding_digest"] + "\n" + text)
            row = cache.execute("SELECT vector FROM vectors WHERE key=?", (key,)).fetchone()
            if not row:
                raise ValueError("Sensitivity must use existing vectors only")
            rows.append(np.frombuffer(row[0], dtype=np.float32))
        return np.asarray(rows)

    qvectors = vectors(["Instruct: Given a Dungeons & Dragons adventure question, retrieve relevant module passages.\nQuery: "
                        + c["question"] for c in cases])
    results = {}
    for method in ("fixed_600", "recursive_600", "adaptive_4metric"):
        items = chunks[method]
        scores = qvectors @ vectors([c["retrieval_text"] for c in items]).T
        results[method] = {}
        for budget in (1500, 1800, 3000, 4800):
            complete, tokens = 0, []
            for case, sims in zip(cases, scores):
                selected = select_budget(items, np.argsort(-sims, kind="stable").tolist(), budget)
                coverage = evidence_coverage(selected, case["evidence"])
                complete += bool(coverage) and all(v >= 1 - 1e-9 for v in coverage)
                tokens.append(count_tokens("\n\n".join(c["retrieval_text"] for c in selected)))
            results[method][str(budget)] = {"complete_questions": complete, "questions": len(cases),
                                            "mean_retrieved_tokens": float(np.mean(tokens))}
    output = {"status": "post-hoc; primary budgets unchanged", "new_model_calls": 0, "retrieval": results}
    (args.run / "budget-sensitivity.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
