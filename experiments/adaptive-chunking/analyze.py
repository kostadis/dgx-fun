"""Export aggregate receipts without module text, questions, answers or vectors."""
import argparse
import json
from pathlib import Path


def export(run):
    source = json.loads((run / "results.json").read_text())
    fields = ("corpus_sha256", "cases_sha256", "source_sha256", "upstream_commit", "budgets",
        "module_title", "module_id", "resolved_references", "reference_export_sha256",
        "embedding_endpoint", "embedding_model", "embedding_digest", "runner_sha256",
        "retrieval_candidate_pool", "candidate_methods", "selection_metrics",
        "coherence_max_windows_per_document", "selection_weights", "module_sections", "questions",
        "source_tokens_o200k", "selection_counts", "chunk_counts", "retrieval", "elapsed_seconds",
        "embedding_seconds_this_run", "embedding_requests_this_run", "embedding_texts_this_run")
    result = {key: source[key] for key in fields if key in source}
    retrieval = json.loads((run / "retrieval-private.json").read_text())
    baseline = {r["id"]: r for r in retrieval["fixed_600"]}
    adaptive = {r["id"]: r for r in retrieval["adaptive_4metric"]}
    result["paired_comparison"] = {}
    for budget in result["budgets"]:
        key = str(budget)
        wins, losses, ties = [], [], []
        for case_id, row in baseline.items():
            old = row["budgets"][key]["complete"]
            new = adaptive[case_id]["budgets"][key]["complete"]
            (wins if new and not old else losses if old and not new else ties).append(case_id)
        result["paired_comparison"][key] = {"adaptive_only": wins, "fixed_only": losses, "same_outcome": ties}
    reader = run / "reader-summary.json"
    if reader.exists():
        reader_data = json.loads(reader.read_text())
        result["answer_spot_check"] = {key: reader_data[key]
            for key in ("reader", "endpoint", "budget", "case_ids", "methods") if key in reader_data}
    sensitivity = run / "budget-sensitivity.json"
    if sensitivity.exists():
        result["posthoc_budget_sensitivity"] = json.loads(sensitivity.read_text())
    result["limitations"] = [
        "One module representation; manually selected questions, not a representative sample.",
        "Four-metric section-level adaptation with a bounded candidate set; not a full paper replication.",
        "Designated source-span coverage can miss alternative correct evidence.",
        "Dense vector retrieval only; not Cognee graph ingestion or production retrieval parity.",
        "Fact-pattern and verbatim-quote checks require manual review before quality conclusions.",
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(export(args.run), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
