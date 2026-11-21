"""The public report must not leak private corpus data or overstate completion."""
import importlib.util
import contextlib
import io
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "experiments/adaptive-chunking/cognee/analyze.py"
spec = importlib.util.spec_from_file_location("cognee_chunking_analysis", SOURCE)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
runner_spec = importlib.util.spec_from_file_location("cognee_chunking_runner", SOURCE.with_name("run.py"))
R = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(R)
audit_spec = importlib.util.spec_from_file_location("cognee_context_audit", SOURCE.with_name("audit_context.py"))
C = importlib.util.module_from_spec(audit_spec)
with patch.dict("sys.modules", {"analyze": A}):
    audit_spec.loader.exec_module(C)


class CogneeChunkingAnalysisTests(unittest.TestCase):
    def test_claimed_context_must_appear_in_actual_reader_request(self):
        text = "Book > Amber Charm\n\nChoose three allies."
        chunks = [{"id": "c", "text": text}]
        recalls = [{"llm_calls": [{"id": "reader"}], "answers": [{"metadata": {"evidence": [
            {"kind": "graph_node", "role": "used_as_context", "artifact_id": "c"}]}}]}]
        calls = [{"id": "reader", "forwarded": {"messages": [{"content": "No source text here"}]}}]
        with self.assertRaisesRegex(ValueError, "absent from actual reader"):
            C.verify_recall_context(chunks, calls, recalls)
        calls[0]["forwarded"]["messages"][0]["content"] = json.dumps({"text": text})
        self.assertEqual(C.verify_recall_context(chunks, calls, recalls),
                         {"queries_audited": 1, "contextual_chunk_appearances_verified": 1})

    def test_changed_concurrency_requires_explicit_comparison_and_is_reported(self):
        manifests = [{"arm": "current"}, {"arm": "adaptive", "llm_concurrency": 8}]
        with self.assertRaisesRegex(ValueError, "Concurrency differs"):
            A.check_concurrency(manifests, False)
        self.assertEqual(A.check_concurrency(manifests, True), [4, 8])
        self.assertEqual(A.check_concurrency([{}, {}], False), [4, 4])

    def test_boundary_fragment_keeps_parent_scope_without_rewriting_source(self):
        source = "### Amber Charm\nintro\nChoose three allies.\n### Blue Charm\nintro"
        start = source.index("Choose")
        raw = source[start:]
        prefix = "Book > Items > Amber Charm\n\n"
        chunk = {"start": start, "end": len(source), "text": raw, "retrieval_text": prefix + raw}
        fixed = R.adaptive_payload(source, chunk, "breadcrumb")
        self.assertTrue(fixed["text"].startswith(prefix + "Choose three allies."))
        self.assertEqual(fixed["text"][len(fixed["context_prefix"]):], source[start:])
        self.assertEqual((fixed["source_start"], fixed["source_end"]), (start, len(source)))
        self.assertEqual(R.adaptive_payload(source, chunk, "raw")["text"], raw)
        with self.assertRaisesRegex(ValueError, "raw source span"):
            R.adaptive_payload(source, {**chunk, "retrieval_text": prefix + raw + "altered"}, "breadcrumb")

    def test_context_is_not_counted_as_retrieved_source_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "adaptive-context"
            (root / "chunks").mkdir(parents=True)
            frozen = root.parent / "frozen"
            frozen.mkdir()
            raw, prefix = "Choose three allies.", "Book > Amber Charm\n\n"
            source = "### Amber Charm\n" + raw
            start = source.index(raw)
            chunk = {"doc_id": "d", "start": start, "end": len(source), "text": raw,
                     "retrieval_text": prefix + raw}
            for name, value in {"corpus": {"documents": [{"id": "d", "text": source}]},
                                "chunks": {"method": [chunk]}, "selection": {"d": {"selected": "method"}}}.items():
                (frozen / (name + ".json")).write_text(json.dumps(value))
            payload = R.adaptive_payload(source, chunk, "breadcrumb")
            census = root / "chunks/d.json"
            census.write_text(json.dumps({"doc_id": "d", "chunks": [{"id": "c", "index": 0, **payload}]}))
            indexed = A.source_chunks(root, "adaptive")
            self.assertEqual(A.coverage(indexed.values(), [{"doc_id": "d", "start": 0, "end": start}]), [0])
            self.assertEqual(A.coverage(indexed.values(), [{"doc_id": "d", "start": start, "end": len(source)}]), [1])
            payload["context_prefix"] = "incorrect"
            census.write_text(json.dumps({"doc_id": "d", "chunks": [{"id": "c", "index": 0, **payload}]}))
            with self.assertRaisesRegex(ValueError, "audited raw text"):
                A.source_chunks(root, "adaptive")

    def test_context_audit_rejects_a_pipeline_that_only_fixes_summarization(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("chunks", "planned-chunks", "calls", "embeddings"):
                (root / name).mkdir()
            raw, prefix = "Choose three allies.\n", "Book > Amber Charm\n\n"
            text = prefix + raw
            chunk = {"id": "c", "index": 0, "text": text, "raw_text": raw, "context_prefix": prefix}
            for name in ("chunks", "planned-chunks"):
                (root / name / "d.json").write_text(json.dumps({"chunks": [chunk]}))
            (root / "manifest.json").write_text(json.dumps({"chunk_context_policy": "breadcrumb"}))
            (root / "graph.json").write_text(json.dumps({"nodes": [["c", {"type": "DocumentChunk", "text": text}]]}))
            (root / "embeddings/e.json").write_text(json.dumps({"input_sha256": [hashlib.sha256(text.strip().encode()).hexdigest()]}))
            def capture(name, system, user):
                (root / "calls" / (name + ".json")).write_text(json.dumps({"id": name,
                    "phase": "remember-d", "status": 200, "forwarded": {"messages": [
                        {"role": "system", "content": system}, {"role": "user", "content": user}]}}))
            capture("summary", "Summarize the chunk for retrieval.", text)
            capture("graph", "Extract entities to build a knowledge graph", raw)
            with patch.object(C, "source_chunks", return_value={"c": {}}):
                with self.assertRaisesRegex(ValueError, "graph_extraction"):
                    C.audit(root)
                capture("graph", "Extract entities to build a knowledge graph", text)
                result = C.audit(root)
            self.assertEqual(result["chunks_received_by_stage"], {"summarization": 1, "graph_extraction": 1})
            self.assertEqual(result["chunks_with_exact_contextual_embedding_input"], 1)

    def fixture(self, root):
        def save(name, value):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        save("manifest.json", {"arm": "current", "private": "PRIVATE_SOURCE"})
        save("progress.json", {"passed": True, "started_unix": 1, "finished_unix": 11,
                               "planned_chunks": 78, "graph": {"document_chunks": 78,
                                                                 "private": "PRIVATE_GRAPH_TEXT"}})
        for i in range(13):
            save(f"remember/doc-{i}.json", {"seconds": 1, "result": "PRIVATE_SOURCE"})
        save("recall/q.json", {"id": "q", "question": "PRIVATE_QUESTION", "seconds": 2,
                               "session_id": "fresh-session", "history_audit_captured": True,
                               "answers": [{"text": "The purple stone. PRIVATE_ANSWER"}]})
        save("calls/call.json", {"status": 200, "phase": "recall-q", "request": {"private": "PRIVATE_PROMPT"},
                                "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5},
                                             "choices": [{"finish_reason": "stop"}]}})
        save("embeddings/embed.json", {"status": 200, "input_count": 3})
        return {"q": {"answer_checks": [["purple"]], "evidence": "PRIVATE_EVIDENCE"}}

    def test_public_export_is_allowlisted_and_does_not_copy_text(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cases = self.fixture(root)
            public, private = A.analyze(root, cases)
            self.assertTrue(public["complete"])
            self.assertNotIn("PRIVATE_", json.dumps(public))
            self.assertIn("PRIVATE_ANSWER", json.dumps(private))
            self.assertEqual(public["fact_check_passed"], 1)
            self.assertEqual(public["llm_usage"]["completion_tokens"], 5)

    def test_a_passed_partial_ingestion_is_not_a_complete_trial(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cases = self.fixture(root)
            (root / "remember/doc-12.json").unlink()
            self.assertFalse(A.analyze(root, cases)[0]["complete"])

    def test_empty_fact_checks_do_not_count_as_quality_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cases = self.fixture(root)
            cases["q"]["answer_checks"] = []
            result, _ = A.analyze(root, cases)
            self.assertEqual(result["fact_check_eligible"], 0)
            self.assertEqual(result["fact_check_passed"], 0)

    def test_percentile_interpolates_and_handles_no_observations(self):
        self.assertEqual(A.percentile([1, 2, 3, 4, 5], .95), 4.8)
        self.assertIsNone(A.percentile([], .95))

    def test_evidence_coverage_unions_overlap_and_respects_source_identity(self):
        gold = [{"doc_id": "right", "start": 0, "end": 10}]
        chunks = [{"doc_id": "wrong", "start": 0, "end": 10},
                  {"doc_id": "right", "start": 0, "end": 6},
                  {"doc_id": "right", "start": 3, "end": 8}]
        self.assertEqual(A.coverage(chunks, gold), [.8])
        self.assertEqual(A.coverage([], gold), [0.])

    def test_quality_export_does_not_copy_private_review_notes(self):
        labels = {"q01": {arm: {"correctness": "correct_and_complete", "grounding": "supported",
                                "notes": "PRIVATE_SOURCE_QUOTE"} for arm in ("current", "adaptive")}}
        result = A.quality_summary(labels, {"q01"})
        self.assertNotIn("PRIVATE_", json.dumps(result))
        self.assertEqual(result["counts"]["current"]["correctness"]["correct_and_complete"], 1)

    def test_quality_export_requires_a_complete_review(self):
        with self.assertRaisesRegex(ValueError, "every frozen question"):
            A.quality_summary({}, {"q01"})

    def test_citation_footer_cannot_satisfy_an_answer_fact_check(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cases = self.fixture(root)
            path = root / "recall/q.json"
            row = json.loads(path.read_text())
            row["answers"][0]["text"] = "Unknown.\n\nEvidence:\nThe purple stone."
            path.write_text(json.dumps(row))
            result, _ = A.analyze(root, cases)
            self.assertEqual(result["fact_check_passed"], 0)

    def test_recovered_schema_retry_is_reported_without_copying_log_text(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "current"
            root.mkdir()
            cases = self.fixture(root)
            (root.parent / "current.log").write_text("validation retry 1/3: PRIVATE_ERROR_TEXT")
            result, _ = A.analyze(root, cases)
            self.assertEqual(result["schema_validation_retries"], 1)
            self.assertNotIn("PRIVATE_", json.dumps(result))

    def test_default_session_answers_cannot_complete_an_independent_trial(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cases = self.fixture(root)
            path = root / "recall/q.json"
            row = json.loads(path.read_text())
            del row["session_id"]
            path.write_text(json.dumps(row))
            self.assertFalse(A.analyze(root, cases)[0]["complete"])

    def test_correct_but_unsupported_answer_is_not_counted_as_grounded(self):
        labels = {"q01": {arm: {"correctness": "correct_and_complete", "grounding": "supported"}
                           for arm in ("current", "adaptive")}}
        labels["q01"]["adaptive"]["grounding"] = "unsupported"
        result = A.quality_summary(labels, {"q01"})
        self.assertEqual(result["counts"]["current"]["grounded_complete"], 1)
        self.assertEqual(result["counts"]["adaptive"]["grounded_complete"], 0)

    def test_controller_rejects_accidental_reuse_without_clobbering_saved_status(self):
        trial_spec = importlib.util.spec_from_file_location("cognee_trial", SOURCE.with_name("trial.py"))
        trial = importlib.util.module_from_spec(trial_spec)
        trial_spec.loader.exec_module(trial)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            root = base / "raw/saved-trial"
            (root / "current").mkdir(parents=True)
            status = root / "control.json"
            status.write_text('{"passed": true}\n')
            with patch.object(trial, "HERE", base / "cognee"), \
                 patch("sys.argv", ["trial.py", "--output", str(root)]), \
                 contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as raised:
                trial.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(status.read_text(), '{"passed": true}\n')


if __name__ == "__main__":
    unittest.main()
