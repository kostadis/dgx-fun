"""Retrieval scoring must respect source identity, evidence unions and budgets."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

if any(importlib.util.find_spec(name) is None for name in ("numpy", "pandas", "tiktoken")):
    raise unittest.SkipTest("Install experiments/adaptive-chunking/requirements.txt for these optional experiment tests")

ROOT = Path(__file__).resolve().parents[1] / "experiments/adaptive-chunking"


def load(name):
    spec = importlib.util.spec_from_file_location("adaptive_trial_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R, P = load("run"), load("prepare")
A = load("analyze")


class AdaptiveTrialTests(unittest.TestCase):
    def test_overlap_is_not_counted_twice_and_gaps_are_not_covered(self):
        self.assertEqual(R.span_coverage([(0, 8), (2, 6)], (0, 10)), .8)
        self.assertEqual(R.span_coverage([(0, 4), (6, 10)], (0, 10)), .8)
        self.assertEqual(R.span_coverage([(0, 6), (4, 12)], (0, 10)), 1.)

    def test_same_offsets_in_a_different_document_cannot_satisfy_evidence(self):
        selected = [{"doc_id": "wrong", "start": 0, "end": 100}]
        evidence = [{"doc_id": "right", "start": 0, "end": 100}]
        self.assertEqual(R.evidence_coverage(selected, evidence), [0.])

    def test_budget_counts_headers_and_separators_and_never_truncates_chunks(self):
        # Deterministic tokenizer substitute isolates the budgeting contract.
        original = R.count_tokens
        R.count_tokens = len
        try:
            chunks = [{"retrieval_text": "Header\nABCDE"}, {"retrieval_text": "XX"}, {"retrieval_text": "Z"}]
            chosen = R.select_budget(chunks, [0, 1, 2], 15)
            self.assertEqual(chosen, chunks[:1] + chunks[2:])
            self.assertLessEqual(len("\n\n".join(c["retrieval_text"] for c in chosen)), 15)
        finally:
            R.count_tokens = original

    def test_renderer_preserves_lists_tables_headings_and_source_coordinates(self):
        node = {"name": "Room", "entries": ["The {@item Orb|T14} is hidden.",
            {"type": "list", "items": ["north", "south"]},
            {"type": "table", "colLabels": ["Roll", "Result"], "rows": [["1", "Secret door"]]}]}
        doc = P.render_document(node, 0)
        self.assertIn("The Orb is hidden.", doc["text"])
        self.assertIn("- north\n- south", doc["text"])
        self.assertIn("| 1 | Secret door |", doc["text"])
        for block in doc["blocks"]:
            self.assertEqual(doc["text"][block["start"]:block["end"]], block["text"])

    def test_fixture_must_match_the_frozen_source(self):
        corpus = {"documents": [{"id": "room", "text": "gold"}]}
        case = {"id": "q", "evidence": [{"doc_id": "room", "start": 0, "end": 4, "quote": "gold"}]}
        R.validate_cases([case], corpus)
        case["evidence"][0]["quote"] = "invented"
        with self.assertRaisesRegex(ValueError, "frozen source"):
            R.validate_cases([case], corpus)

    def test_rich_module_entries_keep_list_names_and_reference_content(self):
        refs = {"creature|PaBTSO|Cave Guard": {"name": "Cave Guard", "source": "PaBTSO", "page": 204,
            "hp": {"average": 30}, "action": [{"name": "Stun", "entries": ["Make a {@dc 15} save."]}]}}
        node = {"name": "Chapter", "page": 20, "entries": [
            {"type": "list", "items": [{"type": "item", "name": "Reward", "entries": ["Take 50 gp."]}]},
            {"type": "statblock", "tag": "creature", "name": "Cave Guard", "source": "PaBTSO"},
            {"type": "gallery", "images": [{"type": "image", "title": "The cavern"}]}]}
        resolved = set()
        doc = P.render_document(node, 3, "pabtso", refs, resolved)
        self.assertEqual(doc["id"], "pabtso-03")
        self.assertIn("Reward: Take 50 gp.", doc["text"])
        self.assertIn("hp: average: 30", doc["text"])
        self.assertIn("Make a DC 15 save.", doc["text"])
        self.assertIn("[Map/image: The cavern]", doc["text"])
        self.assertEqual(resolved, set(refs))
        self.assertTrue(any(b["page"] == 204 for b in doc["blocks"]))

    def test_inline_navigation_flags_do_not_replace_visible_labels(self):
        self.assertEqual(P.inline("See {@area chapter 1|023|x}."), "See chapter 1.")
        self.assertEqual(P.inline("{@quickref cover||3}"), "cover")
        self.assertEqual(P.inline("{@b Use {@spell light|PHB}.}"), "Use light.")

    def test_token_windows_preserve_unicode_and_cover_the_full_source(self):
        text = "".join(f"Room {i}: 灰色 ruins — a dragon 🐲 waits.\n" for i in range(30))
        windows = R.token_windows(text, size=40, overlap=8)
        end = 0
        for window in windows:
            start = text.find(window, max(0, end - len(window)))
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(start, end)
            self.assertNotIn("\ufffd", window)
            end = start + len(window)
        self.assertEqual(end, len(text))

    def test_public_export_does_not_copy_private_questions_or_answers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "results.json").write_text(json.dumps({"budgets": [1200], "questions": 1,
                "source_text": "PRIVATE_MODULE_TEXT"}))
            row = {"id": "q1", "question": "PRIVATE_QUESTION", "answer": "PRIVATE_ANSWER",
                   "budgets": {"1200": {"complete": True, "selected": ["PRIVATE_PASSAGE"]}}}
            (root / "retrieval-private.json").write_text(json.dumps({"fixed_600": [row], "adaptive_4metric": [row]}))
            (root / "reader-summary.json").write_text(json.dumps({"reader": "test", "response": "PRIVATE_RESPONSE"}))
            exported = A.export(root)
            self.assertNotIn("PRIVATE_", json.dumps(exported))
            self.assertEqual(exported["paired_comparison"]["1200"]["same_outcome"], ["q1"])


if __name__ == "__main__":
    unittest.main()
