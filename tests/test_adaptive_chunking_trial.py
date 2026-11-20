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


class MempalaceArmTests(unittest.TestCase):
    TEXT = "\n\n".join(f"Paragraph {i}. " + ("The gnolls guard room %d with spears. " % i) * 6 for i in range(12))

    def test_mempalace_pieces_roundtrip(self):
        pieces = R.mempalace_pieces(self.TEXT)
        contents = [c["content"] for c in R._mempalace_chunk_text()(self.TEXT, "x")]
        self.assertEqual([self.TEXT[s:e] for s, e in pieces], contents)
        starts = [s for s, _ in pieces]
        self.assertEqual(starts, sorted(starts))
        covered = set()
        for s, e in pieces:
            covered.update(range(s, e))
        missing = [i for i, ch in enumerate(self.TEXT) if not ch.isspace() and i not in covered]
        self.assertEqual(missing, [])

    def test_mempalace_pieces_repeated_paragraph(self):
        para = "Read aloud: the obelisk hums with a low violet light that fills the chamber. " * 3
        filler = "\n\n".join("Filler %d " % i + "x " * 300 for i in range(3))
        text = "\n\n".join([para, filler, para, filler, para])
        starts = [s for s, _ in R.mempalace_pieces(text)]
        self.assertTrue(all(a < b for a, b in zip(starts, starts[1:])))

    def test_mempalace_pieces_short_section(self):
        self.assertEqual(R.mempalace_pieces("tiny"), [(0, 4)])

    def test_mempalace_isolated(self):
        import os, subprocess, sys
        with tempfile.TemporaryDirectory() as home:
            code = ("import importlib.util,os;"
                    f"s=importlib.util.spec_from_file_location('r',{str(ROOT / 'run.py')!r});"
                    "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                    "m.mempalace_pieces('alpha beta gamma '*200);"
                    "import sys;assert not any(k.startswith('mempalace') for k in sys.modules);"
                    "print(os.environ['MEMPALACE_PALACE_PATH'])")
            env = {**os.environ, "HOME": home, "MEMPALACE_SRC": str(R.MEMPALACE_ROOT), "MEMPALACE_PALACE_PATH": "/should/be/overwritten"}
            out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
            self.assertEqual(os.listdir(home), [])
            self.assertNotIn("mempalace", out.stderr)
            self.assertTrue(Path(out.stdout.strip()).resolve().is_relative_to((ROOT / "raw/scratch-palace").resolve()))


class ExtraArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs = []
        for d in range(2):
            paras = ["Section %d paragraph %d. " % (d, i) + "The drow patrol crosses the bridge at dusk. " * 12
                     for i in range(10)]
            text, blocks, pos = "", [], 0
            for i, para in enumerate(paras):
                if text:
                    text += "\n\n"
                start = len(text)
                text += para
                blocks.append({"start": start, "end": len(text), "path": ["Chapter %d" % d, "Area %d" % i]})
            docs.append({"id": "d%d" % d, "text": text, "blocks": blocks})
        cls.corpus = {"module_title": "Test Module", "documents": docs}
        cls.chunks = R.prepare_chunks(cls.corpus, R.upstream_modules((ROOT / ".upstream").resolve()))

    def test_extra_arms_present(self):
        for method in ("mempalace_800", "mempalace_800_crumb", "recursive_600_nocrumb"):
            self.assertIn(method, self.chunks)
            self.assertTrue(self.chunks[method])

    def test_nocrumb_retrieval_text(self):
        for method in R.NO_CRUMB:
            for c in self.chunks[method]:
                self.assertEqual(c["retrieval_text"], c["text"])
                self.assertIs(c["crumb"], False)
        for c in self.chunks["mempalace_800_crumb"]:
            self.assertTrue(c["retrieval_text"].startswith("Test Module > "))
            self.assertIs(c["crumb"], True)

    def test_recursive_nocrumb_same_bounds(self):
        bounds = lambda m: [(c["doc_id"], c["start"], c["end"]) for c in self.chunks[m]]
        self.assertEqual(bounds("recursive_600_nocrumb"), bounds("recursive_600"))
        self.assertEqual(bounds("mempalace_800"), bounds("mempalace_800_crumb"))

    def test_candidates_unchanged(self):
        self.assertEqual(R.CANDIDATES, ("recursive_600", "recursive_1100", "module_headings_600"))


if __name__ == "__main__":
    unittest.main()
