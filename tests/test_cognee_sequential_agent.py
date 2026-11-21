"""Guard the sequential-search and evidence-based stopping contract."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "experiments/adaptive-chunking/cognee/sequential.py"
spec = importlib.util.spec_from_file_location("sequential_research", path)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


class SequentialAgentContract(unittest.TestCase):
    def action(self, **updates):
        value = {"action": "search", "query": "What follows the first arc?",
                 "decision_summary": "The transition is missing.",
                 "remaining_gaps": ["transition"], "answer": ""}
        return {**value, **updates}

    def test_batch_and_repeat_cannot_stand_in_for_observing_a_new_answer(self):
        with self.assertRaisesRegex(ValueError, "only one query"):
            S.validate_action(self.action(queries=["one", "two"]), set(), set())
        with self.assertRaisesRegex(ValueError, "already observed"):
            S.validate_action(self.action(), {"what follows the first arc?"}, set())
        self.assertEqual(S.validate_action(self.action(), set(), set())["action"], "search")

    def test_finish_cannot_silently_ignore_declared_missing_evidence(self):
        with self.assertRaisesRegex(ValueError, "no remaining gaps"):
            S.validate_action(self.action(action="finish", query="", answer="A claim [S001]."),
                              set(), {"S001"})

    def test_answer_must_cite_retrieved_sources(self):
        value = self.action(action="finish", query="", remaining_gaps=[], answer="A claim [S002].")
        with self.assertRaisesRegex(ValueError, "actually retrieved"):
            S.validate_action(value, set(), {"S001"})
        value["answer"] = "A supported claim [S001]."
        self.assertEqual(S.validate_action(value, set(), {"S001"})["action"], "finish")

    def test_unused_action_field_can_be_omitted(self):
        search = self.action()
        del search["answer"]
        self.assertEqual(S.validate_action(search, set(), set())["answer"], "")
        finish = self.action(action="finish", remaining_gaps=[], answer="A claim [S001].")
        del finish["query"]
        self.assertEqual(S.validate_action(finish, set(), {"S001"})["query"], "")

    def test_json_fence_does_not_change_action_or_accept_extra_prose(self):
        import json
        encoded = json.dumps(self.action())
        self.assertEqual(S.parse_action(encoded), S.parse_action("```json\n" + encoded + "\n```"))
        with self.assertRaises(json.JSONDecodeError):
            S.parse_action("An explanation\n```json\n" + encoded + "\n```")


if __name__ == "__main__":
    unittest.main()
