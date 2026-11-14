import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("hibana_replay", Path(__file__).resolve().parents[1] / "experiments/hibana/replay.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


class Stream:
    def __init__(self, events):
        self.lines = [b"data: " + json.dumps(e).encode() + b"\n\n" for e in events] + [b"data: [DONE]\n\n"]

    def __enter__(self):
        return iter(self.lines)

    def __exit__(self, *args):
        return False


class ReplayTest(unittest.TestCase):
    def test_role_only_events_do_not_set_ttft_and_batched_tokens_do_not_become_itl(self):
        replay = object.__new__(R.Replay)
        replay.base, replay.model = "http://unused", "model"
        events = [{"choices": [{"delta": {"role": "assistant"}}]},
                  {"choices": [{"delta": {"content": "first"}}]},
                  {"choices": [{"delta": {"content": " last"}, "finish_reason": "stop"}]},
                  {"choices": [], "usage": {"completion_tokens": 5}}]
        with patch.object(R.urllib.request, "urlopen", return_value=Stream(events)), patch.object(R.time, "monotonic", side_effect=[0, .1, .3, .4, .5, .6]):
            row = replay.chat("fixture", [], 10)
        self.assertEqual(row["ttft_seconds"], .3)
        self.assertAlmostEqual(row["chunk_gaps_seconds"][0], .1)
        self.assertAlmostEqual(row["tpot_seconds"], .025)
        self.assertEqual(row["text"], "first last")
        self.assertTrue(row["done"])

    def test_valid_json_cut_off_at_length_or_missing_done_is_not_a_completed_task(self):
        replay = object.__new__(R.Replay)
        replay.lock, replay.rows = threading.Lock(), []
        with tempfile.TemporaryDirectory() as tmp:
            replay.out = Path(tmp)
            for i, (finish, done) in enumerate((("length", True), ("stop", False), ("grammar", True))):
                row = {"name": f"cutoff-{i}", "text": '{"value":323}', "finish_reason": finish, "done": done}
                self.assertFalse(replay.record(row, True)["passed"])

    def test_concurrent_session_throughput_uses_elapsed_wall_time(self):
        replay = object.__new__(R.Replay)
        replay.summary = {"session_turns": 1}
        replay.rows = [{"name": f"session-{i}-turn-0", "started_unix": 100, "seconds": 10,
                        "passed": True, "usage": {"completion_tokens": 20}, "ttft_seconds": 1, "tpot_seconds": .1} for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            replay.out = Path(tmp)
            replay.finish()
        group = replay.summary["groups"]["session"]
        self.assertEqual(group["validated_sessions_per_hour"], 1440)
        self.assertEqual(group["output_tokens_per_wall_second"], 8)


class PublicationTest(unittest.TestCase):
    def test_cognee_receipt_keeps_metrics_without_publishing_campaign_answers(self):
        spec = importlib.util.spec_from_file_location("cognee_analyze",
            Path(__file__).resolve().parents[1] / "experiments/hibana/cognee/analyze.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        private = {"arms": {"fast": {"remember_seconds": 65.67,
            "answers": [{"id": "case", "reader": "native", "seconds": 1,
                "text": "PRIVATE_ANSWER_SENTINEL", "question": "PRIVATE_QUESTION_SENTINEL",
                "expected": "PRIVATE_GOLDEN_SENTINEL", "result_count": 1}],
            "entity_edges": [{"claim": "PRIVATE_GRAPH_SENTINEL"}]}}}
        public = module.public_receipt(private)
        self.assertNotIn("PRIVATE_", json.dumps(public))
        self.assertEqual(public["arms"]["fast"]["remember_seconds"], 65.67)
        self.assertEqual(len(public["arms"]["fast"]["answers"][0]["answer_sha256"]), 64)
        self.assertEqual(private["arms"]["fast"]["answers"][0]["text"], "PRIVATE_ANSWER_SENTINEL")


if __name__ == "__main__":
    unittest.main()
