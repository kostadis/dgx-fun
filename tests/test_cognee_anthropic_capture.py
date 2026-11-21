"""Check provider translation preserves evidence, accounting and stop status."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "experiments/adaptive-chunking/cognee/anthropic_capture.py"
spec = importlib.util.spec_from_file_location("anthropic_trial_capture", path)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


class AnthropicTranslation(unittest.TestCase):
    def test_evidence_is_preserved_without_unsupported_qwen_options(self):
        body = {"model": "qwen", "seed": 604, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_object"}, "max_tokens": 2800,
                "messages": [{"role": "system", "content": "source policy"},
                             {"role": "user", "content": "[S001] exact source\nsecond line"}]}
        native = A.native_request(body, "claude-haiku-5-5")
        self.assertEqual(native["system"], "source policy")
        self.assertEqual(native["messages"], body["messages"][1:])
        self.assertEqual(native["max_tokens"], 2800)
        self.assertTrue({"seed", "temperature", "chat_template_kwargs", "response_format"}.isdisjoint(native))

    def test_truncation_is_not_misreported_as_successful_stop(self):
        value = {"id": "msg-test", "model": "haiku", "stop_reason": "max_tokens",
                 "content": [{"type": "text", "text": "partial"}],
                 "usage": {"input_tokens": 20, "output_tokens": 7}}
        response = A.compatible_response(value)
        self.assertEqual(response["choices"][0]["finish_reason"], "length")
        self.assertEqual(response["usage"]["total_tokens"], 27)

    def test_cache_accounting_keeps_all_processed_input_tokens(self):
        value = {"id": "msg-test", "model": "haiku", "stop_reason": "end_turn",
                 "content": [{"type": "text", "text": "answer"}],
                 "usage": {"input_tokens": 20, "output_tokens": 7,
                           "cache_creation_input_tokens": 30, "cache_read_input_tokens": 50}}
        self.assertEqual(A.compatible_response(value)["usage"]["prompt_tokens"], 100)

    def test_sonnet_uses_supported_control_without_changing_haiku(self):
        body = {"messages": [{"role": "user", "content": "question"}]}
        self.assertEqual(A.native_request(body, "claude-sonnet-5-5")["thinking"],
                         {"type": "between_tools"})
        self.assertEqual(A.native_request(body, "claude-haiku-5-5")["thinking"],
                         {"type": "disabled"})


if __name__ == "__main__":
    unittest.main()
