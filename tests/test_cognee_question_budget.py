"""A bounded replay must retain generation failures without hiding HTTP errors."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'experiments/adaptive-chunking/cognee/questions.py'
spec = importlib.util.spec_from_file_location('question_budget', path)
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)


class QuestionBudget(unittest.TestCase):
    def test_default_preserves_existing_request_and_cap_is_explicit(self):
        self.assertEqual(Q.reader_extra_body('none'),
                         {'chat_template_kwargs': {'enable_thinking': False}, 'reasoning_effort': 'none'})
        self.assertEqual(Q.reader_extra_body('none', 4096)['max_tokens'], 4096)
        with self.assertRaises(ValueError):
            Q.reader_extra_body('none', 0)

    def test_nested_effort_and_channel_suppression_reach_reader_without_top_level_none(self):
        controls = {'chat_template_kwargs': {'reasoning_effort': 0.0}, 'logit_bias': {'200008': False}}
        body = Q.reader_extra_body(None, 4096, controls)
        self.assertNotIn('reasoning_effort', body)
        self.assertEqual(body['chat_template_kwargs'], {'enable_thinking': False, 'reasoning_effort': 0.0})
        self.assertIs(body['logit_bias']['200008'], False)
        with self.assertRaises(ValueError):
            Q.reader_extra_body('none', 4096, controls)
        with self.assertRaises(ValueError):
            Q.reader_extra_body(None, 4096, {'messages': []})

    def test_capped_reasoning_only_response_is_a_failure_even_when_http_succeeds(self):
        calls = [{'status': 200, 'finish_reasons': ['length']}]
        self.assertFalse(Q.validate_reader_call(calls, [], continue_on_truncation=True))
        with self.assertRaises(RuntimeError):
            Q.validate_reader_call(calls, [])

    def test_native_hard_thinking_ban_preserves_original_effort_setting(self):
        body = Q.reader_extra_body('none', 4096, {'bad_words': ['<|content_thinking|>'], 'cache_salt': 'independent-arm'})
        self.assertEqual(body['reasoning_effort'], 'none')
        self.assertEqual(body['bad_words'], ['<|content_thinking|>'])
        self.assertEqual(body['cache_salt'], 'independent-arm')

    def test_continuation_never_hides_http_errors_or_missing_successful_answer(self):
        for calls, answers in (([{'status': 500, 'finish_reasons': ['length']}], []),
                               ([{'status': 200, 'finish_reasons': ['stop']}], []),
                               ([], [])):
            with self.assertRaises(RuntimeError):
                Q.validate_reader_call(calls, answers, continue_on_truncation=True)
        self.assertTrue(Q.validate_reader_call([{'status': 200, 'finish_reasons': ['stop']}], ['answer']))


if __name__ == '__main__':
    unittest.main()
