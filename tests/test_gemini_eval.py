"""The hosted adapter must preserve frozen questions and provider probabilities."""
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'decision_eval_tasks', Path(__file__).resolve().parents[1] / 'clef/eval/tasks.py')
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


class GeminiEvalTest(unittest.TestCase):
    def test_native_request_and_usage(self):
        answers = {'kind': {'type': 'choice', 'choice': 'campaign',
                           'probabilities': {'campaign': 0.7, 'filler': 0.1,
                                             'rules': 0.1, 'realworld': 0.1},
                           'confidence': 0.7}}
        response = {'answers': answers, 'usage': {'total_input_tokens': 123,
                                                 'total_output_tokens': 0}}
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}), patch.object(
                T.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as send:
            got, mid, usage = T.systemone('gemini', 'frozen state', T.Q_TRIAGE)
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, T.SYSTEMONE['gemini'][0])
        self.assertEqual(request.get_header('X-goog-api-key'), 'test-key')
        self.assertEqual(send.call_args.kwargs['timeout'], 60)
        self.assertEqual(json.loads(request.data), {'model': 'gemini-decision-1',
                         'input': 'frozen state', 'questions': T.Q_TRIAGE})
        self.assertEqual(got, answers)
        self.assertEqual(mid, 'gemini-decision-1')
        self.assertEqual(usage['input_tokens'], 123)
        self.assertEqual(usage['total_output_tokens'], 0)

    def test_local_protocol_unchanged(self):
        with patch.object(T, 'post', return_value={'answers': {}, 'model': 'nox',
                                                 'usage': {'input_tokens': 10}}) as send:
            T.systemone('nox', 'frozen state', T.Q_E1_PROSE)
        self.assertEqual(send.call_args.args[1], {'model': T.SYSTEMONE['nox'][1],
                         'state': 'frozen state', 'questions': T.Q_E1_PROSE})

    def test_key_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            keyfile = Path(tmp) / 'key'
            for content in ('file-key', 'GEMINI_API_KEY="file-key"'):
                keyfile.write_text(content)
                with patch.dict(os.environ, {'GEMINI_API_KEY_FILE': str(keyfile)}, clear=True):
                    self.assertEqual(T.gemini_key(), 'file-key')
            with patch.dict(os.environ, {'GEMINI_API_KEY': 'first', 'GOOGLE_API_KEY': 'second',
                                        'GEMINI_API_KEY_FILE': str(keyfile)}, clear=True):
                self.assertEqual(T.gemini_key(), 'first')
            with patch.dict(os.environ, {'GOOGLE_API_KEY': 'second'}, clear=True):
                self.assertEqual(T.gemini_key(), 'second')

    def test_missing_key_fails_before_network(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY_FILE': '/nonexistent/gemini-key'}, clear=True), \
                patch.object(T.urllib.request, 'urlopen') as send:
            with self.assertRaisesRegex(RuntimeError, 'GEMINI_API_KEY'):
                T.systemone('gemini', 'state', T.Q_E1_RAW)
            send.assert_not_called()

    def test_runner_all_arms_with_synthetic_data(self):
        sources = ('e1_cards_prose', 'e1_cards_rawB', 'e1_cards_rawC', 'e2_pick',
                   'e3_triage', 'e4_ent', 'e4_facts', 'e4_ctrl', 'e5_npc')
        row = {'state': 'synthetic state', 'id': 'synthetic', 'session': 'test',
               'options': {'Mug': 'A mug'}, 'gold': 'Mug', 'gold_in_opts': True,
               'a': {'state': 'synthetic a'}, 'b': {'state': 'synthetic b'}}

        def response(url, body):
            self.assertIn('input', body)
            self.assertNotIn('state', body)
            answers = {}
            for name, q in body['questions'].items():
                if q['type'] == 'noul':
                    answers[name] = {'type': 'noul', 'noul': 0.8, 'confidence': 0.8}
                else:
                    keys = list(q['criteria'])
                    answers[name] = {'type': 'choice', 'choice': keys[0], 'confidence': 0.8,
                                     'probabilities': {k: 0.8 if i == 0 else 0.2 / (len(keys) - 1)
                                                       for i, k in enumerate(keys)}}
            return {'answers': answers, 'usage': {'total_input_tokens': 42, 'total_output_tokens': 0}}

        with tempfile.TemporaryDirectory() as tmp:
            for src in sources:
                (Path(tmp) / f'{src}.json').write_text(json.dumps([row]))
            with patch.dict(os.environ, {'DATA': tmp}), patch.dict(sys.modules, {'tasks': T}), \
                    patch.object(sys, 'argv', ['run.py', 'gemini', 'e1', 'e2', 'e3', 'e4', 'e5']), \
                    patch.object(sys, 'path', list(sys.path)), patch.object(T, 'post', side_effect=response) as send, \
                    redirect_stdout(io.StringIO()):
                runpy.run_path(str(Path(T.__file__).with_name('run.py')), run_name='__main__')
            results = list((Path(tmp) / 'results').glob('*/gemini.json'))
            self.assertEqual(len(results), 9)
            self.assertEqual(send.call_count, 11)  # preflight + 8 single rows + 2 NPC sides
            for result in results:
                saved = json.loads(result.read_text())[0]
                self.assertNotIn('error', saved)
                values = [saved['a'], saved['b']] if result.parent.name == 'e5' else [saved]
                for value in values:
                    self.assertEqual(value['served'], 'gemini-decision-1')
                    self.assertEqual(value['input_tokens'], 42)
                    self.assertEqual(value['usage']['total_output_tokens'], 0)

    def test_checkpoint_recovers_interrupted_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{'state': f'synthetic {i}', 'id': i, 'options': {'Mug': 'A mug'},
                     'gold': 'Mug', 'gold_in_opts': True} for i in range(2)]
            (Path(tmp) / 'e2_pick.json').write_text(json.dumps(rows))
            answer = ({'pick': {'choice': 'Mug', 'confidence': 0.9,
                               'probabilities': {'Mug': 0.9, 'LEAVE': 0.05, 'OTHER': 0.05}}},
                      'gemini-decision-1', {'input_tokens': 42})
            script = str(Path(T.__file__).with_name('run.py'))
            with patch.dict(os.environ, {'DATA': tmp}), patch.dict(sys.modules, {'tasks': T}), \
                    patch.object(sys, 'argv', ['run.py', 'gemini', 'e2']), \
                    patch.object(sys, 'path', list(sys.path)), redirect_stdout(io.StringIO()):
                with patch.object(T, 'systemone', side_effect=[answer, answer, KeyboardInterrupt()]):
                    with self.assertRaises(KeyboardInterrupt):
                        runpy.run_path(script, run_name='__main__')
                partial = Path(tmp) / 'results/e2/gemini.partial.jsonl'
                self.assertEqual(len(partial.read_text().splitlines()), 1)
                with partial.open('a') as out:
                    out.write('{interrupted')
                with patch.object(T, 'systemone', return_value=answer) as send:
                    runpy.run_path(script, run_name='__main__')
                    self.assertEqual(send.call_count, 2)  # preflight + only the unfinished row
                saved = json.loads(partial.with_name('gemini.json').read_text())
                self.assertEqual([r['id'] for r in saved], [0, 1])
                self.assertFalse(partial.exists())


if __name__ == '__main__':
    unittest.main()
