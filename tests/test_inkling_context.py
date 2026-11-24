"""Context expansion must keep its rollback and reject invalid probe success."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('inkling_context',
    Path(__file__).resolve().parents[1] / 'experiments/inkling/context/run.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class ContextControls(unittest.TestCase):
    def test_reasoning_or_truncation_is_not_a_successful_no_thinking_probe(self):
        for finish, reasoning in [('length', ''), ('stop', 'A generated thought')]:
            response = {'choices': [{'finish_reason': finish,
                'message': {'content': '323', 'reasoning_content': reasoning}}],
                'usage': {'prompt_tokens': 20, 'completion_tokens': 5}}
            with mock.patch.object(C, 'request', return_value=response):
                with self.assertRaises(AssertionError):
                    C.complete([{'role': 'user', 'content': 'arithmetic'}], 'test')

    def test_rollback_after_promotion_frees_the_name_and_restores_only_inkling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'before.json').write_text(json.dumps({'native': {'id': 'old-id'}, 'clef': {}}))
            (root / 'fp8-candidate.json').write_text(json.dumps({'id': 'candidate-id'}))
            def inspect(identity):
                if identity == 'candidate-id':
                    return {'id': identity, 'running': True, 'name': '/inkling2mini'}
                return {'id': 'old-id', 'running': False, 'name': '/inkling2mini-64k-20261122'}
            occupied = subprocess.CompletedProcess([], 0, json.dumps([{'Id': 'candidate-id'}]))
            with mock.patch.object(C, 'inspect', side_effect=inspect), \
                 mock.patch.object(C, 'stop_guard'), mock.patch.object(C, 'run') as run, \
                 mock.patch.object(C.subprocess, 'run', return_value=occupied), \
                 mock.patch.object(C, 'wait_ready', return_value={}), \
                 mock.patch.object(C, 'clef_unchanged'):
                C.restore(root)
            commands = [call.args for call in run.call_args_list]
            self.assertIn(('docker', 'stop', '--timeout', '30', 'candidate-id'), commands)
            self.assertIn(('docker', 'rename', 'candidate-id', 'inkling-context-retained-candidate-id'), commands)
            self.assertIn(('docker', 'rename', 'old-id', 'inkling2mini'), commands)
            self.assertIn(('docker', 'start', 'old-id'), commands)
            self.assertFalse(any('clef-card-train' in command for command in commands))
            self.assertTrue(json.loads((root / 'restoration.json').read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
