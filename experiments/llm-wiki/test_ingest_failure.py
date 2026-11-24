"""A final service-check failure must fail the ingestion CLI and its receipt."""
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import runtime


class IngestFailure(unittest.TestCase):
    def test_final_service_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory)
            runtime.save(trial / 'frozen/corpus.json', {'documents': [
                {'id': str(i), 'title': 'Synthetic', 'text': 'Synthetic source.'}
                for i in range(13)
            ]})
            runtime.save(trial / 'frozen/cases.json', [])
            notes = [SimpleNamespace(rel=f'n{i}.md', id=str(i), frontmatter={},
                full_text=lambda: 'Synthetic note.', main_body=lambda: 'Synthetic note.')
                for i in range(13)]
            report = SimpleNamespace(stopped=None, not_done=[], results=[(None, None)] * 13,
                                     to_dict=lambda: {'completed': 13})
            wiki = SimpleNamespace(cfg=SimpleNamespace(model_dump=lambda **kwargs: {}),
                init=lambda: None, ingest_many=lambda *args, **kwargs: report,
                store=SimpleNamespace(load=lambda: SimpleNamespace(all_notes=lambda: notes)))

            class FakeCapture:
                def __init__(self, root):
                    self.root = root

                def wiki(self, path):
                    return nullcontext(wiki)

                def check(self, name):
                    raise RuntimeError('Final service guard verification failed')

                def close(self):
                    pass

            fake = SimpleNamespace(Capture=FakeCapture, REPO=runtime.REPO, TRIAL=trial,
                SOURCE=trial / 'source', PIN=runtime.PIN, load=runtime.load,
                save=runtime.save, sha=runtime.sha,
                SERVICES=SimpleNamespace(services=lambda: {'load': {'server_load': 0}},
                    check_services=lambda *args: None))
            output = trial / 'result'
            with mock.patch.dict(sys.modules, {'runtime': fake}), \
                 mock.patch.object(sys, 'argv', ['ingest.py', '--output', str(output)]), \
                 mock.patch('subprocess.check_output', side_effect=lambda args, **kwargs:
                            runtime.PIN if args[-1] == 'HEAD' else ''), \
                 redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                runpy.run_path(str(Path(__file__).with_name('ingest.py')), run_name='__main__')
            self.assertEqual(stopped.exception.code, 1)
            state = json.loads((output / 'ingest-progress.json').read_text())
            self.assertFalse(state['passed'])
            self.assertEqual(state['stage'], 'failed')
            self.assertIn('Final service guard verification failed', state['error'])


if __name__ == '__main__':
    unittest.main()
