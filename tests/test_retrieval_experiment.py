import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parents[1] / 'clef/eval'
with patch.object(sys, 'path', [str(HERE)] + sys.path):
    spec = importlib.util.spec_from_file_location('retrieval_experiment', HERE / 'retrieval_experiment.py')
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)


class RetrievalTest(unittest.TestCase):
    def test_alignment_localizes_changed_name(self):
        self.assertEqual(R.aligned_queries('Misrim', 'The bard greets Misrim at the gate.',
                                         'The bard greets Mizzrym at the gate.'), ['Mizzrym'])
        self.assertEqual(R.aligned_queries('A B', 'We met A B yesterday.', 'We met C D yesterday.'), ['C D'])
        self.assertEqual(R.aligned_queries('absent', 'The bard sings.', 'The bard dances.'), [])

    def test_sibling_and_learned_alias_retrieve_name(self):
        pool = {'Xeno': '', 'Buppido': '', 'Random': ''}
        retriever = R.Retriever(pool, lambda s: set())
        self.assertEqual(retriever.rank('Peto', queries=['Buppido'])[0], 'Buppido')
        self.assertEqual(retriever.rank('Peto', history=[('Peto', 'Buppido')])[0], 'Buppido')

    def test_OTHER_does_not_count_as_recovered_spelling(self):
        row = {'session': 's', 'id': 1, 'gold': 'Buppido', 'gold_in_opts': False}
        result = {'session': 's', 'id': 1, 'ans': {'choice': 'OTHER', 'confidence': .99}}
        score = R.summarize([row], [result])
        self.assertEqual(score['correct_decisions'], 1)
        self.assertEqual(score['correct_spellings'], 0)

    def test_target_and_future_rulings_do_not_affect_target_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'data'; data.mkdir()
            (data / 'results/e2').mkdir(parents=True)
            (data / 'results/e2/gemini.json').write_text('[]')
            phonetics = root / 'phonetics.py'; phonetics.write_text('def codes(s): return set()\n')
            pool = {f'Name{i}': 'name' for i in range(40)}
            pool.update({'Alpha': 'name', 'Beta': 'name', 'Gamma': 'name'})
            def row(date, key, gold):
                return {'session': date + '-session', 'id': key, 'token': 'zzzzz',
                        'gold': gold, 'gold_in_opts': False, 'options': {'Name0': 'name'},
                        'state': 'Raw transcript around its first occurrence:\nzzzzz walks.' + R.SIBLING + '(none)'}
            rows = [row('2025-01-01', 1, 'Alpha'), row('2025-01-02', 2, 'Beta'),
                    row('2025-01-02', 3, 'Gamma'), row('2025-01-03', 4, 'Gamma')]
            def prepare(name):
                (data / 'e2_pick.json').write_text(json.dumps(rows))
                (data / 'MANIFEST.json').write_text(json.dumps({'campaign_commit': 'test', 'files': {
                    'e2_pick': {'sha256': hashlib.sha256((data / 'e2_pick.json').read_bytes()).hexdigest()}}}))
                args = argparse.Namespace(data=data, workdir=root/name, campaign=root, phonetics=phonetics)
                with patch.object(R, 'snapshot_pool', return_value=pool.copy()), redirect_stdout(io.StringIO()):
                    R.prepare(args)
                return json.loads((args.workdir / 'history32.json').read_text())[1]['options']
            first = prepare('first')
            rows[1]['gold'] = 'Gamma'  # target label
            rows[2]['gold'] = 'Beta'   # another ruling on the same date
            rows[3]['gold'] = 'Beta'   # future ruling
            second = prepare('second')
            self.assertEqual(first, second)
            self.assertEqual(next(iter(first)), 'Alpha')  # strictly earlier evidence is usable


if __name__ == '__main__':
    unittest.main()
