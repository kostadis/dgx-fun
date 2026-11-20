import importlib.util
import argparse
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / 'clef/eval'
with patch.object(sys, 'path', [str(HERE)] + sys.path):
    spec = importlib.util.spec_from_file_location('card_input_ablation', HERE / 'card_input_ablation.py')
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)


class CardInputAblationTest(unittest.TestCase):
    def test_local_model_selection_does_not_call_gemini(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=argparse.Namespace(model='clef',workdir=root,repeats=1,report=root/'report.json')
            prompts={'model':'gemini-decision-1','states':{'original':'private state'},
                     'questions':{'accept':{'type':'noul','instructions':'Approve?'}},
                     'previous_scores':{'original':.8},'provenance':{}}
            with patch.object(R.T,'systemone',return_value=({'accept':{'noul':.7}},'clef',{'input_tokens':10})) as call:
                report=R.run(args,prompts)
            self.assertEqual(call.call_args.args[0],'clef')
            self.assertEqual(report['model'],'clef')
            self.assertEqual(report['reference_model'],'gemini-decision-1')

    def setUp(self):
        self.state = ('Transcript token: Entemok\nProposed replacement: Target\nOccurrences on tape: 1'
                      + R.SIB_START + 'query: abc\nsibling: x\nquote truncated'
                      + R.SIB_END + ' registry\nWider context stays unchanged')
        self.full = 'query: abc\nsibling: x\nquote truncated and now complete'
        self.edit = {'old': "Entemok's moon", 'new': 'Target', 'cue': 627,
                     'was': "Path to Entemok's moon.", 'now': 'Path to Target.'}

    def test_factors_are_isolated_and_original_is_exact(self):
        variants = R.variants(self.state, self.edit, self.full)
        self.assertEqual(len(variants), 9)
        self.assertEqual(variants['original'], self.state)
        self.assertEqual(variants['span'], self.state.replace('Transcript token: Entemok\n',
                                                           "Transcript token: Entemok's moon\n", 1))
        self.assertNotIn('Edit scope:', variants['sibling'])
        self.assertIn('and now complete', variants['sibling'])
        self.assertNotIn('and now complete', variants['scope'])
        self.assertIn('Apply only at cue 627', variants['scope'])
        self.assertIn('Exact sentence before this proposal:', variants['faithful_edit'])
        self.assertTrue(all(v.endswith('Wider context stays unchanged') for k,v in variants.items()
                            if k != 'faithful_edit'))

    def test_recovered_quote_must_preserve_original_prefix(self):
        with self.assertRaisesRegex(ValueError, 'extend the exact'):
            R.variants(self.state, self.edit, 'Different quote')

    def test_before_after_must_be_only_the_proposed_change(self):
        self.edit['now'] = 'Unrelated changed sentence.'
        with self.assertRaisesRegex(ValueError, 'additional changes'):
            R.variants(self.state, self.edit, self.full)


if __name__ == '__main__':
    unittest.main()
