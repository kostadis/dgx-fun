import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('expand_card_corpus',
    Path(__file__).resolve().parents[1] / 'clef/eval/expand_card_corpus.py')
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)


class ExpandedCorpusTest(unittest.TestCase):
    def test_proposal_mapping_supports_cue_lists_and_open_questions(self):
        self.assertTrue(R.concrete([{'cue': 1, 'was': 'Bupito', 'now': 'Buppido'}]))
        self.assertTrue(R.concrete({'token': ['Bupito'], 'canonical': ['Buppido']}))
        self.assertFalse(R.concrete({'token': 'word', 'canonical': None}))
        self.assertFalse(R.concrete({'token': 'word', 'canonical': 'WORD'}))

    def test_outcomes_are_not_structured_features(self):
        item = {'token': 'Bupito', 'canonical': 'Buppido', 'context': 'A voice on tape.',
                'decision': 'reject', 'note': 'SECRET_LABEL_NOTE', 'notes': {'x': 'reject'},
                'savedAt': 'yesterday', 'recommended_decision': 'reject'}
        actual = R.features(item)
        self.assertNotIn('SECRET_LABEL_NOTE', actual['card_text'])
        self.assertNotIn('reject', actual['card_text'])
        self.assertIn('Buppido', actual['card_text'])

    def test_original_card_text_retains_approve_and_reject_meanings(self):
        actual = R.features({'t': '<b>Correction?</b>', 'y': 'Use &lt;name&gt;',
                             'n': 'Keep tape', 'ev': 'Source line', 'note': 'SECRET'})
        self.assertIn('Approve: Use <name>', actual['card_text'])
        self.assertIn('Reject: Keep tape', actual['card_text'])
        self.assertNotIn('SECRET', actual['card_text'])


if __name__ == '__main__':
    unittest.main()
