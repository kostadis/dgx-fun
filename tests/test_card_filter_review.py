import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'card_filter_review', Path(__file__).resolve().parents[1] / 'clef/eval/card_filter_review.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


class CardFilterTest(unittest.TestCase):
    def test_replay_routes_only_scored_concrete_corrections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'results/e1_rawC').mkdir(parents=True)
            prose, raw, scored = [], [], []
            for session, fix, p in [('a', 'Buppido', .91), ('b', '(none)', .99), ('c', 'Buppido', .7)]:
                key = {'session': session, 'id': 'same-id'}
                prose.append({**key, 'state': 'Card: Fix a name\n\nProposal: Use Buppido.\n\nEvidence: Original tape.'})
                raw.append({**key, 'state': f'Transcript token: Bupito\nProposed replacement: {fix}\nRaw tape'})
                scored.append({**key, 'verdict': 'reject', 'note': 'SECRET_RULING',
                               'answers': {'accept': {'noul': p}}})
            for name, rows in [('e1_cards_prose.json', prose), ('e1_cards_rawC.json', raw),
                               ('results/e1_rawC/gemini.json', scored)]:
                (root / name).write_text(json.dumps(rows))
            review, counts = R.replay(root)
            self.assertEqual([it['likely_yes'] for it in review['items']], [True, False, False])
            self.assertEqual(len({it['id'] for it in review['items']}), 3)
            self.assertEqual(counts['historical_nonapprovals_in_flat_list'], 1)
            self.assertNotIn('SECRET_RULING', json.dumps(review))
            self.assertEqual(review['items'][0]['proposed_text'], 'Use Buppido.')
            self.assertNotIn('decisions', review)
            self.assertNotIn('state', R.shared_spec(review))

    def test_embedded_transcript_cannot_close_json_script(self):
        review = {'title': 'Replay', 'reviewId': 'test', 'items': [
            {'id': 'x', 'title': '</script><img src=x onerror=alert(1)>', 'evidence': 'x'}]}
        page = R.render(review)
        self.assertNotIn('</script><img', page)
        self.assertIn('\\u003c/script\\u003e', page)

    def test_no_change_is_not_bulk_eligible(self):
        self.assertFalse(R.concrete_proposal('word', 'WORD'))
        self.assertFalse(R.concrete_proposal('word', 'none'))
        self.assertFalse(R.concrete_proposal('', 'name'))
        self.assertTrue(R.concrete_proposal('Bupito', 'Buppido'))

    def test_three_tier_boundaries_and_unscored_cards(self):
        for p, expected in ((.90001, 'high'), (.9, 'middle'), (.8, 'middle'), (.7999, 'cards'),
                            (None, 'cards'), (float('nan'), 'cards'), (True, 'cards'), (1.1, 'cards')):
            self.assertEqual(R.tier_for(True, p), expected)
        self.assertEqual(R.tier_for(False, .99), 'cards')


if __name__ == '__main__':
    unittest.main()
