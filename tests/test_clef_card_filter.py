import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]/'clef/eval'
with patch.object(sys,'path',[str(HERE)]+sys.path):
    spec=importlib.util.spec_from_file_location('clef_card_filter',HERE/'clef_card_filter.py')
    C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


class ClefCardFilterTest(unittest.TestCase):
    def test_open_questions_never_enter_bulk_groups(self):
        rows=[{'proposal':True,'p':.95,'verdict':'approve'},
              {'proposal':True,'p':.85,'verdict':'reject'},
              {'proposal':True,'p':.5,'verdict':'discuss'},
              {'proposal':False,'p':.99,'verdict':'discuss'}]
        result=C.tier_summary(rows)
        self.assertEqual(result['high']['cards'],1)
        self.assertEqual(result['middle']['proposal_nonapprovals'],1)
        self.assertEqual(result['cards']['cards'],2)
        self.assertEqual(result['cards']['proposal_nonapprovals'],1)
        self.assertEqual(sum(r['cards'] for r in result.values()),4)


if __name__=='__main__':unittest.main()
