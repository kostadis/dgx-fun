import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('clef_readout',
    Path(__file__).resolve().parents[1]/'clef/finetune/clef_readout.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


class ClefReadoutTest(unittest.TestCase):
    def test_folds_hold_out_entire_sessions_once(self):
        rows=[{'session':str(i//3)} for i in range(27)]
        seen=[]
        for train,test in R.folds(rows):
            self.assertFalse({rows[i]['session'] for i in train}&{rows[i]['session'] for i in test})
            seen.extend(test)
        self.assertEqual(sorted(seen),list(range(27)))

    def test_metrics_keep_open_questions_on_cards(self):
        rows=[{'concrete_proposal':True,'target':{'approve_as_presented':True}},
              {'concrete_proposal':True,'target':{'approve_as_presented':False}},
              {'concrete_proposal':False,'target':{'approve_as_presented':False}}]
        result=R.evaluate(rows,[.95,.85,.999])
        self.assertEqual(result['groups']['over90']['cards'],1)
        self.assertEqual(result['groups']['80to90']['nonapprovals'],1)
        self.assertEqual(result['groups']['regular']['nonapprovals'],1)
        self.assertEqual(result['concrete_auc'],1)
        self.assertEqual(result['concrete_nonapprovals'],1)

    def test_auc_handles_ties_and_missing_class(self):
        self.assertEqual(R.auc([True,False],[.5,.5]),.5)
        self.assertIsNone(R.auc([True],[.5]))


if __name__=='__main__':unittest.main()
