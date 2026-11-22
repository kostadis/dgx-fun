import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('atlaskv_prepare',Path(__file__).parents[1]/'experiments/atlaskv/prepare.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)


class FactExportTests(unittest.TestCase):
    def test_provenance_is_required(self):
        with self.assertRaises(ValueError):P.exported_assertions([],[],{})

    def test_missing_display_name_and_endpoint_stay_explicit(self):
        rows=P.exported_assertions([{'id':'A','description':'A named source entity with a description.'}],
            [{'source_node_id':'A','target_node_id':'B','relationship_name':'visits'}],
            {'chunk_id':'chunk','capture_id':'capture'})
        edge=rows[-1]
        self.assertEqual((edge['head'],edge['tail']),('A','B'))
        self.assertEqual(edge['unresolved_endpoint_labels'],['B'])
        self.assertFalse(any(r['head']=='B' and r['relation']=='description' for r in rows))

    def test_deduplication_keeps_both_sources_and_reverse_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[{'head':'Arin','relation':'visits','tail':'Haven','directions':['forward','reverse'],
                   'provenance':{'chunk_id':c,'capture_id':c}} for c in ['one','two']]
            (root/'assertions.json').write_text(json.dumps(rows))
            P.materialize(root,{'visits':{'forward':'destination','reverse':'visitor'}})
            facts=json.loads((root/'facts.json').read_text())
            self.assertEqual(len(facts),2)
            self.assertTrue(all(len(f['provenance'])==2 for f in facts))
            inverse=next(f for f in facts if f['direction']=='reverse')
            self.assertEqual((inverse['key_string'],inverse['description']),('the visitor of Haven','Arin'))

    def test_conflicting_values_are_not_silently_merged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[{'head':'Arin','relation':'visits','tail':town,'directions':['forward'],
                   'provenance':{'chunk_id':town,'capture_id':town}} for town in ['Haven','Port']]
            (root/'assertions.json').write_text(json.dumps(rows))
            P.materialize(root,{'visits':{'forward':'destination'}})
            facts=json.loads((root/'facts.json').read_text())
            self.assertEqual({f['description'] for f in facts},{'Haven','Port'})


if __name__=='__main__':unittest.main()
