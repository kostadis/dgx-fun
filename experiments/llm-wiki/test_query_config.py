"""The benchmark must read its declared eight-note budget with a custom classifier."""
import ast
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest

import httpx
from llmw2.agents.steps import CROW
from runtime import Capture

HERE=Path(__file__).resolve().parent


class NoteBudget(unittest.TestCase):
    def test_custom_provider_does_not_silently_reduce_reader_to_five_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            capture=Capture.__new__(Capture)
            capture.root=root;capture.phase='test'
            capture.client=httpx.Client(transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(AssertionError('Unexpected model request'))))
            capture.embed=lambda texts: [[1.0]+[0.0]*1023 for _ in texts]
            with capture.wiki(root/'wiki') as seed:
                seed.init();seed.create_folder('/','topic','Synthetic artifact notes')
            for i in range(8):
                (root/'wiki/topic'/f'n{i}.md').write_text(
                    f'---\ntype: Note\ntitle: Artifact {i}\n---\nArtifact {i} has a recorded property.\n')
            tree=ast.parse((HERE/'query.py').read_text())
            definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fresh')
            scope={'root':root,'capture':capture,'shutil':shutil,'replace':replace,'CROW':CROW,
                   'answer_step':lambda ctx,question,notes:str(len(notes))}
            exec(compile(ast.Module(body=[definition],type_ignores=[]),'query.py','exec'),scope)
            with scope['fresh']('fixture',rerank=False) as wiki:
                wiki.cfg.answer_conflicts=False
                answer=wiki.ask('artifact')
                self.assertEqual(answer.text,'8')
                self.assertEqual(len(answer.notes),8)
            capture.client.close()


if __name__=='__main__':unittest.main()
