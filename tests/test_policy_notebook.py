import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/kaggle_policy_v2.ipynb'


@unittest.skipUnless(NOTEBOOK.exists(), 'policy notebook is generated after the source commit is pinned')
class PolicyNotebookTests(unittest.TestCase):
    def test_offline_bundle_matches_pinned_source_and_imports(self):
        notebook=json.loads(NOTEBOOK.read_text())
        self.assertTrue(all(not cell.get('outputs') and cell.get('execution_count') is None
                            for cell in notebook['cells'] if cell['cell_type']=='code'))
        code=''.join(next(c for c in notebook['cells'] if c['cell_type']=='code')['source'])
        values={}
        for statement in ast.parse(code).body[:4]:
            self.assertIsInstance(statement,ast.Assign)
            values[statement.targets[0].id]=ast.literal_eval(statement.value)
        raw=base64.b64decode(values['SOURCE_BASE64'],validate=True)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),values['SOURCE_ZIP_SHA256'])
        self.assertEqual(values['DATA_ZIP_SHA256'],notebook['metadata']['private_data_zip_sha256'])
        with zipfile.ZipFile(io.BytesIO(raw)) as bundle,tempfile.TemporaryDirectory() as directory:
            origin=json.loads(bundle.read('SOURCE_POLICY.json'))
            self.assertEqual(origin['commit'],values['PINNED_CODE_COMMIT'])
            for name in bundle.namelist():
                self.assertFalse(name.endswith(('.bin','.pt','.pth','.npz','.exe','.so')))
                if name.startswith('src/') and name.endswith('.py'):
                    expected=subprocess.check_output(['git','-C',str(ROOT),'show',f'{origin["commit"]}:{name}'])
                    self.assertEqual(bundle.read(name),expected)
            self.assertIn('licenses/python-shogi-GPL-3.0.txt',bundle.namelist())
            bundle.extractall(directory)
            env=os.environ.copy();env['PYTHONPATH']=str(Path(directory)/'src')+os.pathsep+str(Path(directory)/'vendor')
            result=subprocess.run([sys.executable,'-c','from shogi_auto.policy.common import source_identity; import json;print(json.dumps(source_identity()))'],
                                  env=env,cwd=directory,capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(result.stdout)['source_sha256'],origin['source_sha256'])
        self.assertIn("'--max-wall-seconds','3300'",code)
        self.assertIn('3500-(time.monotonic()-started)',code)
        self.assertNotIn('pip install',code)
        self.assertNotIn('policy.arena',code)


if __name__=='__main__':unittest.main()
