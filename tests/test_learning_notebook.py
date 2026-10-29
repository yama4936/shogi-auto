import ast
import base64
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/kaggle_learning_v1.ipynb'


@unittest.skipUnless(NOTEBOOK.exists(), 'build the offline learning Notebook first')
class OfflineNotebookTests(unittest.TestCase):
    def test_output_free_offline_bundle_and_source_identity(self):
        notebook = json.loads(NOTEBOOK.read_text())
        constants = {}
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
                source = ''.join(cell['source'])
                compile(source, '<learning-notebook>', 'exec')
                for node in ast.parse(source).body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id in ('BUNDLE_B64', 'BUNDLE_SHA256'):
                                constants[target.id] = ast.literal_eval(node.value)
        raw = base64.b64decode(constants['BUNDLE_B64'])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), constants['BUNDLE_SHA256'])
        with ZipFile(BytesIO(raw)) as bundle, tempfile.TemporaryDirectory() as directory:
            names = bundle.namelist()
            self.assertIn('licenses/python-shogi-GPL-3.0.txt', names)
            self.assertFalse(any(name.endswith(('.pt', '.pth', '.npz', '.bin', '.exe')) for name in names))
            self.assertEqual(bundle.read('src/shogi_auto/engine.py'), (ROOT / 'src/shogi_auto/engine.py').read_bytes())
            bundle.extractall(directory)
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(directory) / 'src') + os.pathsep + str(Path(directory) / 'vendor')
            result = subprocess.run([sys.executable, '-c',
                                     'import json,shogi; from shogi_auto.learning.common import source_identity; '
                                     'print(json.dumps({"source":source_identity(),"shogi":shogi.__file__}))'],
                                    cwd=directory, env=env, capture_output=True, text=True, check=True)
            observed = json.loads(result.stdout)
            self.assertTrue(observed['source']['offline_bundle'])
            self.assertEqual(observed['source']['commit'], json.loads(bundle.read('SOURCE.json'))['commit'])
            self.assertTrue(observed['shogi'].startswith(directory))


if __name__ == '__main__':
    unittest.main()
