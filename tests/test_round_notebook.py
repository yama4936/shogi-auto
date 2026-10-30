import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/kaggle_policy_round3.ipynb'


@unittest.skipUnless(NOTEBOOK.exists(),'Round Notebook is built only after pinning code and data')
class RoundNotebookTests(unittest.TestCase):
    def test_private_warm_start_input_bounded_training_and_pinned_source(self):
        n=json.loads(NOTEBOOK.read_text());self.assertEqual(n['metadata']['round_id'],3)
        code=''.join(n['cells'][1]['source']);statements=ast.parse(code).body
        values={s.targets[0].id:ast.literal_eval(s.value) for s in statements[:5]}
        self.assertIsNone(values['DATA_BASE64'])
        raw=base64.b64decode(values['SOURCE_BASE64'],validate=True)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),values['SOURCE_ZIP_SHA256'])
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for name in z.namelist():
                self.assertFalse(name.endswith(('.bin','.pt','.pth','.npz','.exe')))
                if name.startswith('src/') and name.endswith('.py'):
                    expected=subprocess.check_output(['git','-C',str(ROOT),'show',f'{values["PINNED_CODE_COMMIT"]}:{name}'])
                    self.assertEqual(z.read(name),expected)
        for fragment in ['--initial-model','--defer-test','--patience',"'1200'",'1500-(time.monotonic()-started)',
                         'policy-round3-inputs.zip','policy-round3-return.txt']:
            self.assertIn(fragment,code)
        self.assertNotIn('pip install',code)
        self.assertNotIn('policy.arena',code)
        self.assertFalse(n['cells'][1]['outputs'])


if __name__=='__main__':unittest.main()
