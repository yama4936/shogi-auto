import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gpu_smoke', ROOT / 'scripts/gpu_smoke.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeTests(unittest.TestCase):
    def test_cuda_preferred(self):
        self.assertEqual(smoke.select_device(True), 'cuda')
        self.assertEqual(smoke.select_device(True, True), 'cuda')

    def test_cuda_required_by_default(self):
        with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
            smoke.select_device(False)

    def test_explicit_cpu_fallback(self):
        self.assertEqual(smoke.select_device(False, True), 'cpu')

    def test_missing_torch_has_actionable_error(self):
        with patch.dict(sys.modules, {'torch': None}):
            with self.assertRaisesRegex(RuntimeError, 'PyTorch is missing'):
                smoke.run_smoke()

    def test_notebook_is_clean_and_code_matches_script(self):
        notebook = json.loads((ROOT / 'notebooks/kaggle_gpu_smoke.ipynb').read_text())
        self.assertEqual(notebook['nbformat'], 4)
        code = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        for cell in code:
            self.assertIsNone(cell['execution_count'])
            self.assertEqual(cell['outputs'], [])
            compile(''.join(cell['source']), '<notebook>', 'exec')
        script = (ROOT / 'scripts/gpu_smoke.py').read_text().split('\nif __name__ == "__main__":')[0]
        self.assertEqual(''.join(code[0]['source']), script)
        self.assertIn('report = run_smoke()', ''.join(code[-1]['source']))


if __name__ == '__main__':
    unittest.main()
