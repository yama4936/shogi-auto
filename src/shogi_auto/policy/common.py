"""Policy experiment identity includes every local source dependency."""
import hashlib
import json
from pathlib import Path
import subprocess

from ..learning.common import Budget, BudgetExpired, sha256, write_json


def source_files(root):
    root = Path(root)
    return (sorted((root / 'src/shogi_auto/policy').glob('*.py')) +
            [root / 'src/shogi_auto' / name for name in ('__init__.py', 'engine.py', 'game.py')] +
            [root / 'src/shogi_auto/learning' / name for name in ('__init__.py', 'common.py', 'data.py', 'features.py', 'arena.py')])


def source_identity():
    root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for path in source_files(root):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    bundle = root / 'SOURCE_POLICY.json'
    if result.returncode:
        origin = json.loads(bundle.read_text())
        if origin['source_sha256'] != digest.hexdigest():
            raise ValueError('Policy source bundle changed')
        return origin
    dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True)
    return {'commit': result.stdout.strip(), 'source_sha256': digest.hexdigest(),
            'dirty': bool(dirty.stdout.strip()), 'offline_bundle': False}
