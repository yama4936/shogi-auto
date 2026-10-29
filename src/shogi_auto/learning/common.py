"""Experiment provenance, safe JSON artifacts and a cumulative wall-time budget."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def source_identity():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    files = sorted(Path(__file__).parent.glob('*.py')) + [root / 'src' / 'shogi_auto' / 'engine.py']
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True)
    bundle_path = root / 'SOURCE.json'
    bundle = json.loads(bundle_path.read_text()) if bundle_path.exists() else {}
    if result.returncode != 0 and bundle and bundle.get('source_sha256') != digest.hexdigest():
        raise ValueError('Offline bundle source checksum changed')
    return {'commit': result.stdout.strip() if result.returncode == 0 else bundle.get('commit', 'unavailable'),
            'source_sha256': digest.hexdigest(), 'dirty': bool(dirty.stdout.strip()),
            'offline_bundle': result.returncode != 0 and bool(bundle)}


class BudgetExpired(RuntimeError):
    pass


class Budget:
    """Persist consumed wall time. Resuming never resets the run's time allowance.

    A hard process watchdog in the Notebook additionally bounds the whole launch.
    Checks are cooperative, so reserve time for a final checkpoint and shutdown.
    """
    def __init__(self, directory, seconds=1500, visible_gpus=0):
        self.path = Path(directory) / 'budget.json'
        previous = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.prior = float(previous.get('used_wall_seconds', 0))
        self.seconds = min(float(seconds), float(previous.get('limit_wall_seconds', seconds)))
        self.visible_gpus = visible_gpus
        self.started = time.monotonic()

    def used(self):
        return self.prior + time.monotonic() - self.started

    def remaining(self):
        return self.seconds - self.used()

    def check(self, reserve=10):
        if self.remaining() <= reserve:
            raise BudgetExpired('Wall-time budget reached; saved files can be resumed with the remaining allowance.')

    def save(self):
        write_json(self.path, {'used_wall_seconds': self.used(), 'limit_wall_seconds': self.seconds,
                              'visible_gpus': self.visible_gpus,
                              'estimated_allocation_gpu_hours': self.used() * self.visible_gpus / 3600,
                              'quota_remaining': 'must be checked in Kaggle UI; not inferred from CUDA'})
