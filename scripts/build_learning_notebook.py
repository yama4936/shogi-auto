"""Build an offline Kaggle Notebook from reviewed source (no data/weights).

PyTorch/NumPy are supplied by Kaggle and are never upgraded. The existing
python-shogi 1.1.1 source is included with its copyright headers and GPL text.
"""
import base64
from io import BytesIO
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shogi_auto.learning.common import source_identity


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def main():
    import hashlib
    import shogi
    if importlib.metadata.version('python-shogi') != '1.1.1':
        raise RuntimeError('Bundle only the reviewed python-shogi 1.1.1')
    identity = source_identity()
    entries = {}
    for path in sorted((ROOT / 'src' / 'shogi_auto' / 'learning').glob('*.py')):
        entries[str(path.relative_to(ROOT))] = path.read_bytes()
    for name in ('__init__.py', 'engine.py', 'game.py'):
        path = ROOT / 'src' / 'shogi_auto' / name
        entries[str(path.relative_to(ROOT))] = path.read_bytes()
    vendor_root = Path(shogi.__file__).parent
    for path in sorted(vendor_root.rglob('*.py')):
        entries['vendor/shogi/' + str(path.relative_to(vendor_root))] = path.read_bytes()
    entries['licenses/python-shogi-GPL-3.0.txt'] = (ROOT / 'docs/licenses/python-shogi-GPL-3.0.txt').read_bytes()
    entries['SOURCE.json'] = json.dumps(identity, sort_keys=True).encode()
    output = BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(entries.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    payload = base64.b64encode(output.getvalue()).decode()
    checksum = hashlib.sha256(output.getvalue()).hexdigest()
    introduction = f'''# shogi-auto: 学習と対局評価（初回実験）

このNotebookは **GPU動作確認だけではなく、モデルの新規学習と学習前後の対局評価** を行います。
小さな駒位置評価モデルの教師あり学習であり、強化学習・dlshogi本体・大会参加可能な完成エンジンではありません。

- 元コードcommit: `{identity['commit']}` / source SHA-256: `{identity['source_sha256']}`。
- 自前生成局面と固定CPU探索だけを使用。第三者の棋譜・データ・重みは含みません。
- python-shogi 1.1.1 (GPL-3.0-only) の公開ソースとライセンスを同梱。PyTorch/NumPyはKaggle既存版を使います。
- **非公開、Internet OFF**。追加APIキー・パッケージインストール・課金は不要です。
- 実行直前にGPU無料残量をUIで確認し、次セルに数値を入力してください。未確認のまま起動しません。
- 初回は全体45分以内を目安。Notebook内の監視は全体50分で打ち切り、T4 x2なら最大約1.67割当GPU時間。終了後ただちにセッションをOFFにしてください。
- 途中経過は `/kaggle/working/shogi-learning-v1/`。中断・予算到達時は未完了です。GPUが有効でも強化成功とは判定しません。
- 32開始局面×先後2局×比較相手2種類＝128局を予定。時間不足時は完了局を保存し、未完了と報告します。
- ルールは実験用です（最大手数で引き分け、入玉宣言・CSA時計は未対応）。棋力の段級位やEloは推定しません。
'''
    setup = '''# 親側の担当者がKaggle UIから確認して入力。APIから残量を推測しません。
REMAINING_FREE_GPU_HOURS = None
assert isinstance(REMAINING_FREE_GPU_HOURS, (int, float)) and REMAINING_FREE_GPU_HOURS >= 2, "UIで無料GPU残量を確認し、2時間以上の場合だけ値を入力してください"

import time
NOTEBOOK_STARTED = time.monotonic()
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import numpy as np
assert torch.cuda.is_available(), "GPU必須です。CPUへの自動切替は行いません"
VISIBLE_GPUS = torch.cuda.device_count()
assert 1 <= VISIBLE_GPUS <= 2
print({"torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda,
       "gpu_names": [torch.cuda.get_device_name(i) for i in range(VISIBLE_GPUS)],
       "quota_checked_in_ui": REMAINING_FREE_GPU_HOURS})
'''
    unpack = f'''import base64, hashlib, io, json, os, sys, zipfile
from pathlib import Path
BUNDLE_B64 = {payload!r}
BUNDLE_SHA256 = {checksum!r}
raw = base64.b64decode(BUNDLE_B64)
assert hashlib.sha256(raw).hexdigest() == BUNDLE_SHA256
CODE = Path('/kaggle/working/shogi-learning-code')
CODE.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
    for member in bundle.infolist():
        target = (CODE / member.filename).resolve()
        assert target.is_relative_to(CODE.resolve()), 'Unsafe archive path'
        assert member.file_size < 2_000_000
    bundle.extractall(CODE)
os.environ['PYTHONPATH'] = str(CODE / 'src') + os.pathsep + str(CODE / 'vendor')
print(json.loads((CODE / 'SOURCE.json').read_text()))
'''
    execute = '''import signal, subprocess
RUN = Path('/kaggle/working/shogi-learning-v1')
RUN.mkdir(parents=True, exist_ok=True)
command = [sys.executable, '-m', 'shogi_auto.learning.run', '--run-dir', str(RUN),
           '--device', 'cuda', '--max-wall-seconds', '2400']
remaining = max(1, 3000 - (time.monotonic() - NOTEBOOK_STARTED))
with (RUN / 'launch.log').open('a') as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               env=os.environ.copy(), start_new_session=True)
    try:
        process.wait(timeout=remaining)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        print('Stopped at the launch time limit. Partial progress is saved; this is not completion.')
    print('exit_code:', process.returncode)
print((RUN / 'launch.log').read_text()[-12000:])
if process.returncode not in (0, 2):
    raise RuntimeError('Experiment failed; preserve launch.log and stop the GPU session. Do not change permissions or install packages automatically.')
'''
    inspect = '''report_path = RUN / 'run-report.json'
assert report_path.exists(), 'No completed report; inspect launch.log and preserve partial files'
report = json.loads(report_path.read_text())
print(json.dumps({key: report.get(key) for key in ('status', 'device', 'gpu_training_verified', 'strength_improvement_verified', 'budget')}, ensure_ascii=False, indent=2))
training = report.get('training', {})
print('CUDA optimizer steps:', training.get('cuda_optimizer_steps_total'))
print('Validation/test metrics:', training.get('selected_metrics'))
for name, comparison in report.get('arena', {}).get('comparisons', {}).items():
    print(name, json.dumps(comparison, ensure_ascii=False))
assert report.get('gpu_training_verified'), 'GPU optimizer updates have not been verified'
if report.get('status') != 'evaluated':
    print('対局評価は未完了です。完了・棋力向上とは報告しないでください。')
'''
    finish = '''## 保存・中断・再開

1. エラー、CUDA不可、残量不足、予算到達、権限・追加インストール要求時は停止。
2. `shogi-learning-v1/` の全ファイルと `launch.log` を非公開Outputとして保存またはダウンロード。モデル・棋譜・checkpointはGitへ入れません。
3. 保存Versionが実際に残っていることを確認してセッションをOFF。終了後のアイドル時間も割当を消費します。
4. 再開は同じコード・設定・run directoryで行います。保存済みOutputをworkingへ戻して実行すると、生成済み局面・epoch・対局から再開します。
   累積実験予算はリセットされません。別セッションの割当・復元時の時間もUIで別途記録してください。
5. 比較結果が不明瞭・低下ならそのまま報告し、同じテスト開始局面で都合の良いモデルを選び直しません。
   次の実験は別seedの開始局面と新しいrun directoryを使い、原因を分けて評価します。
'''
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python'}},
                'cells': [cell('markdown', introduction), cell('code', setup), cell('code', unpack),
                          cell('code', execute), cell('code', inspect), cell('markdown', finish)]}
    path = ROOT / 'notebooks/kaggle_learning_v1.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    print(path, 'bundle bytes:', len(output.getvalue()), 'notebook bytes:', path.stat().st_size)


if __name__ == '__main__':
    main()
