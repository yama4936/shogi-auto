# Kaggle GPU 動作確認

## 最短手順（API キー不要）
1. Kaggle にログインし、[新規 Notebook](https://www.kaggle.com/code/new) を開く。
2. GitHub の `notebooks/kaggle_gpu_smoke.ipynb` をダウンロードし、Notebook エディタの Import / Upload Notebook で読み込む。
3. Settings → Accelerator で **そのアカウントで利用できる GPU** を選択する。
   無効なら Kaggle に表示されるアカウント認証や割当量の案内に従う。機種、GPU 枠、利用時間は保証しない。
4. 全セルを実行。`status: passed` と `device: cuda` を確認する。
   CUDA がなければ意図的にエラーとなる。CPU 実行を GPU 成功として扱わない。
5. `/kaggle/working/gpu-smoke.json` を出力として保存またはダウンロードする。
6. 終わったらセッションを停止する。GPU セッションは割当量を消費する。

Notebook は自己完結しており、スモークテストには Internet 設定や秘密情報は不要です。
学習・モデル精度・大会参加可否を確認するテストではありません。

## 後で公開 GitHub コードを利用する場合
Internet を有効にし、Notebook のセルで以下を実行できます（公開リポジトリのためトークン不要）。

```python
import subprocess
from pathlib import Path
repo = Path('/kaggle/working/shogi-auto')
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/yama4936/shogi-auto.git', str(repo)], check=True)
subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True)
# 実験を再現する場合は、レビュー済みのコミット SHA を checkout して記録する。
subprocess.run(['python', str(repo / 'scripts/gpu_smoke.py'),
                '--output', '/kaggle/working/gpu-smoke.json'], check=True)
```

既存 checkout は自動で pull/reset しません。再実行時は表示された SHA を確認してください。
学習時はコード SHA・データ出所/ライセンス・seed・設定・実行環境を保存し、途中 checkpoint で再開可能にします。
重みやデータは Git に入れず、保存済み Output/Dataset などを使います。セッション終了前に保存を確認してください。

## 参考（2026-10-29 確認）
- [Kaggle Notebooks 公式ドキュメント](https://www.kaggle.com/docs/notebooks)
- [NVIDIA: Kaggle GPU Notebook の設定](https://docs.nvidia.com/datascience/deployment/26.10/platforms/kaggle/)
- [Kaggle 公式 CLI: kernels](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)

CLI 自動投稿や API 認証は初期設定では導入しません。必要になったら秘密情報ストアを使い、Notebook/Git にキーを書かないでください。
