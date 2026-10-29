# shogi-auto

電竜戦への参加を目指す将棋 AI 開発リポジトリ。
**現在は開発・GPU 動作確認の土台のみです。対局エンジン、学習済みモデル、大会接続は未実装です。**

## 役割分担
- GitHub: ソースコード・テスト・実験手順を管理。
- Codex Cloud: 実装、レビュー、CPU のユニットテスト。GPU は前提にしません。
- Kaggle: GPU 上での実験・将来の学習。常時稼働する大会用ホストにはしません。
- 大会用マシン: 推論・探索・対局接続を別途設計し、対象大会の規約を確認します。

## 開発を開始
Python 3.11 以上。基本テストは標準ライブラリだけで実行できます。

```bash
bash scripts/setup.sh
python -m unittest discover -s tests -v
```

任意のパッケージインストール: `python -m pip install -e .`。
Kaggle では既存の PyTorch / CUDA を使い、setup スクリプトから再インストールしません。
ローカルで行列演算確認も行う場合のみ、[PyTorch 公式](https://pytorch.org/get-started/locally/)で環境に合う版を入れて
`python scripts/gpu_smoke.py --allow-cpu` を実行します。

## Kaggle GPU
[手順](docs/kaggle.md)に沿って `notebooks/kaggle_gpu_smoke.ipynb` をインポートしてください。
この Notebook は外部データや API キー不要で CUDA 上の行列演算を確認します。
**Notebook を作っただけでは GPU 利用可能とは言えません。実際の CUDA テスト成功が必要です。**

## 次の開発段階
1. 将棋ルール／合法手生成ライブラリを評価し、ライセンス・棋譜の利用条件を確認する。
2. 最小の USI エンジン（まず合法手を返す）とプロトコルテストを実装する。
3. 評価関数・探索の基準実装を作り、固定対局条件で棋力を測る。
4. 学習データ、モデル、再開可能な checkpoint と実験記録を設計して Kaggle GPU 学習へ進む。
5. 対象回の電竜戦の公式要項を確認し、必要なら CSA 対局クライアントを接続試験する。

大会申込・日程・接続方式・本番機の要件はまだ確定していません。
学習と推論の設計は、既存エンジン利用か自作かを決めてから進めます。

## 安全な管理
このリポジトリは公開です。API キー、`kaggle.json`、大会パスワード、棋譜・重みなどはコミットしません。
成果物は Kaggle の Output などに別保存し、Git には取得元・ライセンス・ハッシュを記録します。
