# 使用ライブラリ

対局機能の盤面・合法手生成には、作者が公開する **python-shogi 1.1.1** を利用します。
将棋連盟などの公式エンジンではありません。

- 配布元: <https://pypi.org/project/python-shogi/1.1.1/>
- ソース: <https://github.com/gunyarakun/python-shogi>
- 作者: Tasuku SUENAGA および upstream contributors（配布元ソースの著作権表示を参照）
- 配布パッケージのライセンス表記: **GPL-3.0-only**
- 配布パッケージに同梱された原文: [GPL-3.0](licenses/python-shogi-GPL-3.0.txt)
- 利用箇所: `game.py`、`engine.py`、`usi.py`。コードの直接コピーや改変版の同梱は行っていません。

ブラウザ自動検証だけで使用する任意依存は **Playwright 1.62.0**（Apache-2.0）。
<https://github.com/microsoft/playwright-python>。通常の対局とユニットテストには不要です。
ブラウザには別途インストール済みの Chromium または Playwright の Chromium を使います。

再配布時は依存ライブラリのライセンス条件を確認してください。
通常の対局UIには棋譜データ・学習済み重み・外部エンジン・画像・外部フォントを追加していません。

学習用の任意依存は NumPy（BSD-3-Clause）と PyTorch（BSD形式、同梱部分の条件は各配布物を参照）。
[NumPyライセンス](https://github.com/numpy/numpy/blob/main/LICENSE.txt)、[PyTorchライセンス](https://github.com/pytorch/pytorch/blob/main/LICENSE)。
CPU検証では NumPy 2.3.5 / PyTorch 2.11.0+cpu を使用。Kaggleでは既存版を再利用し、更新しません。
オフライン学習Notebookは、公開済みの本リポジトリの学習コードと python-shogi 1.1.1 の `.py` ソース・GPL原文だけを同梱します。
第三者のモデル・棋譜・データ・バイナリ実行ファイルは同梱しません。cshogiは依存関係の比較のみで、採用していません。

方策・価値モデルv2のCPU教師は、公式やねうら王v8.60gitのGPLv3ソースから自前ビルドし、
tanuki- Háo 2023-05-08の公式重み配布（重みとGPLv3原文を同梱）を使用します。
取得元・固定commit・SHA256・用途別の条件は[実験計画](learning-policy-v2-plan.md)を参照してください。
外部重み・ビルド済みエンジン・生成棋譜はGit及びGPU用ソースバンドルに含めません。
