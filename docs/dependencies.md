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
棋譜データ・学習済み重み・外部エンジン・画像・外部フォントは追加していません。

学習用の任意依存は NumPy（BSD-3-Clause）と PyTorch（BSD形式、同梱部分の条件は各配布物を参照）。
[NumPyライセンス](https://github.com/numpy/numpy/blob/main/LICENSE.txt)、[PyTorchライセンス](https://github.com/pytorch/pytorch/blob/main/LICENSE)。
CPU検証では NumPy 2.3.5 / PyTorch 2.11.0+cpu を使用。Kaggleでは既存版を再利用し、更新しません。
オフライン学習Notebookは、公開済みの本リポジトリの学習コードと python-shogi 1.1.1 の `.py` ソース・GPL原文だけを同梱します。
第三者のモデル・棋譜・データ・バイナリ実行ファイルは同梱しません。cshogiは依存関係の比較のみで、採用していません。
