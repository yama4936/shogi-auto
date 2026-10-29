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
