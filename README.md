# shogi-auto

電竜戦への参加を目指す将棋 AI 開発リポジトリ。
**ブラウザで人間とCPU AIが対局できる最小版です。**
材料評価と短時間探索の練習用AIで、学習済みモデルや大会接続は未実装です。

## まず将棋を指す

Python 3.11以上（`venv` / `pip` を含む）。初回の依存インストールにはネット接続が必要です。

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m shogi_auto.web
```

同じマシンのブラウザで **http://127.0.0.1:8000** を開いてください。
インストール後は `shogi-auto-web` でも起動できます。停止は Ctrl+C。

- 自分の駒をタップし、光った移動先をタップして指します。成れる場合は「成る / 成らない」を選択します。
- 持ち駒をタップしてから光った空きマスを選ぶと駒を打ちます。行き所のない駒は成りが自動確定します。
- 「あなたの手番」を先手・後手から選び「もう一局」で開始。後手の場合は盤が反転し、AIが先に指します。
- 日本語の手番、王手、棋譜（簡易表記とUSI）、持ち駒、終局を表示します。投了・再対局に対応します。
- 不正手、自玉への王手放置、二歩、打ち歩詰め、行き所のない駒打ちはサーバー側でも拒否します。
- 同じブラウザでは再読み込み後も対局を再開できます。ブラウザごとに対局を分け、別タブの古い操作は拒否します。

Cloudで起動した場合、このURLは **Cloud内のループバック** です。スマホで開くには、利用環境が提供する認証付きポート転送等が別途必要です。
外部公開や転送の設定は本実装では行いません。開発用HTTPサーバーであり、インターネット公開用ではありません。
必要な場合だけ `--host <信頼できる接続先IP> --port 8000` を指定できます。通常は既定のループバックで使用してください。

### 対応ルールと制限

盤面・合法手は [python-shogi 1.1.1](https://github.com/gunyarakun/python-shogi) に委ねます。
詰み・合法手なし・投了で終局し、4回の同一局面は千日手の引き分け、連続王手の千日手は王手を続けた側の負けとして判定します。
**持将棋、入玉宣言、時間切れ、待った、対局ファイルへの保存・読み込みは未対応**です。
対局はメモリ上のみで、サーバー停止または6時間の未操作で消えます。棋譜は正式なKIFではありません。

AIは持ち駒を含む材料評価と、最大3手先の反復深化・alpha-beta探索を使います。
既定の探索予算は約0.35秒または6000ノード（早い方）。ノード境界で停止するため厳密な時間上限ではありません。
探索中の千日手は簡易的に引き分け評価とし、実対局の終局処理では連続王手を区別します。
詰み探索、定跡、学習済みモデルはなく、強さはまだ測定していません。
依存元とライセンスは [使用ライブラリ](docs/dependencies.md) を参照してください。

## 役割分担
- GitHub: ソースコード・テスト・実験手順を管理。
- Codex Cloud: 実装、レビュー、CPU のユニットテスト。GPU は前提にしません。
- Kaggle: GPU 上での実験・将来の学習。常時稼働する大会用ホストにはしません。
- 大会用マシン: 推論・探索・対局接続を別途設計し、対象大会の規約を確認します。

## 開発を開始
Python 3.11 以上。セットアップは `.venv` に通常依存のみをインストールします。PyTorchはインストール・更新しません。

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m unittest discover -s tests -v
```

依存インストール後のユニットテストはCPUのみで実行でき、外部ネット接続・認証情報・CUDAは不要です。
HTTPテストではローカルの一時ポートを使用します。

任意のブラウザ自動検証（Chromiumが必要）:

```bash
python -m pip install -e '.[browser]'
python scripts/browser_smoke.py --chromium /usr/bin/chromium
# システムのChromiumがない場合:
# python -m playwright install chromium
# python scripts/browser_smoke.py
```

実際のChromiumで初手から3往復、成り・不成・強制成り、駒打ち、不正操作、詰み、投了、再対局、後手、スマホ幅を確認します。
`outputs/browser/` に確認画像を保存します（Git対象外）。特殊局面はテストプロセス内だけで設定し、Web APIには任意局面の入力機能を公開しません。

### USI（基本対応）

`python -m shogi_auto.usi` または `shogi-auto-usi` で標準入出力のUSIを使えます。
`usi`、`isready`、`usinewgame`、`position startpos [moves ...]`、`position sfen ... [moves ...]`、`go [movetime ミリ秒]`、`stop`、`quit` に対応します。
`bestmove` は合法手（合法手がない場合は `resign`）を返します。探索は最大2秒に制限します。
`btime/wtime/byoyomi` の時計管理、ponder、infinite、go mate、オプション設定、USIでの千日手裁定は未対応です。
**対局GUI接続の試作用であり、大会用のUSI完全対応ではありません。**

Kaggle では既存の PyTorch / CUDA を使い、setup スクリプトから再インストールしません。
ローカルで行列演算確認も行う場合のみ、[PyTorch 公式](https://pytorch.org/get-started/locally/)で環境に合う版を入れて
`python scripts/gpu_smoke.py --allow-cpu` を実行します。

## Kaggle GPU
[手順](docs/kaggle.md)に沿って `notebooks/kaggle_gpu_smoke.ipynb` をインポートしてください。
この Notebook は外部データや API キー不要で CUDA 上の行列演算を確認します。
**Notebook を作っただけでは GPU 利用可能とは言えません。実際の CUDA テスト成功が必要です。**

## 次の開発段階
1. 固定対局条件で現在のCPU AIの棋力を測り、詰み探索・評価関数を改善する。
2. 持将棋・入玉宣言などのルールと、対局保存・読み込みを追加する。
3. USIの時計管理とGUI互換性を検証する。
4. 学習データ、モデル、再開可能な checkpoint と実験記録を設計して Kaggle GPU 学習へ進む。
5. 対象回の電竜戦の公式要項を確認し、必要なら CSA 対局クライアントを接続試験する。

大会申込・日程・接続方式・本番機の要件はまだ確定していません。
学習と推論の設計は、既存エンジン利用か自作かを決めてから進めます。

## 安全な管理
このリポジトリは公開です。API キー、`kaggle.json`、大会パスワード、棋譜・重みなどはコミットしません。
成果物は Kaggle の Output などに別保存し、Git には取得元・ライセンス・ハッシュを記録します。
