# 方策・価値モデル v2 の実行資材

この文書は実行前の固定資材・手順を記録したもの。GPU学習とCPU256局の実行結果は
[実測結果](learning-policy-v2-results.md)を参照。

2026-10-30 JST。CPU準備は完了、GPU学習と本対局は未実行。
新規CNNへの教師あり学習であり、強化学習・大会級の棋力・改善成功とはまだ言わない。
[教師の一次資料・条件・設計・事前評価計画](learning-policy-v2-plan.md)も参照する。

## 固定コードと自己生成データ

- Notebook: `notebooks/kaggle_policy_v2.ipynb`、1,080,220 bytes。
- Notebook SHA256: `7729162f847b8b17884362023f5b5179bfdb1e0f174750bc25898b054105a0da`
- 同梱コード生成元commit: `a4854e6f48d3ea22ee2504b5405f9921668c14fb`
- コードの内容SHA256: `e1a2940b8c07e32fcd906afe93c46cb265cbc89dc1c6b5ba56dde4787fefe5fb`
- ソースZIP SHA256: `b52427e7dc55a7515dd4c4f9111bf4eca0091a02e36ccc6d9ed6327cd4f1ff7d`
- データ生成元commit: `78540c56f0796f0d969824cc1bff0980bed1d419`。アプリの内容SHAは同じ。
- データZIP SHA256: `3c0dbbe0743ee2e2e6bb8bac4305c4c300223c891f3fbb830c0280338c11299a`、705,352 bytes。
- NPZ SHA256: `7be04d753bd5b4cb39fee474b1f8db33e07ee787e5b97a277558456ab7109af1`。
- train 3,804 / validation 515 / test 484。全splitでゲームIDと回転・鏡像等価局面の重複なし。

512ゲームの生成はCPUで459.85秒。491ゲームが詰み、21ゲームが生成上限160手に到達した。
8,192の予定採取時点のうち、早い終局で3,389に到達せず、到達した4,803件はすべてラベル化した。
教師深さ平均13.47、以前の実験データとの重複・教師応答による除外は0。
深さの数値だけから教師のEloを推定しない。

データZIPは盤面特徴・合法手ラベル・教師の数値応答・来歴だけを含む。
第三者のNNUE重み・実行ファイル・外部棋譜・非公開情報・checkpointはNotebookへ含めない。
ソースZIPの26項目にはレビュー済みのソースとpython-shogi 1.1.1のGPL原文を含める。
ソースZIP内の全アプリ`.py`を生成元commitと照合し、別ディレクトリでオフラインimportと内容SHA一致を確認済み。

## 親側のKaggle操作

1. このNotebookを含む最終コミットのraw GitHub URLを、Kaggleの **File → Import Notebook → Link** へ渡す。
   GitHubのbranch名ではなくコミットSHAをURLへ使う。同梱コードの生成元SHAと、Notebookを格納したコミットSHAは別。
2. 非公開で保存し、InternetはOFF。別Inputのアップロード・APIキー・pipインストールは不要。
3. その時点の無料GPU残量をKaggle UIで確認し、コード内の `REMAINING_FREE_GPU_HOURS=None` を実残量へ変更する。
   T4×2の場合は安全余裕込み約2.01 GPU時間以上、1 GPUの場合は約1.03 GPU時間以上を要求する。
   これは1 GPUで55分、2 GPU割当で約2倍消費する上限の検査。学習本体は1 GPUで動く。
4. 既存のPyTorch **2.11.0**、CUDAが利用可能な1〜2 GPU環境でコードセルを実行する。
   環境が異なる場合は更新せず停止して報告する。GPUが使えない場合のCPU代替実行はしない。
5. 80 epoch、batch 128、seed 314160。通常の見込みは数分〜20分だが、実機測定前なので保証しない。
   学習は累計3300秒、全体watchdogは3500秒＋終了猶予15秒。1時間以内の処理上限。
   validationのCE＋MSEだけで重みを選び、testは学習完了後の選択モデルでのみ測る。
6. 完了・停止後はOutputを**非公開**保存し、すぐセッションをOFFにする。
   セッションの起動待ちや終了待ちの無料枠消費は処理時間の推計に含まれないので、UIでも終了を確認する。

無料残量不足・CUDA不在・バージョン/ソース/データ/設定不一致・非有限値・watchdog到達は停止条件。
失敗した状態で追加GPUを再起動せず、`run-report.json`と`training-process.log`を親へ共有する。
再開は同じデータ・設定・保存済みcheckpointで、累計予算をリセットしない場合だけに限る。

## 保存物とCPUへの返却

Outputディレクトリは `/kaggle/working/shogi-policy-v2/`。
初期モデル`initial-policy.json`、選択モデル`best-policy.json`、`checkpoint.pt`、設定、
epochログ、`optimizer-accounting.json`、`training-report.json`、`run-report.json`、`budget.json`を保存する。
CUDA更新回数・実重みSHAの変化・validation/test指標を確認し、損失改善と棋力改善を分ける。

CPUへの推奨返却は `policy-v2-return.txt`。
JSONモデル・設定・実行ログ・SHA検査表を圧縮してBase64テキストにしたもので、pickleは含まない。
親側でダウンロードした同ファイルを正式なLibraryへ登録し、Library file ID・File Service file ID・
ファイル全体のbytesとSHA256をこのCPU担当へ渡す。
以前のsigned ZIP転送は失敗したが、Libraryの正式なテキスト読込とローカル復元は実際に成功している。
このテキストなら約数千行で分割取得できる。認証情報やネットワーク設定の変更は不要。

受取側はテキスト・ZIP・各項目のSHA、モデルの実重みSHA、生成元commit/データSHAを照合する。

```bash
.venv/bin/python scripts/import_policy_results.py outputs/policy-v2-return.txt \
  --destination runs/policy-v2-gpu-import --text-sha256 '<親から受け取ったテキストSHA256>'
```

CPUモデルによる同形式の往復復元では、元の全ファイルとバイト単位で一致した。
誤ったSHAの入力は展開前に拒否した。GPU出力での返却はこれから行う。

## 受取後のCPU評価

モデル受取後、同じ持ち時間・先後・開始局面を揃えた256局をCPUで行う。
学習後対学習前、学習後対材料評価、学習後対固定NNUE、学習前対固定NNUEの各64局。
開始局面seed20261106、各32先後ペア、0.10秒/手、上限384手、2 CPU並列。
教師は公式ソースから自前ビルドした固定バイナリと固定Háo NNUEをCPU側で再利用する。

```bash
.venv/bin/python -m shogi_auto.policy.arena \
  --run-dir runs/policy-v2-gpu-import --data-dir runs/policy-v2-data \
  --teacher-binary outputs/teacher-review/YaneuraOu-source/source/YaneuraOu-by-gcc \
  --eval-dir outputs/teacher-review/teacher/eval
```

CPU評価は累計3600秒上限。全局が終わらなければpausedとして不足を報告し、成功とは扱わない。
先後ペアの95%区間・保守的境界、千日手、打切り、探索深さの意味、時間超過・不正手・例外を報告する。
強い教師への敗北を隠さず、旧ベースラインへの改善と教師を超えたかは別に判定する。
入玉宣言・持将棋の完全裁定や大会CSA接続はこの実験の未対応項目。

## 確認済み範囲

全42テストがskipなしで成功。標準setupも成功。
少量のCPU教師データで学習・推論・合法手・8局の接続確認、optimizer再開一致、
両手番の実教師評価符号、バンドルのコミット対応、返却ファイルの復元を確認した。
8局の接続確認は短い手数上限で全引分なので、棋力評価へ流用しない。
GPU学習・本番256局・改善の有無は未検証。
