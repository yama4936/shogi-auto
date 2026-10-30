# 追加1回目/3（v3）のGPU実行資材

公式CSA仕様との追加照合で時計・通信部を修正した。[照合結果](csa-protocol-validation.md)を参照。
以下は引渡し時の固定資材の記録。GPU学習と短時間比較のコードは保持し、大会時計の検証は別版で行う。

2026-10-30 UTC。準備・CPUルール試験は完了。v3 GPU学習と学習後の対局は未実施。
追加実験の完了数は現在0/3。[全3回の計画](learning-three-round-plan.md)を参照。
後の2回の方法は直前の実結果から確定し、同じ実験を形式的に3回走らせない。

## 固定Notebookと入力

- Notebook: `notebooks/kaggle_policy_round3.ipynb`、149,368 bytes。
- Notebook SHA256: `e291c05b577a1f4a6fab8b7cc5b704f38cd9b2e1428cf2295933e50c5f815f11`
- 同梱コードcommit: `71f8cd513399dfc00d48baff22a796affee9edbb`
- アプリ内容SHA256: `32bf855152e97a5a590ccab56ff5cbaba9867ac13cccd3fbbf0c8d3ac76b6610`
- ソースZIP SHA256: `6d445882551616571e5f171b05a9047b9b8ce331f28dcabee94bcf33ff31679f`
- 非公開入力ZIP: `policy-round3-inputs.zip`、2,246,639 bytes。
- 入力ZIP SHA256: `0f631cb0db48c11934314bba5a00c39e7bb29a7955e0c2e0f854094c8f795dd6`
- NPZ SHA256: `8a5d1c28fdf525df3c44fa78eabc880435b1ab3aab49d3fa3e4dfe7c444505e9`
- 自前のv2開始重みSHA256: `922df69128314e622bfbca469bba3fb19867adedccb8ae9d9d8890fb489f7abb`

新規7,318局面＋v2 trainだけ3,804局面。train9,636 / validation738 / test748。
旧validation/testの再利用は0。ゲームIDと回転・鏡像を含む局面の分離を検査済み。
新規生成は名前空間`denryusen-dr7-round3`、大会開始4手を含めて768ゲーム、CPU676.92秒。
生成元commitは`e17e2aba93c5e33dd6e06cd0b5297506eafed887`。
後続の学習・ルール用ソースとはcommitが違うため、生成元と準備元をmanifestで分けて記録している。
特徴定義SHAは同じで、GPU側でも照合する。

最初の候補は旧seed+game_id方式の範囲がv2と重なると検出し、不採用とした。
その候補6,452件・CPU654.34秒はGPU学習に使わず、修正後の新規生成と混ぜていない。
局面単位の除外だけに頼らず、実験名を含むSHA256から軌跡seedを分離して再生成した。
両生成の合計CPU時間は約1331秒で、今回の生成予算1800秒以内。

教師は既存と同じ公式GPLv3ソース由来のやねうら王＋Háo NNUE。
大会前の教師生成許可と個別ライセンスを[計画](learning-policy-v2-plan.md)に記録済み。
入力は自己生成データ・自分のv2 JSONモデル・来歴・SHAだけで、元の教師重み・バイナリ・外部棋譜を含まない。
モデルを含む入力を公開Git/Notebookへ埋め込まず、非公開Libraryから渡す。

## 入力の受取とKaggle操作

親へ共有した非公開Libraryの`policy-round3-inputs.txt`を取得する。
テキストは3,007,408 bytes、SHA256 `3fe1a4919dae4277473a636e7b8acc253e293ae6cfec4b799d8021a311a41f06`。
正式なLibraryテキストreadで全11,705行を取得し、ZIPまで往復復元して元とバイト一致することを確認済み。
署名付きダウンロードが使えない場合も、native readの連続1000行チャンクをそのままLF付きで結合できる。

```bash
python3 scripts/unpack_round_input_text.py policy-round3-inputs.txt policy-round3-inputs.zip \
  --text-sha256 3fe1a4919dae4277473a636e7b8acc253e293ae6cfec4b799d8021a311a41f06 \
  --zip-sha256 0f631cb0db48c11934314bba5a00c39e7bb29a7955e0c2e0f854094c8f795dd6
```

1. 固定したNotebook格納commitのraw GitHub URLをKaggleのFile → Import Notebook → Linkへ渡す。
2. ZIPを非公開Dataset/Inputとしてアップロードし、このNotebookへ追加する。
   KaggleがZIPを展開する場合も、4ファイルが同じフォルダにあれば中身のSHAで一致確認する。
   ZIPそのものを置く場合は全体SHAも確認する。両方式・改変時の拒否をCPUで実行確認済み。
3. NotebookとInputを非公開、Internet OFFにする。APIキー・外部ダウンロード・pip更新は不要。
4. その時点の無料GPU残量をUIで確認し、`REMAINING_FREE_GPU_HOURS=None`を実残量へ変更する。
   1〜2 GPU割当のみ許可し、T4×2なら上限と余裕を含め約0.90 GPU時間以上を要求する。
   表示の解釈や将来の利用可能性をこちらから推測せず、残量不足・利用上限では止める。
5. torch2.11.0と既存CUDAをそのまま使う。違う環境なら自動更新せず停止して共有する。
6. 実行後にOutputを非公開保存し、セッションOFFを確認する。教師生成と対局はGPU Notebookで行わない。

学習は自分のv2重みから、学習率0.0003・weight decay0.03・合法手内label smoothing0.05。
seed314161、batch128、最大40epoch（最大3040更新）、最小6epoch、validation改善なし6epochで終了。
validationの非平滑CE＋MSEだけで選択し、testはGPUで評価しない。
見込みは1〜5分だが実機未測定。累計学習1200秒、全体監視1500秒＋終了猶予15秒。
起動・停止待ちの無料枠消費は別なので、UIでも終了を確認する。

非有限値・CUDA不在・バージョン/ソース/データ/モデル不一致・残量不足・上限到達は停止条件。
途中終了や選択重み据置を改善成功と扱わず、実更新数・epoch・停止理由を記録する。

## 結果の回収

Outputは`/kaggle/working/shogi-policy-round3/`。
`policy-round3-return.txt`をLibraryへ保存し、そのLibrary ID・File Service ID・bytes・全体SHA256をCPU担当へ渡す。
初期/選択モデルJSON、設定、GPU実更新数、ログ、チェックサムを含み、CPU評価でpickleは読まない。
全epochのcheckpointはKaggle Outputへ非公開保存し、返却用テキストには含めない。

CPU側で全SHAと実重みを検証後、seed20261121の256局短時間比較と、別集計のローカル大会ルール対局を実行する。
v3の結果を受けてv4の仮説・設定を決める。全3回を固定してから最終testと共通最終対局を行う。

## CPUで確認済みの範囲

- 全53テストがskipなしで成功。旧v2 Notebookも維持。
- 開始4手、512手境界の詰み優先順位、先後27点宣言条件、残時間、誤宣言、時計切捨て・加算、非対称得点。
- CSAの模擬login/Agree/STARTと、両色の序盤切断の負け判定。socketpair上の検証で、実サーバー接続ではない。
- v2モデルを使った大会ルールのローカル予備6局が288.37秒で完了。開始4手・正式持ち時間・512手・宣言と得点を適用。
  v2学習後はv2初期に2勝0敗、材料評価に0勝2敗、教師に0勝2敗。
  不正手・時間切れ・誤宣言・異常0。v3の学習/対局として数えず、6局から棋力の信頼区間を出さない。
- 入力ZIP/展開済みファイルの復元、改変拒否、ソースZIP内の全アプリ`.py`と固定commitの一致。

実大会CSAサーバーの認証・通信遅延・互換性は未検証。完全なクライアント統合と本番接続試験を残している。
この段階で大会出場可能・学習改善成功とは報告しない。
