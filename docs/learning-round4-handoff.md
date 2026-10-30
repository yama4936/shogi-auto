# 追加2回目/3（v4）の実行資材

v3の学習・256局比較・大会ルール6局は[結果](learning-round3-results.md)に記録済みで、追加1/3回が完了した。
v4は未学習。CPUの小さな機能試験4更新を実験完了へ数えない。
[変更理由と固定条件](learning-round4-plan.md)、[機械可読な事前検証](evidence/policy-round4-preflight.json)を参照。

## 固定する来歴

Notebookは`notebooks/kaggle_policy_round4.ipynb`。
学習コードcommitは`eaa315becdc9eb041ebe8d98ac6633c167151ca0`。
Notebookを格納する後続commitとは役割が異なり、後続の修正を勝手に取り込まない。

| 対象 | SHA256 |
|---|---|
| Notebook | `0b6604fc7f7e507103b35e9355ae37554d9d58cd0eb74d0a22b585fb74fe11b0` |
| アプリ内容 | `28664d55b5901c2fd6d3de38e3de44d895ef5f59ee81a7b5646af5b32d53079b` |
| ソースZIP | `d9aed9099766085bed286a951482a15e70766fb0c99a0d8ab3c36c6375231433` |
| 入力ZIP | `7a37d2ff044a37a621f97a05737e427275f3479c09fc9edf13b3f87b4f0cbe2e` |
| 入力NPZ | `b1efbe76e8c695875ff44861ddf5d6f7743ebe31841c3390b308705035bccbc5` |
| 開始モデル実重み | `99600837fb9af11e5e7c21921adf77eaa434e33a63be15065f261e4552c23d8e` |

新規5,600行＋v3のtrainだけ9,636行。train14,178 / validation538 / test520。
新規生成384ゲーム・CPU594.28秒。生成・入力準備のcommitは`6987ccbcfc742fe3254f765bd6a436cfd89c5203`。
その後、評価用開始局面の除外を補強したため学習コードcommitとは異なる。特徴・学習・探索・ネットワークの実装は同じである。
過去の入力と65,082開発対局到達局面を除外し、分割と回転・鏡像の重複を検査した。
入出力は自己生成データと自分のJSONモデルのみ。教師の重みやバイナリ、他者棋譜は含まない。
やねうら王・Háoの固定URL・commit・GPLv3と大会前の利用根拠は[利用条件](learning-policy-v2-plan.md)を維持する。

## 非公開入力の復元

Libraryの`policy-round4-inputs.txt`は3,770,434 bytes、14,674行。
テキストSHA256は`bc5061ade9507ed62d7db6da38f132e78f11d4d51b061554d533d6a35a996311`。
復元する`policy-round4-inputs.zip`は2,816,688 bytesで、上表の入力ZIP SHAと一致させる。

```bash
python3 scripts/unpack_round_input_text.py policy-round4-inputs.txt policy-round4-inputs.zip \
  --text-sha256 bc5061ade9507ed62d7db6da38f132e78f11d4d51b061554d533d6a35a996311 \
  --zip-sha256 7a37d2ff044a37a621f97a05737e427275f3479c09fc9edf13b3f87b4f0cbe2e
```

ZIPは`data.npz`、`data-manifest.json`、`start-policy.json`、`SHA256SUMS.json`の4ファイル。
manifest内の前回レビューSHAと開始重みは、完了したv3の記録に結び付いている。
入力のZIPそのもの・Kaggleが展開した4ファイルの両方をCPUで検証し、改変時の拒否も確認した。
署名付き転送が使えない場合はLibrary native readを連続1000行ずつ取得し、LFを保って結合できる。

## 親側Kaggleでの実行

1. Notebook保存commitの固定raw URLをFile → Import Notebook → Linkへ渡す。
2. 入力ZIPを非公開Dataset/Inputとして追加する。既存のv3入力と混ぜない。
3. Notebookも非公開、Internet OFF。現在の無料残量を確認して`REMAINING_FREE_GPU_HOURS`へ記入する。
   1〜2 GPUのみ対応し、2基なら上限と余裕を含め約0.90 GPU時間以上を要求する。
4. 既存torch2.11.0/CUDAを使う。pip更新・外部ダウンロード・APIキーは不要。
5. 最大40epoch（最大4,440更新）・累計学習1200秒・全体1500秒＋終了猶予15秒。
   seed314162、batch128、学習率0.0003、weight decay0.03、合法手smoothing0.05。
   train抽出重みは非飽和局面2／その他1、訓練損失CE＋2×MSE。
   validationの非平滑CE＋MSEだけで選択し、最小6epoch・改善なし6epochで終了する。
6. 見込みは1〜5分だが実機未測定。Outputを非公開保存し、セッションOFFを確認する。

CUDA不在、バージョン・SHA・開始モデル不一致、無料残量不足、非有限値、予算上限では停止する。
CPU教師生成や対局をGPU Notebookへ追加せず、途中終了・選択重み据置もそのまま報告する。

## 回収とCPU検証

Outputは`/kaggle/working/shogi-policy-round4/`、返却は`policy-round4-return.txt`。
Library ID・File Service ID・bytes・テキストSHA256・復元ZIPのbytes/SHAを共有する。
安全なモデルJSONで検証し、checkpointはKaggle Outputに保存する。CPU側で不要なpickleを読まない。

CPUでは固定コード`eaa315b…`を使い、seed20261122の32開始局面を先後交換して320局。
前回直前モデル、固定材料評価、固定教師、直前モデル対教師に加え、固定v2対戦を64局行う。
全比較の相手と予算を維持し、重複を避けるため開始候補13件を除外した。
開始局面ハッシュは`9ed93b4cfa746f43651948d7c03fa9fbb453b3dbf7cd3efb478960b215a722de`。
CPU呼出しでは`--fixed-v2-model`と、事前記録にある`--exclude-data`一式を必ず渡す。
大会時計・開始4手・512手・宣言の少数対局は別集計する。実大会サーバーへは接続しない。
最終testは全3回の方法・モデルを固定するまで計算しない。v4結果を確認してからv5を選ぶ。
