# Claude Code Instructions

このリポジトリでは、MPMAB シミュレーターを段階的に実装する。

## 最初に読む資料

初回の実装前のみ、以下を読む。

1. `docs/README.md`
2. `docs/20260510_implementation_and_experiments.md`
3. `docs/pseudocode/20260510_huang2022_pseudocode.md`
4. `docs/pseudocode/20260510_izumi2026_pseudocode.md`

heterogeneous 版に着手する場合のみ、以下も読む。

- `docs/pseudocode/20260510_shi2021_beacon_pseudocode.md`
- `docs/pseudocode/20260510_wang2020_orthogonalization_pseudocode.md`

## 実装ルール

- Python 内部では arm index と player index を 0-based に統一する。
- 論文中の 1-based index を実装へ移す箇所では、対応関係を日本語コメントで明記する。
- no-sensing アルゴリズムには collision flag を渡さない。collision は環境と評価指標では記録してよいが、意思決定には使わせない。
- 既存ファイルの不要なリファクタリングは避け、実装計画に沿って小さく追加する。
- 確率的処理は seed を受け取り、実験とテストで再現できるようにする。
- 実装後は `pytest` で確認できるテストを追加する。

## コメント方針

作成するコードには、日本語で細かくコメントを書く。

また、メソッドや関数には Docstring を書いて、引数、返り値、処理内容を説明する。

特に論文アルゴリズムを実装する箇所では、以下のように処理手順が追えるコメントを入れる。

```python
# 1. フェーズ内で使う報酬和と試行回数を初期化する
# 2. active arm から一様ランダムに選んで探索する
# 3. 閾値を超えた arm だけ確認フェーズに進める
```

ただし、代入や単純な return など自明な行にまでコメントを付けすぎない。コメントは、論文の手順、index 変換、同期処理、collision の扱い、通信コストの近似など、実装者やレビュアーが迷いやすい箇所を優先する。

## レビューしやすさ

- 論文擬似コードの関数名に近い名前を使う。
- 近似や省略をした場合は、該当コードに日本語コメントで理由を書く。
- 実験結果を出すコードでは、設定値、seed、trial 番号を保存する。
