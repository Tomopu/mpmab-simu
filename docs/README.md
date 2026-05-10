# MPMAB Simulator Docs

このディレクトリは、MPMAB シミュレーター実装のための設計メモ、論文擬似コード、比較実験手順をまとめる場所です。

## ドキュメント一覧

- [20260510_implementation_and_experiments.md](20260510_implementation_and_experiments.md)
  - シミュレーターの実装手順、比較実験、Claude Code への依頼テンプレート、Codex レビュー観点。

## 擬似コード

論文 PDF を直接読みに行かずに実装できるよう、TeX に記述された擬似コードを論文ごとに抽出しています。

- [Huang 2022](pseudocode/20260510_huang2022_pseudocode.md)
  - Homogeneous MPMAB without collision sensing。
  - `FindGoodArm`, `VirtualMusicalChairs`, `VirtualNumberPlayers`, `DistributedExploration` など。

- [Izumi 2026](pseudocode/20260510_izumi2026_pseudocode.md)
  - Homogeneous multi-channel MPMAB。
  - `data.tex` に記述された parallel 版のみ。
  - `FindMultipleGoodArms`, `ParallelVirtualMusicalChairs`, `ParallelVirtualNumberPlayers`, `HierarchicalDistributedExploration` など。

- [Shi 2021 BEACON](pseudocode/20260510_shi2021_beacon_pseudocode.md)
  - Heterogeneous MPMAB。
  - `BEACON: Leader`, `BEACON: Follower`, `Send`, `Receive` など。

- [Wang 2020 Orthogonalization](pseudocode/20260510_wang2020_orthogonalization_pseudocode.md)
  - collision-sensing initialization と DPE。
  - orthogonalization, rank assignment, exploration-exploitation phase など。

## 推奨の読み方

最初に実装する対象は homogeneous の 2 モデルなので、Claude Code には次の順に読ませる。

1. [実装計画](20260510_implementation_and_experiments.md)
2. [Huang 2022 擬似コード](pseudocode/20260510_huang2022_pseudocode.md)
3. [Izumi 2026 擬似コード](pseudocode/20260510_izumi2026_pseudocode.md)

heterogeneous 版に進むときは、Shi 2021 と Wang 2020 の擬似コードを追加で参照する。
