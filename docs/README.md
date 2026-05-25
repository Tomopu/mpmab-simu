# MPMAB Simulator Docs

このディレクトリは、MPMAB シミュレーター実装のための設計メモ、論文擬似コード、比較実験手順をまとめる場所です。

## ドキュメント一覧

### 設計・計画

- [20260510_implementation_and_experiments.md](20260510_implementation_and_experiments.md)
  - シミュレーターの実装手順、比較実験設定、Claude Code への依頼テンプレート、Codex レビュー観点。

### 実験ガイド

- [20260522_experiment_guide.md](20260522_experiment_guide.md)
  - 実験コマンド・パラメータ設定のリファレンス。プリセット一覧、全 CLI オプション、出力 CSV の列定義、テスト実行方法。

### 実装メモ

- [implementation_notes.md](implementation_notes.md)
  - Homogeneous 版（Huang 2022 / Izumi 2026）の実装概要、ソースパス対応、簡略化箇所、比較実験コマンド。

- [20260511_fallback_inventory.md](20260511_fallback_inventory.md)
  - 論文手順から外れる安全装置 fallback の所在と方針。

- [20260511_paper_faithful_reproduction_plan.md](20260511_paper_faithful_reproduction_plan.md)
  - simplified 実装を論文再現実装に近づけるための作業計画。

- [20260518_bugfix_report.md](20260518_bugfix_report.md)
  - Huang 2022 / Izumi 2026 の実装バグ調査と修正レポート（T1 式、reject arm 確認チェック、τ_rank など）。

- [20260522_bugfix_hde_n_greater_than_1.html](20260522_bugfix_hde_n_greater_than_1.html)
  - HDE n>1 バグ（C_assign 除外による Leader 割当経路なし・M0 デクリメントによる Follower idx 負値）の原因・修正を
    インタラクティブアニメーション付きで解説した HTML レポート。

## 擬似コード

論文 PDF を直接読みに行かずに実装できるよう、TeX に記述された擬似コードを論文ごとに抽出しています。

- [Huang 2022](pseudocode/20260510_huang2022_pseudocode.md)
  - Homogeneous MPMAB without collision sensing。
  - `FindGoodArm`, `VirtualMusicalChairs`, `VirtualNumberPlayers`, `DistributedExploration` など。

- [Izumi 2026](pseudocode/20260510_izumi2026_pseudocode.md)
  - Homogeneous multi-channel MPMAB。
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
