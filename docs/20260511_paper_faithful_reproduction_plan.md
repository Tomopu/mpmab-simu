# Paper-Faithful Reproduction Plan

作成日: 2026-05-11

このドキュメントは、現在の simplified simulator を「論文再現を名乗れる実装」に近づけるための作業計画である。疑似コード Markdown だけでなく、`papers/` 配下の TeX 原文・付録・参考コードを一次情報として照合する。

## 背景

現在の実装には、実験を止めないための安全装置が入っていた。

- rank 推定後の `j_list` 正規化（2026-05-11 削除済み）
- `M_hat_list = max(m_hat, M)` による真値補正（2026-05-11 削除済み）
- Grand Leader 不在時の fallback leader（2026-05-11 削除済み）
- `C_accept` 後の補完割当
- Izumi 2026 の `_try_assign` における accepted good arms fallback
- forced-collision bit 通信の直接集約近似

これらは実験用プロトタイプとしては有用だが、論文再現としてはバグや仕様ズレを隠す。削除は最後に行い、まず原文と照合して前段の実装を正す。

## 一次情報

### Huang 2022

| 対象 | 原文 |
|------|------|
| 本文 TeX | `papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/arXiv-2103.13059v2/main.tex` |
| PDF | `papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/2103.13059v2.pdf` |
| 参考コード | `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/code/huang2022/huang2022.py` |

確認済みの原文要点:

- `VirtualMusicalChairs` は `K * tau` time slots で、distinct external rank を高確率で割り当てる。
- `VirtualNumberPlayers` は `2K` rounds の sequential hopping により、`M_hat = M` と distinct internal rank `j in {1,...,M}` を高確率で出力する。
- `DistributedExploration` では `j=1` が leader であり、他は follower。
- `ComLeader` / `ComFollow` は accepted / rejected set を通信し、`C_accept` の長さで active player 数 `M'` を更新する。

### Izumi 2026

| 対象 | 原文 |
|------|------|
| 本文 TeX | `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/data.tex` |
| 付録 TeX | `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/appendix.tex` |
| PDF | `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/C000063_Izumi_fin.pdf` |
| 参考コード | `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/code/` |

確認済みの原文要点:

- `ParallelVirtualMusicalChairs` は外部 rank `s in {1,...,K}` を一意に割り当てる意図。
- `ParallelVirtualNumberPlayers` は `M_hat` と internal rank `j in {1,...,M_hat}` を推定する。
- `data.tex` では `ComGrandLeader`, `ComSubLeader`, `ComFollower` の詳細擬似コードは省略されている。
- `appendix.tex` には Sequential 版と ComGrandLeader / ComSubLeader / ComFollower の補助的擬似コードがあるため、Izumi 2026 の faithful 実装では `data.tex` だけでなく `appendix.tex` も照合する。

## 論文上の不変条件

まず以下をテスト化する。

### Rank / Player Count

- VMC / ParallelVMC の完了後、全 player の `s >= 0`。
- VMC / ParallelVMC の完了後、`s` は重複しない。
- VNP / ParallelVNP の完了後、全 player の `M_hat == M`。
- VNP / ParallelVNP の完了後、`j_list` は `1..M` の permutation。
- `j=1` の player が必ず存在する。

### Distributed Exploration

- `j=1` 不在 fallback なしで DE / HDE を開始できる。
- `C_accept` が非空のとき、論文の assign 条件だけで割当対象 player が決まる。
- `M'` の更新は論文通り `M' - |C_accept|` とし、未割当補完で調整しない。
- accepted arm と rejected arm の inactive 化が ComLeader / ComFollow / AssignAndUpdate と一致する。

### Communication

- Huang 2022 の `EncoderSendFloat` / `DecoderReceiveFloat` を原文通りに再現する。
- follower から leader への float 送信と leader から follower への int set 送信を、forced-collision bit 通信として時刻単位で表現する。
- Izumi 2026 は `appendix.tex` の ComGrandLeader / ComSubLeader / ComFollower をもとに、階層通信経路を explicit に再現する。

## 作業順序

### Phase 1: 原文照合表を作る

以下の対応表を docs に作る。

- Huang 2022: Algorithm 2, 3, ComLeader, ComFollow と現実装の行単位対応
- Izumi 2026: Algorithm 2, 3, HDE, appendix Com* と現実装の行単位対応

成果物:

- `docs/20260511_huang2022_paper_implementation_mapping.md`
- `docs/20260511_izumi2026_paper_implementation_mapping.md`

### Phase 2: fallback を残したまま不変条件テストを追加する

この段階では fallback を削除しない。まず現状でどの不変条件が破れるかを確認する。

追加予定テスト:

- `tests/test_huang2022_paper_invariants.py`
- `tests/test_izumi2026_paper_invariants.py`

テストはまず `xfail` または fail expected として追加し、どの実装差分を直すべきかを明確化する。

### Phase 3: VMC / VNP / ParallelVMC / ParallelVNP を修正する

目的:

- `j_list` 正規化なしで `j_list` が `1..M` の permutation になる。
- `M_hat` 真値補正なしで `M_hat == M` になる。
- `j=1` が自然に存在する。

ここで見るべき原文:

- Huang 2022 `main.tex`: `VirtualMusicalChairs`, `VirtualNumberPlayers`
- Izumi 2026 `data.tex`: `ParallelVirtualMusicalChairs`, `ParallelVirtualNumberPlayers`
- Izumi 2026 `appendix.tex`: Sequential 版の対応箇所と説明表

進捗:

- 2026-05-11: `ParallelVirtualMusicalChairs` の rank 確定時更新を原文通り即時反映に修正した。
  - 原文: `s <- ell_i; ell_j <- s for all j`
  - 旧実装: 次ブロックまで `ell_j <- s` を遅延していた。
  - 影響: 同一ブロック内で同じ virtual rank に複数 player が座ることを防げるようになった。
- 2026-05-11: fallback なしの初期化不変条件テストを追加した。
  - Huang VNP: `M_hat == M`, `j_list` が `1..M` の permutation。
  - Izumi ParallelVNP: `M_hat == M`, `j_list` が `1..M` の permutation。
- 2026-05-11: `run()` から `j_list` 正規化と `M_hat` 真値補正を削除した。
- 2026-05-11: Izumi HDE の Grand Leader 不在 fallback を削除し、`j==1` がない場合はエラーに戻した。

### Phase 4: Assign / active set 更新を修正する

目的:

- `C_accept` 補完割当なしで、ComLeader / ComFollow / AssignAndUpdate の論文条件だけで割当する。
- `M' = M' - |C_accept|` をそのまま使っても未割当 player 数と矛盾しない。

ここで見るべき原文:

- Huang 2022 `main.tex`: `ComLeader`, `ComFollow`
- Izumi 2026 `appendix.tex`: `ComGrandLeader`, `ComSubLeader`, `ComFollower`

### Phase 5: fallback 削除

Phase 3〜4 の不変条件テストが通ってから、以下を削除する。

- `_normalize_internal_ranks`
- `M_hat_list = [max(m_hat, M) ...]`
- Grand Leader 不在 fallback
- `C_accept` 補完割当 fallback（未削除）
- `_try_assign` の accepted good arms fallback（未削除）。ただし、Izumi 2026 の論文意図として good arms を割当対象から除外するのか、top-M に含まれる可能性をどう扱うのかは原文確認後に決める。

### Phase 6: forced-collision communication

Huang 2022 から先に実装する。

1. `EncoderSendFloat`
2. `DecoderReceiveFloat`
3. accepted / rejected set の int 送受信
4. communication phase の action trace 検証

その後、Izumi 2026 の階層通信へ拡張する。

## 完了条件

論文再現を名乗るための最低条件:

- 上記 fallback が削除されている。
- paper invariant tests が通常の pytest として通る。
- forced-collision bit 通信が trace 上で再現されている。
- `K, M, n, seed` を振った比較実験で、成功率と regret が CSV/figure として再現できる。
- README / docs に simplified 実装ではなく paper-faithful 実装としての対象範囲と未実装範囲が明記されている。

## 注意

Izumi 2026 は `data.tex` 本文で一部通信プロトコルが省略されているため、Huang 2022 と `appendix.tex` から補完する必要がある。補完した仕様は、実装前に必ず docs に明記する。
