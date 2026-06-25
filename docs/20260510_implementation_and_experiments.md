# MPMAB Simulator Implementation and Experiment Plan

作成日: 2026-05-10

このドキュメントは、`papers/README.md` の内容をもとに、MPMAB シミュレーターを Claude Code に実装してもらうための作業手順と、実装後に Codex がレビューしやすい比較実験手順をまとめたものです。

## 目的

まず以下の 2 モデルを実装し、同一条件で性能差を比較する。

1. Homogeneous MPMAB without Collision Sensing
   - Huang et al., 2022
   - `FindGoodArm`
   - `VirtualMusicalChairs`
   - `VirtualNumberPlayers`
   - `DistributedExploration`

2. Homogeneous Multi-Channel MPMAB
   - Izumi et al., 2026
   - `FindMultipleGoodArms`
   - `ParallelVirtualMusicalChairs`
   - `ParallelVirtualNumberPlayers`
   - `HierarchicalDistributedExploration`

その後、同じ基盤の上に heterogeneous 版と multi-channel heterogeneous 版を追加する。

## 擬似コード参照

Claude Code に PDF を読ませず、以下の Markdown を実装仕様として参照させる。

- `docs/pseudocode/20260510_huang2022_pseudocode.md`
- `docs/pseudocode/20260510_izumi2026_homogeneous_pseudocode.md`
- `docs/pseudocode/20260510_shi2021_beacon_pseudocode.md`
- `docs/pseudocode/20260510_wang2020_orthogonalization_pseudocode.md`

まず実装する 1 と 2 では、上のうち Huang 2022 と Izumi 2026 の 2 ファイルを優先して読む。

## 実装方針

論文の各プレイヤー用擬似コードを個別 Agent に直接閉じ込めるより、同期フェーズを持つ分散シミュレーション基盤として実装する。

理由は、MPMAB では各時刻に全プレイヤーの action を同時に集めて collision を判定する必要があるため、単一 Agent の `step()` だけでは collision と通信フェーズを自然に扱いにくいため。

## 推奨ディレクトリ構成

既存 README の構成を基本にしつつ、論文アルゴリズムは `algorithms/` に分離する。

```text
envs/
  __init__.py
  base_env.py
  bernoulli_mpmab.py

simulator/algorithms/
  __init__.py
  base.py
  homogeneous/
    huang2022/
      algorithm.py
      algorithm1_find_good_arm.py
      algorithm2_virtual_musical_chairs.py
      algorithm3_virtual_number_players.py
      algorithm4_distributed_exploration.py
    izumi2026/
      algorithm.py
      algorithm1_find_multiple_good_arms.py
      algorithm2_parallel_virtual_musical_chairs.py
      algorithm3_parallel_virtual_number_players.py
      algorithm4_hierarchical_distributed_exploration.py

core/
  __init__.py
  runner.py
  trace.py

simulator/utils/
  __init__.py
  metrics.py
  plotter.py

experiments/
  compare_homogeneous.py

tests/
  test_env.py
  test_huang2022_initialization.py
  test_izumi2026_initialization.py

simulator/main.py
requirements.txt
```

## 実装ステップ

### Step 1: 環境

`BernoulliMPMABEnv` を実装する。

必要な入力:

- `means`: shape `(K,)` の arm 平均報酬
- `num_players`: `M`
- `collision_sensing`: bool
- `seed`: 乱数 seed

必要な API:

```python
reset(seed: int | None = None) -> None
step(actions: list[int]) -> StepResult
```

`actions` は player ごとの arm index。index は Python 内部では 0-based に統一する。

`StepResult` に含めるもの:

- `actions`
- `rewards`
- `collisions`
- `total_reward`
- `optimal_total_reward`
- `instant_regret`

no-sensing アルゴリズムには `rewards` のみ渡す。`collisions` は記録と評価には使うが、意思決定には使わせない。

完了条件:

- 同じ arm を 2 人以上が選ぶと全員 reward 0。
- collision していない player は Bernoulli reward を受け取る。
- `optimal_total_reward` は homogeneous では top-M arm の平均報酬和。
- seed 固定時に結果が再現する。

### Step 2: Runner と Trace

`Runner` は全 player の action を同時に環境へ渡し、履歴を保存する。

最低限必要な履歴:

- time
- actions
- rewards
- collisions
- phase name
- cumulative reward
- cumulative regret

アルゴリズム側は `run(env, horizon, delta, seed)` のようにフェーズを進め、Runner または Trace に全時刻を記録する。

完了条件:

- どのフェーズでどれだけ時間を使ったか集計できる。
- horizon に達したら途中フェーズでも停止できる。
- 実験後に DataFrame または dict list として保存できる。

### Step 3: Huang 2022

`HomogeneousHuang2022` を実装する。

実装対象:

1. `find_good_arm`
2. `virtual_musical_chairs`
3. `virtual_number_players`
4. `distributed_exploration`
5. `run`

論文内の 1-based index は実装では 0-based に変換する。コードコメントでは論文の変数名も残す。

重要な点:

- no-sensing なので collision と自然な Bernoulli 0 を区別しない。
- `FindGoodArm` は全プレイヤーが同じ処理を同期して実行する。
- `VirtualMusicalChairs` は good arm 1 本を K 個の virtual arm に時間分割する。
- `VirtualNumberPlayers` は外部 rank `s` から内部 rank `j` と `M_hat` を推定する。
- `DistributedExploration` の通信サブルーチンは最初から完全再現を狙いすぎない。まずは比較可能な基準実装として、論文の accept/reject ロジックと通信時間コストを明示的にシミュレートする。

`tau`:

```python
tau_rank = ceil(K * log(1 / delta) / mu_tilde)
tau_comm = ceil(log(1 / delta) / mu_tilde)
```

完了条件:

- 各 player に外部 rank `s` が割り当てられる。
- 各 player が同じ `M_hat` を得る。
- 最終的に各 player に異なる arm が割り当てられる。
- 小さい問題設定で top-M arm を選ぶ成功率を測定できる。

### Step 4: Izumi 2026 Multi-Channel

`HomogeneousMultiChannelIzumi2026` を実装する。

実装対象:

1. `find_multiple_good_arms`
2. `parallel_virtual_musical_chairs`
3. `parallel_virtual_number_players`
4. `hierarchical_distributed_exploration`
5. `run`

重要な点:

- `n` 本の good arms を通信チャネルとして使う。
- 同一時刻に 1 player が複数 arm を引かないよう、位相ずらしの禁止集合を実装する。
- `ParallelVirtualMusicalChairs` は探索機会を最大 `n` 倍にする。
- `ParallelVirtualNumberPlayers` は仮想時間 `v` を `n` 個ずつ処理する。
- `HierarchicalDistributedExploration` は grand leader, sub leader, follower の構造にする。

`tau` は論文 TeX の括弧が曖昧なので、2022 との対応を優先して以下で実装する。

```python
mu_min = min(mu_tilde[k] for k in good_arms)
tau = ceil(log(1 / delta) / mu_min)
```

完了条件:

- `n=1` のとき、初期化フェーズの挙動が Huang 2022 に近くなる。
- `n>1` のとき、rank assignment と number players の所要時間が短くなる。
- 同一 player が同時刻に複数 arm を引く実装になっていない。
- 最終割り当てが重複しない。

### Step 5: Metrics

`simulator/simulator/utils/metrics.py` に以下を実装する。

- cumulative regret
- average regret over trials
- initialization duration
- rank assignment success rate
- player count estimation success rate
- final top-M assignment success rate
- collision count
- phase-wise time cost

homogeneous の regret は以下で計算する。

```python
instant_regret = sum(top_m_means) - sum(observed_rewards)
```

理論 regret と比較したい場合は observed rewards ではなく expected reward ベースの regret も別途出せるようにする。

### Step 6: Tests

最低限のテストを追加する。

- env の collision 処理
- seed 再現性
- `FindGoodArm` が正の平均報酬 arm を返す
- `VirtualMusicalChairs` が rank 重複を減らす
- `ParallelVirtualMusicalChairs` で同時複数 pull が発生しない
- `n=1` multi-channel が single-channel と大きく矛盾しない

確率的アルゴリズムなので、テストでは seed 固定、小さな `K, M`、十分差のある means を使う。

## 比較実験手順

### 実験 1: Small sanity check

目的: 実装が破綻していないことを確認する。

設定:

```python
K = 5
M = 2
T = 5000
means = [0.9, 0.8, 0.5, 0.2, 0.1]
delta = 1 / (T * log(T))
n_values = [1, 2]
trials = 20
```

確認:

- Huang 2022 と Izumi 2026 のどちらも top-M arm に収束するか。
- `M_hat == M` になるか。
- final assignment に重複がないか。
- cumulative regret が極端に発散していないか。

### 実験 2: Multi-channel speedup

目的: 2026 版で初期化と通信の時間短縮が見えるかを確認する。

設定:

```python
K = 10
M = 5
T = 50000
means = [0.9, 0.85, 0.8, 0.75, 0.7, 0.45, 0.35, 0.25, 0.15, 0.1]
delta = 1 / (T * log(T))
n_values = [1, 2, 3]
trials = 50
```

比較:

- cumulative regret
- initialization duration
- rank assignment duration
- number players duration
- collision count
- final top-M assignment success rate

期待:

- `n` が増えるほど rank assignment と number players の時間が短くなる。
- `FindMultipleGoodArms` は `n` に応じて重くなる。
- 全体では一定条件で 2026 版が有利になる。

### 実験 3: Good arm 探索コストの影響

目的: `FindMultipleGoodArms` の追加コストと後続フェーズ短縮の tradeoff を見る。

設定:

```python
K = 20
M = 5
T = 100000
means = sorted random means, descending, min positive
n_values = [1, 2, 4, 6]
trials = 100
```

記録:

- `find_good_arm` または `find_multiple_good_arms` の所要時間
- 初期化全体の所要時間
- cumulative regret
- success rate

期待:

- `n` を増やしすぎると good arm 探索コストが増える。
- 中程度の `n` で最も regret が低くなる可能性がある。

## 出力ファイル

実験結果は以下に保存する。

```text
results/
  homogeneous_small_sanity.csv
  homogeneous_multichannel_speedup.csv
  good_arm_tradeoff.csv

figures/
  regret_huang_vs_izumi.png
  init_duration_by_n.png
  collision_count_by_n.png
  success_rate_by_n.png
```

CSV には最低限以下の列を含める。

```text
algorithm, K, M, T, delta, n, trial, seed,
cumulative_regret, total_reward,
init_duration, find_good_duration, rank_duration,
number_players_duration, exploration_duration,
collision_count, final_assignment_success,
player_count_success
```

## Claude Code への実装依頼テンプレート

以下のように依頼するとよい。

```text
docs/20260510_implementation_and_experiments.md を読んで、この手順に沿って実装してください。

まずは Step 1 から Step 3 まで、つまり BernoulliMPMABEnv、Runner/Trace、HomogeneousHuang2022 を実装してください。
実装後に pytest が通る状態にしてください。

注意:
- Python 内部の arm/player index は 0-based に統一してください。
- 論文中の 1-based 変数はコメントで対応を書いてください。
- no-sensing アルゴリズムには collision flag を渡さないでください。
- 既存ファイルの不要なリファクタリングは避けてください。
```

Step 3 が通った後、次に以下を依頼する。

```text
次に Step 4 から Step 6 を実装してください。
HomogeneousMultiChannelIzumi2026、metrics、比較実験スクリプト、テストを追加してください。

特に ParallelVirtualMusicalChairs と ParallelVirtualNumberPlayers では、同一 player が同時刻に複数 arm を pull しないことをテストしてください。
```

## Codex レビュー観点

Claude Code 実装後、Codex は以下を重点的にレビューする。

1. 環境の collision 処理が正しいか。
2. no-sensing アルゴリズムが collision flag を参照していないか。
3. 同期フェーズで horizon を超えて走り続けないか。
4. 1-based と 0-based の変換ミスがないか。
5. `tau` の式が 2022 と 2026 で一貫しているか。
6. `n=1` の multi-channel 版が single-channel 版と大きく矛盾しないか。
7. regret 計算が observed reward ベースか expected reward ベースか明示されているか。
8. 確率的テストが不安定すぎないか。
9. 実験結果 CSV に再現に必要な seed と設定値が残っているか。
10. 論文ロジックを省略・近似した箇所がコメントや README に明記されているか。

## 先に割り切る点

初回実装では、論文の implicit communication を完全に物理時刻レベルで再現しすぎない。まずは通信に必要な時刻コストと情報伝達結果をシミュレートし、比較可能な regret と phase duration を出すことを優先する。

その後、必要に応じて `communication.py` を詳細化し、forced collision による bit transmission を時刻単位で再現する。

## 将来拡張

3 番目以降のモデルでは heterogeneous 報酬行列 `means[m, k]` が必要になる。

追加予定:

- `HeterogeneousMPMABEnv`
- Wang et al., 2020 の orthogonalization と rank assignment
- Shi et al., 2021 の BEACON leader/follower
- Izumi 2026 の heterogeneous multi-channel 版
- matching oracle
- CUCB ベースの batch exploration

heterogeneous 版では regret の最適値が top-M arm 和ではなく、player-arm matching の最適値になるため、metrics を拡張する必要がある。
