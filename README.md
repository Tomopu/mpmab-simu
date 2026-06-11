# mpmab-simu

Multi-Player Multi-Armed Bandit Simulator

no-sensing 設定（collision を観測できない）の MPMAB アルゴリズムを同期シミュレーションで実装・比較するためのリポジトリ。

## 実装状況

### Homogeneous（実装済み）
- **Step 1〜3**: `BernoulliMPMABEnv`、`Runner`/`Trace`、`HomogeneousHuang2022`（Huang et al., 2022）
- **Step 4**: `HomogeneousMultiChannelIzumi2026`（Izumi et al., 2026）
- **Step 5〜6**: metrics、比較実験 CSV 出力、グラフ生成、pytest
- 注意: forced-collision bit 通信は通信結果の直接集約 + 通信時間コストの簡略実装。

### Heterogeneous（実装済み）
- **`HeterogeneousMPMABEnv`**: player-arm 報酬行列 `means[m][k]`、collision sensing、最大重み二部マッチング（Hungarian 法）による `optimal_total_reward` 計算。
- **`HeterogeneousRunner`**: `HeterogeneousMPMABEnv` 用の同期 Runner。collision flag を意思決定に使える。
- **`HeterogeneousShiBeacon2021`** (Model 3): Shi et al. (2021) BEACON の実装。
  - Wang et al. (2020) Orthogonalization + Rank Assignment（初期化）
  - BEACON Leader/Follower エポックループ（通信 + 探索）
  - forced collision ビット伝送プロトコル（Send/Receive）を 1 bit = 1 ステップで実際にシミュレート
  - 最大重み二部マッチング Oracle
- **`HeterogeneousMultiChannelIzumi2026`** (Model 4): Izumi et al. (2026) ParallelBEACON の実装。
  - 初期化フェーズ: FindMultipleGoodArms + ParallelVirtualMusicalChairs + ParallelVirtualNumberPlayers（Izumi 2026 homogeneous 版と共用）
  - 学習フェーズ: ParallelBEACON（grand leader / sub-leader / follower 階層グループ通信 + Oracle 割当 + 探索）
  - 通信コストは dummy ステップで消費する簡略実装
- **`compute_hetero_metrics`**: Heterogeneous 向け評価指標（最適マッチング比較、フェーズ別通信コスト等）
- **`compare_heterogeneous.py`**: BEACON vs ParallelBEACON 比較実験 CLI（CSV / PNG 出力）

## ディレクトリ構造

```
mpmab-simu/
├── simulator/
│   ├── envs/
│   │   ├── bernoulli_mpmab.py          # Homogeneous 環境
│   │   └── heterogeneous_mpmab.py      # Heterogeneous 環境（player-arm 報酬行列）
│   ├── core/
│   ├── algorithms/
│   │   ├── homogeneous/
│   │   │   ├── huang2022/              # Huang 2022（no-sensing）
│   │   │   ├── izumi2026/              # Izumi 2026（multi-channel, no-sensing）
│   │   │   ├── states.py
│   │   │   └── math_helpers.py
│   │   └── heterogeneous/
│   │       ├── shi2021/
│   │       │   ├── algorithm.py                     # BEACON クラス本体・run()
│   │       │   ├── runner_hetero.py                 # HeterogeneousRunner
│   │       │   ├── wang2020_orthogonalization.py    # Orthogonalization + Rank Assignment
│   │       │   ├── shi2021_beacon_communication.py  # Send/Receive ビット伝送
│   │       │   ├── shi2021_beacon_epoch.py          # エポックループ（通信+探索）
│   │       │   ├── oracle.py                        # Matching Oracle（Hungarian 法）
│   │       │   ├── states.py
│   │       │   └── results.py
│   │       └── izumi2026/
│   │           ├── algorithm.py                     # ParallelBEACON クラス本体・run()
│   │           ├── parallel_beacon_epoch.py         # ParallelBEACON エポックループ
│   │           ├── states.py
│   │           └── results.py
│   ├── utils/
│   ├── experiments/
│   └── main.py
├── tests/
│   ├── envs/
│   ├── algorithms/
│   │   ├── homogeneous/
│   │   └── heterogeneous/
│   │       ├── shi2021/                # 環境・初期化・エンドツーエンドテスト（BEACON）
│   │       └── izumi2026/              # ParallelBEACON エンドツーエンドテスト
│   └── invariants/
├── docs/
├── papers/
├── outputs/
└── requirements.txt
```

## クイックスタート

```bash
pip install -r requirements.txt

# sanity check (K=5, M=2, T=50000)
python -m simulator.main

# テスト（homogeneous + heterogeneous）
python -m pytest tests/ -v

# 比較実験（Homogeneous: CSV と PNG を生成）
python -m simulator.experiments.compare_homogeneous --experiment small --trials 20

# 比較実験（Heterogeneous: BEACON vs ParallelBEACON）
python -m simulator.experiments.compare_heterogeneous --experiment small --trials 10
```

### Heterogeneous ParallelBEACON の使い方（Izumi 2026, Model 4）

```python
from simulator.envs.heterogeneous_mpmab import HeterogeneousMPMABEnv
from simulator.algorithms.heterogeneous.izumi2026 import HeterogeneousMultiChannelIzumi2026
from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner
from simulator.utils.metrics import compute_hetero_metrics

means = [
    [0.9, 0.1, 0.5, 0.3, 0.2],  # player 0
    [0.1, 0.9, 0.3, 0.5, 0.2],  # player 1
]
K, M, n, T = 5, 2, 2, 500000

env = HeterogeneousMPMABEnv(means, collision_sensing=True, seed=0)
runner = HeterogeneousRunner(env, horizon=T)
algo = HeterogeneousMultiChannelIzumi2026(K=K, M=M, n=n, delta=1e-3, seed=0)
result = algo.run(runner)

metrics = compute_hetero_metrics(runner.trace, result["player_states"], means, M)
print(f"cumulative_regret={metrics['cumulative_regret']:.1f}")
for ps in result["player_states"]:
    print(f"j={ps.internal_rank_j}, arm={ps.assigned_arm}, group={ps.group}")
```

### Heterogeneous BEACON の使い方

```python
from simulator.envs.heterogeneous_mpmab import HeterogeneousMPMABEnv
from simulator.algorithms.heterogeneous.shi2021 import HeterogeneousShiBeacon2021, HeterogeneousRunner

# player-arm 報酬行列（means[m][k] = player m が arm k を引いたときの期待報酬）
means = [
    [0.9, 0.1, 0.5, 0.3],  # player 0
    [0.1, 0.9, 0.3, 0.5],  # player 1
]
K, M, T = 4, 2, 200000

env = HeterogeneousMPMABEnv(means, collision_sensing=True, seed=0)
runner = HeterogeneousRunner(env, horizon=T)
algo = HeterogeneousShiBeacon2021(K=K, M=M, seed=0)
result = algo.run(runner)

for ps in result["player_states"]:
    print(f"rank={ps.rank}, arm={ps.assigned_arm}, M_hat={ps.M_hat}")
```

## 設計方針

- arm index / player index は Python 内部で 0-based に統一。
- no-sensing: collision flag はアルゴリズムの意思決定に渡さない。
- 各ステップの記録は `Trace` に積み、`to_dataframe()` で pandas DataFrame として取得できる。
- 再現性のため、確率的処理はすべて seed を受け取る。
- 論文アルゴリズムは `algorithm.py` を入口にし、論文の Algorithm 番号に対応するファイルへ phase 実装を分ける。
- 比較実験は CLI、設定、実行、I/O を分け、実験追加時に `run_homogeneous.py` へ処理が集中しすぎないようにする。
