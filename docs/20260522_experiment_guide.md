# MPMAB 実験ガイド

作成日: 2026-05-22  
更新日: 2026-06-26（heterogeneous `large` プリセット追加・ガイド最新化）

このドキュメントは、MPMAB シミュレーターの実験コマンドとパラメータ設定をまとめたものです。
コードを読まずに実験を実行できるリファレンスとして使ってください。

---

## 目次

1. [前提条件](#前提条件)
2. [動作確認](#動作確認)
3. [Homogeneous 比較実験（compare_homogeneous.py）](#homogeneous-比較実験)
4. [Heterogeneous 比較実験（compare_heterogeneous.py）](#heterogeneous-比較実験)
5. [テスト実行](#テスト実行)
6. [参考: n の有効範囲](#参考-n-の有効範囲)
7. [参考: delta の計算](#参考-delta-の計算)

---

## 前提条件

```bash
pip install -r requirements.txt
```

依存パッケージ: `numpy>=1.24`, `pandas>=2.0`, `matplotlib>=3.7`, `pytest>=7.4`

---

## 動作確認

比較実験の前に、アルゴリズムが正常に動くかを確認する。

```bash
python -m simulator.main
```

内部設定: `K=5, M=2, T=50000, means=[0.9, 0.8, 0.5, 0.2, 0.1], seed=42`

Huang 2022 と Izumi 2026（n=2）の両方を実行し、各プレイヤーの状態と metrics をターミナルに表示する。

出力例:

```
=== Huang 2022 Sanity Check ===
  player 0: s=1, j=0, M_hat=2, k_tilde=0, assigned=0
  player 1: s=2, j=1, M_hat=2, k_tilde=0, assigned=1
  ...
  cumulative_regret: 12345.67
  final_assignment_success: True
```

---

## Homogeneous 比較実験

Huang 2022 と Izumi 2026 を同一条件で複数 trial 比較し、CSV と PNG を出力する。

### 基本コマンド

```bash
# リポジトリルートから実行する
python -m simulator.experiments.compare_homogeneous --experiment <name> [options]
```

### 実験プリセット

`--experiment` に以下の 3 つから指定する。

#### `small` — 動作確認（sanity check）

```
K=5, M=2, T=50000
means=[0.9, 0.8, 0.5, 0.2, 0.1]
n_values=[1, 2]
trials=20, seed_base=20260510
delta = 1 / (T * ln(T))
```

目的: 実装が破綻していないことを確認する。

```bash
python -m simulator.experiments.compare_homogeneous --experiment small
python -m simulator.experiments.compare_homogeneous --experiment small --trials 20
```

#### `speedup` — Multi-channel 短縮効果の検証

```
K=10, M=5, T=50000
means=[0.9, 0.85, 0.8, 0.75, 0.7, 0.45, 0.35, 0.25, 0.15, 0.1]
n_values=[1, 2, 3]
trials=50, seed_base=20260510
```

目的: `n` を増やすことで初期化フェーズの所要時間が短縮されるか確認する。

```bash
python -m simulator.experiments.compare_homogeneous --experiment speedup
python -m simulator.experiments.compare_homogeneous --experiment speedup --trials 50
```

#### `tradeoff` — good arm 探索コストとの tradeoff

```
K=20, M=5, T=100000
means=[0.95, 0.9, 0.86, 0.82, 0.78, 0.7, 0.64, 0.58, 0.52, 0.46,
       0.4, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06, 0.03]
n_values=[1, 2, 4, 6]
trials=100, seed_base=20260510
```

目的: `n` を大きくしすぎると `FindMultipleGoodArms` のコストが増えることを確認する。

```bash
python -m simulator.experiments.compare_homogeneous --experiment tradeoff
python -m simulator.experiments.compare_homogeneous --experiment tradeoff --trials 100
```

---

### CLI オプション一覧

| オプション | デフォルト | 内容 |
|---|---|---|
| `--experiment` | `small` | 実験プリセット名。`small` / `speedup` / `tradeoff` から選ぶ。 |
| `--trials N` | プリセット依存 | trial 数。指定するとプリセットの値を上書きする。 |
| `--horizon T` | プリセット依存 | horizon T。指定するとプリセットの値を上書きする。 |
| `--K K` | プリセット依存 | arm 数 K を上書きする。 |
| `--M M` | プリセット依存 | player 数 M を上書きする。 |
| `--n-values 1,2,3` | プリセット依存 | Izumi 2026 の n 値をカンマ区切りで指定する。 |
| `--means 0.9,0.8,...` | プリセット依存 | arm 平均報酬を直接指定する。長さは K と一致させること。 |
| `--mean-high F` | `0.9` | `--means` 未指定で `--K` を変えたとき、最大平均報酬として使う。 |
| `--mean-low F` | `0.1` | `--means` 未指定で `--K` を変えたとき、最小平均報酬として使う。 |
| `--sample-points N` | `300` | regret curve CSV/PNG 用のサンプル点数。 |
| `--ci F` | `0.95` | regret curve に描く信頼区間（現在 0.95 を想定）。 |
| `--no-plots` | なし | CSV のみ出力し PNG を生成しない。 |
| `--no-phase-lines` | なし | regret curve に phase 終了時刻の縦線を描かない。 |
| `--log-xscale` | なし | regret curve の x 軸を対数スケールにする。 |
| `--retry-on-failure` | なし | `final_assignment_success=0` の trial を seed を変えて再試行する。 |
| `--max-attempts N` | `3` | `--retry-on-failure` 時の最大試行回数。 |
| `--name-suffix STR` | `K{K}_M{M}_T{T}` | 出力ディレクトリ名の suffix。実験を区別するために使う。 |
| `--output-root PATH` | `outputs/runs` | 出力ディレクトリの親パス。 |

---

### 使用例

```bash
# 最小確認（デフォルト試行数 20 trials）
python -m simulator.experiments.compare_homogeneous --experiment small

# trial 数を増やす
python -m simulator.experiments.compare_homogeneous --experiment small --trials 50

# horizon を短縮してデバッグ
python -m simulator.experiments.compare_homogeneous --experiment small --trials 5 --horizon 20000

# K と means を直接指定
python -m simulator.experiments.compare_homogeneous --experiment small \
    --K 6 --M 2 --means 0.9,0.85,0.7,0.5,0.3,0.1

# K を変えて means を自動生成（mean-high と mean-low から線形補間）
python -m simulator.experiments.compare_homogeneous --experiment small \
    --K 8 --M 3 --mean-high 0.95 --mean-low 0.05

# n の候補を増やして speedup を比較
python -m simulator.experiments.compare_homogeneous --experiment speedup \
    --n-values 1,2,4

# CSV のみ出力（PNG なし）
python -m simulator.experiments.compare_homogeneous --experiment small --no-plots

# phase 縦線なしの regret curve
python -m simulator.experiments.compare_homogeneous --experiment small --no-phase-lines

# 割当失敗 trial を最大 5 回まで再試行
python -m simulator.experiments.compare_homogeneous --experiment small \
    --retry-on-failure --max-attempts 5

# 出力先を変える
python -m simulator.experiments.compare_homogeneous --experiment small \
    --output-root /tmp/my_runs

# 出力ディレクトリの suffix を指定（同プリセットを複数回比較するとき）
python -m simulator.experiments.compare_homogeneous --experiment small \
    --K 8 --M 3 --name-suffix mytest
```

---

### 出力ファイル構成

実験ごとに `outputs/runs/YYYYMMDD_HHMMSS_<name>/` が作られる。

```
outputs/runs/20260522_120000_homogeneous_small_sanity_K5_M2_T50000/
  summary.csv               # trial ごとの集計結果
  curves.csv                # regret curve 用の時系列データ
  run_config.json           # 実験設定の記録
  regret_huang_vs_izumi.png # cumulative regret の比較グラフ
  init_duration_by_n.png    # n 別の初期化所要時間
  collision_count_by_n.png  # n 別の collision 数
  success_rate_by_n.png     # n 別の top-M 割当成功率
```

`outputs/` は `.gitignore` 済みのため、リポジトリには含まれない。

---

### 出力 CSV の列

#### summary.csv

| 列 | 内容 |
|---|---|
| `algorithm` | `huang2022` または `izumi2026` |
| `K` | arm 数 |
| `M` | player 数 |
| `T` | horizon |
| `delta` | 信頼パラメータ（`1 / (T * ln(T))` で計算） |
| `n` | Izumi 2026 の good arm チャネル数（Huang 2022 は常に 1） |
| `trial` | trial 番号（0-indexed） |
| `seed` | 乱数 seed |
| `cumulative_regret` | 累積 regret（observed reward ベース） |
| `total_reward` | 累積報酬和 |
| `total_steps` | 使用した実ステップ数 |
| `init_duration` | 初期化フェーズ全体のステップ数 |
| `find_good_duration` | FindGoodArm / FindMultipleGoodArms のステップ数 |
| `rank_duration` | VirtualMusicalChairs / ParallelVMC のステップ数 |
| `number_players_duration` | VirtualNumberPlayers / ParallelVNP のステップ数 |
| `exploration_duration` | DistributedExploration / HDE のステップ数 |
| `collision_count` | 全ステップを通じた collision 総数 |
| `final_assignment_success` | top-M arm への重複なし割当成功 (0/1) |
| `player_count_success` | `M_hat == M` 成功 (0/1) |
| `rank_assignment_success` | rank に重複なし (0/1) |
| `assignment_duplicate` | 最終割当に重複あり (0/1) |
| `attempt` | 再試行の attempt 番号（`--retry-on-failure` 使用時） |
| `attempts_used` | 実際に試行した attempt 数 |
| `retry_exhausted` | 全 attempt が失敗したとき 1 |

#### curves.csv

| 列 | 内容 |
|---|---|
| `algorithm` / `K` / `M` / `T` / `delta` / `n` / `trial` / `seed` | summary と同じ |
| `time` | サンプル時刻（0 〜 T の等間隔グリッド） |
| `cumulative_regret` | その時刻での累積 regret |
| `phase` | その時刻でのフェーズ名 |
| `attempt` / `attempts_used` | summary と同じ |

---

## Heterogeneous 比較実験

Shi 2021 BEACON と Izumi 2026 ParallelBEACON を同一条件で複数 trial 比較し、CSV と PNG を出力する。

### 基本コマンド

```bash
python -m simulator.experiments.compare_heterogeneous --experiment <name> [options]
```

### 実験プリセット

`--experiment` に以下の 3 つから指定する。

#### `small` — 動作確認（sanity check）

```
K=5, M=2, T=500000
means_matrix = [
    [0.9, 0.7, 0.5, 0.3, 0.1],  # player 0
    [0.1, 0.3, 0.5, 0.7, 0.9],  # player 1
]
n_values=[2]
trials=10, seed_base=20260510
delta = 1 / (T * ln(T))
```

最適割当: player 0 → arm 0 (0.9), player 1 → arm 4 (0.9)

```bash
python -m simulator.experiments.compare_heterogeneous --experiment small
python -m simulator.experiments.compare_heterogeneous --experiment small --trials 10
```

#### `asym` — 非対称な報酬行列

```
K=6, M=3, T=1000000
means_matrix = [
    [0.9, 0.6, 0.4, 0.3, 0.2, 0.1],  # player 0: arm 0 が最適
    [0.2, 0.8, 0.5, 0.3, 0.1, 0.1],  # player 1: arm 1 が最適
    [0.1, 0.2, 0.3, 0.7, 0.4, 0.1],  # player 2: arm 3 が最適
]
n_values=[2, 3]
trials=20, seed_base=20260510
```

最適割当（Hungarian 法）: player 0 → arm 0, player 1 → arm 1, player 2 → arm 3

```bash
python -m simulator.experiments.compare_heterogeneous --experiment asym
python -m simulator.experiments.compare_heterogeneous --experiment asym --trials 20
```

#### `large` — 大規模・巡回報酬行列

```
K=20, M=8, T=1000000
means_matrix: homogeneous tradeoff の means 値（0.95〜0.03）を
              シフト=2 の巡回行列で配置
n_values=[1, 2, 3, 4, 5, 6]
trials=20, seed_base=20260510
```

最適割当: p0→arm0, p1→arm18, p2→arm16, p3→arm14, p4→arm12, p5→arm10, p6→arm8, p7→arm6（全員 0.95）

```bash
python -m simulator.experiments.compare_heterogeneous --experiment large
python -m simulator.experiments.compare_heterogeneous --experiment large --no-beacon
```

---

### CLI オプション一覧

| オプション | デフォルト | 内容 |
|---|---|---|
| `--experiment` | `small` | 実験プリセット名。`small` / `asym` / `large` から選ぶ。 |
| `--trials N` | プリセット依存 | trial 数。 |
| `--horizon T` | プリセット依存 | horizon T。 |
| `--K K` | プリセット依存 | arm 数 K を上書き。`--means-matrix` も必須。 |
| `--M M` | プリセット依存 | player 数 M を上書き。`--means-matrix` も必須。 |
| `--n-values 1,2` | プリセット依存 | ParallelBEACON の n 値をカンマ区切りで指定。 |
| `--means-matrix JSON` | プリセット依存 | 報酬行列を JSON で指定。例: `'[[0.9,0.1],[0.1,0.9]]'` |
| `--sample-points N` | `300` | regret curve CSV/PNG 用のサンプル点数。 |
| `--no-plots` | なし | CSV のみ出力し PNG を生成しない。 |
| `--no-phase-lines` | なし | regret curve に phase 終了時刻の縦線を描かない。 |
| `--log-xscale` | なし | regret curve の x 軸を対数スケールにする。 |
| `--retry-on-failure` | なし | `final_assignment_success=0` の trial を seed を変えて再試行する。 |
| `--max-attempts N` | `3` | `--retry-on-failure` 時の最大試行回数。 |
| `--name-suffix STR` | `K{K}_M{M}_T{T}` | 出力ディレクトリ名の suffix。 |
| `--output-root PATH` | `outputs/runs` | 出力ディレクトリの親パス。 |
| `--no-beacon` | なし | Shi 2021 BEACON を省略し ParallelBEACON のみ実行する。 |

---

### 使用例

```bash
# 最小確認（ParallelBEACON のみ）
python -m simulator.experiments.compare_heterogeneous --experiment small --no-beacon

# BEACON と ParallelBEACON を比較
python -m simulator.experiments.compare_heterogeneous --experiment small --trials 10

# horizon を短縮してデバッグ
python -m simulator.experiments.compare_heterogeneous \
    --experiment small --trials 2 --horizon 50000 --no-plots

# 報酬行列を直接指定（--K / --M 変更時は --means-matrix が必須）
python -m simulator.experiments.compare_heterogeneous \
    --experiment small --K 4 --M 2 --n-values 1 \
    --means-matrix '[[0.9,0.1,0.5,0.3],[0.1,0.9,0.3,0.5]]'

# large プリセットを n=1〜4 に絞って実行
python -m simulator.experiments.compare_heterogeneous \
    --experiment large --n-values 1,2,3,4 --no-beacon

# CSV のみ出力・再試行あり
python -m simulator.experiments.compare_heterogeneous \
    --experiment asym --trials 20 --no-plots \
    --retry-on-failure --max-attempts 5
```

---

### 出力ファイル構成

```
outputs/runs/20260611_120000_hetero_small_sanity_K5_M2_T500000/
  summary.csv                              # trial ごとの集計結果
  curves.csv                               # regret curve 用の時系列データ
  run_config.json                          # 実験設定の記録
  regret_beacon_vs_parallel_beacon.png     # cumulative regret の比較グラフ
  init_duration_by_algo.png                # アルゴリズム別の初期化所要時間
  collision_count_by_algo.png              # アルゴリズム別の collision 数
  success_rate_by_algo.png                 # アルゴリズム別の割当成功率
  comm_duration_by_algo.png                # アルゴリズム別の通信フェーズ所要時間
```

---

### 出力 CSV の列（Heterogeneous 版）

#### summary.csv

| 列 | 内容 |
|---|---|
| `algorithm` | `shi2021_beacon` または `izumi2026_parallel_beacon` |
| `K` | arm 数 |
| `M` | player 数 |
| `T` | horizon |
| `delta` | 信頼パラメータ（`1 / (T * ln(T))`） |
| `n` | ParallelBEACON の good arm チャネル数（BEACON は 1） |
| `trial` | trial 番号（0-indexed） |
| `seed` | 乱数 seed |
| `cumulative_regret` | 累積 regret（observed reward ベース） |
| `total_reward` | 累積報酬和 |
| `total_steps` | 使用した実ステップ数 |
| `init_duration` | 初期化フェーズ全体のステップ数 |
| `ortho_duration` | SelectCollisionSensingChannels のステップ数 |
| `rank_assignment_duration` | CollisionSensing ParallelVMC + ParallelVNP のステップ数 |
| `init_sample_duration` | ParallelBEACON Initial Sampling のステップ数 |
| `beacon_comm_duration` | エポック通信フェーズの累計ステップ数 |
| `beacon_explore_duration` | エポック探索フェーズの累計ステップ数 |
| `collision_count` | 全ステップを通じた collision 総数 |
| `optimal_matching_reward` | 最適マッチングの期待報酬和（Hungarian 法） |
| `final_assignment_success` | 最終割当が最適マッチングの arm 集合と一致 (0/1) |
| `player_count_success` | `M_hat == M` 成功 (0/1) |
| `rank_assignment_success` | rank に重複なし (0/1) |
| `assignment_duplicate` | 最終割当に重複あり (0/1) |
| `attempt` | 再試行の attempt 番号 |
| `attempts_used` | 実際に試行した attempt 数 |
| `retry_exhausted` | 全 attempt が失敗したとき 1 |

#### curves.csv

homogeneous 版と同じ列構成。`algorithm` の値が `shi2021_beacon` / `izumi2026_parallel_beacon` になる。

---

## テスト実行

```bash
# 全テスト（Homogeneous + Heterogeneous）
pytest

# Homogeneous アルゴリズムのみ
pytest tests/algorithms/homogeneous/huang2022/
pytest tests/algorithms/homogeneous/izumi2026/

# Heterogeneous アルゴリズムのみ
pytest tests/algorithms/heterogeneous/shi2021/
pytest tests/algorithms/heterogeneous/izumi2026/

# 環境テストのみ
pytest tests/envs/

# 詳細出力
pytest -v

# 失敗時に即停止
pytest -x
```

---

## 参考: n の有効範囲

### Homogeneous（Izumi 2026）

`1 <= n < K - M` が必要。

| K | M | 有効な n の最大値 |
|---|---|---|
| 5 | 2 | 2 |
| 10 | 5 | 4 |
| 20 | 5 | 14 |

### Heterogeneous（ParallelBEACON）

BEACON は `K >= M + 1`（broadcast arm が 1 本必要）、ParallelBEACON は `1 <= n < K - M` が必要。

| K | M | 有効な n の最大値 | プリセット |
|---|---|---|---|
| 5 | 2 | 2 | `small` |
| 6 | 3 | 2 | `asym` |
| 20 | 8 | 11 | `large` |

`--n-values` に範囲外の値を指定すると自動的に除外され、有効な値がなければエラーになる。

---

## 参考: delta の計算

全プリセットで `delta = 1 / (T * ln(T))` を使う。`--horizon` で T を変えると delta も連動して変わる。

| T | delta |
|---|---|
| 50,000 | `3.5e-6` |
| 100,000 | `1.7e-6` |
| 500,000 | `3.2e-7` |
| 1,000,000 | `1.4e-7` |
