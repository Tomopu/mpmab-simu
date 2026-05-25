# Homogeneous MPMAB 実験ガイド

作成日: 2026-05-22

このドキュメントは、Homogeneous MPMAB シミュレーターの実験コマンドとパラメータ設定をまとめたものです。
コードを読まずに実験を実行できるリファレンスとして使ってください。

---

## 前提条件

```bash
pip install -r requirements.txt
```

依存パッケージ: `numpy>=1.24`, `pandas>=2.0`, `matplotlib>=3.7`, `pytest>=7.4`

---

## 動作確認（単発実行）

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

## 比較実験（compare_homogeneous.py）

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

## 出力ファイル構成

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

## 出力 CSV の列

### summary.csv

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

### curves.csv

| 列 | 内容 |
|---|---|
| `algorithm` / `K` / `M` / `T` / `delta` / `n` / `trial` / `seed` | summary と同じ |
| `time` | サンプル時刻（0 〜 T の等間隔グリッド） |
| `cumulative_regret` | その時刻での累積 regret |
| `phase` | その時刻でのフェーズ名 |
| `attempt` / `attempts_used` | summary と同じ |

---

## テスト実行

```bash
# 全テスト
pytest

# 特定のアルゴリズムのみ
pytest tests/algorithms/homogeneous/huang2022/
pytest tests/algorithms/homogeneous/izumi2026/

# 環境テストのみ
pytest tests/envs/

# 詳細出力
pytest -v

# 失敗時に即停止
pytest -x
```

---

## n の有効範囲

Izumi 2026 では `n` の制約として `1 <= n < K - M` が必要。

| K | M | 有効な n の最大値 |
|---|---|---|
| 5 | 2 | 2 |
| 10 | 5 | 4 |
| 20 | 5 | 14 |

`--n-values` に範囲外の値を指定すると自動的に除外され、有効な値がなければエラーになる。

---

## delta の計算

全プリセットで `delta = 1 / (T * ln(T))` を使う。`--horizon` で T を変えると delta も連動して変わる。

| T | delta |
|---|---|
| 5,000 | `3.7e-5` |
| 50,000 | `3.5e-6` |
| 100,000 | `1.7e-6` |
