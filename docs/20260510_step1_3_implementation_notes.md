# Step 1〜3 実装メモ

作成日: 2026-05-10

このドキュメントは `docs/20260510_implementation_and_experiments.md` の Step 1〜3 の実装内容をまとめたものです。

---

## 実装したコードの説明

### Step 1: BernoulliMPMABEnv（環境）

#### `envs/bernoulli_mpmab.py`

- `BernoulliMPMABEnv`: Bernoulli 報酬を持つ MPMAB 環境。
  - `step(actions)`: M 人の action を同時に受け取り、collision 判定と報酬生成を行う。
    - 同一 arm を 2 人以上が選んだ場合（collision）: 全員の報酬 = 0。
    - collision なし: `Bernoulli(means[k])` の報酬。
  - `reset(seed)`: 乱数シードをリセットして再現可能な状態に戻す。
  - `optimal_total_reward`: top-M arm の平均報酬和（homogeneous 設定での最適値）。初期化時に計算。
- `StepResult`: 1 ステップの結果を保持するデータクラス。
  - `actions`, `rewards`, `collisions` は全プレイヤー分のリスト。
  - `collisions` は環境と評価指標での記録専用。no-sensing アルゴリズムには渡さない。

#### `envs/__init__.py`

`BernoulliMPMABEnv` と `StepResult` をエクスポート。

---

### Step 2: Runner と Trace

#### `core/trace.py`

- `StepRecord`: 1 ステップの記録（time, phase, actions, rewards, collisions, 累積報酬/regret）。
- `Trace`: シミュレーション全体の履歴を保持するクラス。
  - `set_phase(phase)`: 現在のフェーズ名を設定。
  - `append_step(...)`: 1 ステップの記録を追加し、累積報酬/regretを更新。
  - `phase_durations`: フェーズごとの所要ステップ数の dict。
  - `to_records()`: 全記録を dict のリストで返す。
  - `to_dataframe()`: pandas DataFrame で返す。

#### `core/runner.py`

- `HorizonReached`: horizon 到達を通知する例外。
- `Runner`: 環境との同期インターフェースを担うクラス。
  - `step(actions, phase)`: 1 ステップ実行。horizon に達すると `HorizonReached` を送出。
  - `set_phase(phase)`: Trace のフェーズ名を更新。
  - `elapsed`, `remaining`: 経過・残りステップ数のプロパティ。

#### `core/__init__.py`

`Trace`, `Runner` をエクスポート。

---

### Step 3: HomogeneousHuang2022

#### `algorithms/homogeneous_huang2022.py`

論文 Huang et al. (2022) "Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information" の実装。

- `PlayerState`: 各プレイヤーの初期化フェーズ終了時の状態サマリー（dataclass）。

- `HomogeneousHuang2022(K, M, delta, seed)`: アルゴリズム本体クラス。

  - `find_good_arm(runner)` → `(k_tilde, mu_tilde)`:
    - Algorithm 1 FindGoodArm を M 人同期実行。
    - Phase p ごとに sub-phase 1（一様ランダム探索）と sub-phase 2（確認フェーズ）を繰り返す。
    - 全プレイヤーが同じ good arm を得るまで繰り返す。

  - `virtual_musical_chairs(runner, k_tilde, tau)` → `s_list`:
    - Algorithm 2 VirtualMusicalChairs を M 人同期実行。
    - good arm k_tilde を K 個の仮想スロットに時間分割し、各プレイヤーに外部 rank s を割り当てる。
    - 報酬 > 0 = 自分だけが k_tilde を引けた（collision なし）→ rank 確定。

  - `virtual_number_players(runner, k_tilde, s_list, tau)` → `(M_hat_list, j_list)`:
    - Algorithm 3 VirtualNumberPlayers を M 人同期実行。
    - 仮想時刻 n=1..2K を sequential hopping しながら、τ 回 k_tilde を引いて報酬合計 = 0 を collision と判断。
    - M_hat（推定プレイヤー数）と j（内部 rank, 1-based）を各プレイヤーが独立に推定する。

  - `distributed_exploration(runner, k_tilde, j_list, M_hat_list, tau)` → `f_list`:
    - Algorithm 4 DistributedExploration を実行。
    - Leader (j=1) が推定値を集約し、accept/reject を決定する。
    - accept が出た時点で即座に割当を確定して終了する。

  - `run(runner)` → `{"player_states": ..., "phase_durations": ...}`:
    - Algorithm 5 Proposed algorithm のエントリーポイント。
    - 各フェーズを段階的に実行し、HorizonReached 時はそれ以前の取得済み状態を保持する。

- `_compute_accept_reject(...)` → `(C_accept, C_reject)`:
  - ρ[k] と B[k] を計算して accept/reject 集合を返す。
  - 論文 Algorithm 6 の集約推定と accept/reject 判定に対応。

#### `algorithms/__init__.py`

`HomogeneousHuang2022` をエクスポート。

#### `algorithms/base.py`

`BaseAlgorithm` 抽象基底クラス（`run()` メソッドを定義）。

---

### その他

#### `utils/metrics.py`

- `compute_metrics(trace, player_states, means, M)`:
  - Trace と player_states から各種評価指標を計算。
  - cumulative_regret, collision_count, phase_durations などを返す。
  - `player_states` がある場合: rank_assignment_success, player_count_success, final_assignment_success も計算。

#### `main.py`

小規模 sanity check のエントリーポイント。`K=5, M=2, T=50000` で Huang 2022 を実行して結果を表示。

#### `requirements.txt`

依存パッケージ: `numpy>=1.24`, `pandas>=2.0`, `matplotlib>=3.7`, `pytest>=7.4`

---

## ソースパスの対応

| 実装内容 | ソースパス |
|---------|-----------|
| 環境（MPMAB env） | `envs/bernoulli_mpmab.py` |
| 環境のエクスポート | `envs/__init__.py` |
| 履歴記録 | `core/trace.py` |
| 実行ランナー | `core/runner.py` |
| アルゴリズム基底クラス | `algorithms/base.py` |
| Huang 2022 | `algorithms/homogeneous_huang2022.py` |
| 評価指標 | `utils/metrics.py` |
| 動作確認スクリプト | `main.py` |
| テスト: 環境 | `tests/test_env.py` |
| テスト: Huang 2022 初期化 | `tests/test_huang2022_initialization.py` |

---

## 簡略化した箇所

### DistributedExploration の通信（最大の簡略化）

**簡略化内容:**  
論文の forced collision bit 伝送（Algorithm 8/9: EncoderSendFloat / DecoderReceiveFloat）は完全には再現していない。

**実際の実装:**  
- 推定値 E[k] は直接集約する（量子化・通信誤りなし）。
- 通信時間コストとして `Q * tau * Ka ステップ` を Trace に記録するため、全プレイヤーが同じダミー腕を選ぶ `runner.step()` を消費する。
- `_consume_comm_steps()` が担当。

**影響:**  
- regret には通信コストが正確に反映される。
- 通信誤りによる誤決定はシミュレートされない。
- 実際より少ない phase で accept/reject が確定する可能性がある。

**実装箇所:**  
`algorithms/homogeneous_huang2022.py` の `distributed_exploration()` コメント「通信の簡略実装」参照。

### FindGoodArm の T1 式

**簡略化内容:**  
TeX の擬似コードは `6K * 2^p * ln(2/delta)` と表記されているが、実装は `6K^2 * 2^p * ceil(ln(2/delta))` を使用している。  
これは論文の参照実装（`papers/.../huang2022.py`）に合わせた。  
理由: 一様ランダム探索で腕あたりのサンプル数を確保するには K^2 が必要（腕 k への期待サンプル数 = T1/K = 6K * 2^p * ...）。

**実装箇所:** `find_good_arm()` の T1 計算箇所コメント参照。

### VirtualMusicalChairs の未確定フォールバック

**簡略化内容:**  
tau_rank の時間内に rank を得られなかったプレイヤーは s = -1 のまま残る。  
`run()` では `s if s >= 0 else 0` でフォールバックする。

**影響:**  
十分な tau（means が大きく delta が緩い場合）では発生しにくいが、厳しい設定では j/M_hat の推定精度が下がる可能性がある。

### DistributedExploration の leader 割当ロジック

**簡略化内容:**  
`f[leader_pid]` の決定に `len(C0_accept) >= M0` 条件を追加している。  
論文の「C = C' かどうかで leader が k_tilde を受け取るか分岐する」ロジックを C_accept に k_tilde が含まれているかどうかで代替している。

---

## 未実装の残タスク

Step 4〜6 に相当する部分:

1. **`HomogeneousMultiChannelIzumi2026`** (Step 4)
   - `find_multiple_good_arms`
   - `parallel_virtual_musical_chairs`
   - `parallel_virtual_number_players`
   - `hierarchical_distributed_exploration`

2. **`utils/plotter.py`** (Step 5 の一部)
   - `compare_homogeneous.py` の可視化に必要

3. **`experiments/compare_homogeneous.py`** (Step 6)
   - Huang 2022 vs Izumi 2026 の比較実験スクリプト
   - CSV 出力（`results/` 配下）
   - 図の出力（`figures/` 配下）

4. **通信の完全実装** (将来タスク)
   - `algorithms/communication.py` に forced collision bit 伝送を実装
   - `DistributedExploration` の `_consume_comm_steps` を本実装に置き換え

5. **テスト追加** (Step 6)
   - Izumi 2026 の同時複数 pull が発生しないことのテスト
   - `n=1` の multi-channel が single-channel と大きく矛盾しないことのテスト

---

## 実行したテストと結果

### テスト実行コマンド

```bash
python -m pytest tests/test_env.py tests/test_huang2022_initialization.py -v
```

### 結果

```
============================= test session starts ==============================
collected 18 items

tests/test_env.py::TestCollision::test_collision_gives_zero_reward PASSED
tests/test_env.py::TestCollision::test_collision_three_players PASSED
tests/test_env.py::TestCollision::test_partial_collision PASSED
tests/test_env.py::TestCollision::test_no_collision PASSED
tests/test_env.py::TestReproducibility::test_same_seed_same_results PASSED
tests/test_env.py::TestReproducibility::test_reset_reproducibility PASSED
tests/test_env.py::TestReproducibility::test_different_seeds_different_results PASSED
tests/test_env.py::TestOptimalReward::test_optimal_is_top_m_sum PASSED
tests/test_env.py::TestOptimalReward::test_instant_regret_nonneg_on_average PASSED
tests/test_env.py::TestValidation::test_invalid_arm_index PASSED
tests/test_env.py::TestValidation::test_invalid_actions_length PASSED
tests/test_huang2022_initialization.py::TestFindGoodArm::test_returns_positive_mean_arm PASSED
tests/test_huang2022_initialization.py::TestFindGoodArm::test_mu_tilde_is_lower_bound PASSED
tests/test_huang2022_initialization.py::TestVirtualMusicalChairs::test_rank_assignment PASSED
tests/test_huang2022_initialization.py::TestVirtualMusicalChairs::test_ranks_tend_to_be_unique PASSED
tests/test_huang2022_initialization.py::TestFullRun::test_assigned_arms_no_duplicate PASSED
tests/test_huang2022_initialization.py::TestFullRun::test_all_players_get_assignment PASSED
tests/test_huang2022_initialization.py::TestFullRun::test_phase_durations_recorded PASSED

18 passed in 0.19s
==============================
```

---

## テスト設定

| 設定 | 値 |
|------|----|
| K | 5 |
| M | 2 |
| means | `[0.9, 0.8, 0.1, 0.05, 0.02]` |
| top-M arms | arm 0, arm 1 |
| delta | 0.1（テスト高速化のため緩め） |
| horizon | 500,000 |
| seed | 0〜42 のいくつかの値で固定 |

確率的アルゴリズムのため、`test_ranks_tend_to_be_unique` は 10 回試行中 5 回以上の rank 重複なし達成を条件としている。
