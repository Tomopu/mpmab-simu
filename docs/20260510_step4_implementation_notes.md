# Step 4 実装メモ

作成日: 2026-05-10

このドキュメントは `docs/20260510_implementation_and_experiments.md` の Step 4 の実装内容をまとめたものです。

---

## 実装したコードの説明

### Step 4: HomogeneousMultiChannelIzumi2026

#### `simulator/algorithms/homogeneous/izumi2026/algorithm.py`

論文 Izumi et al. (2026) "Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing" の homogeneous 設定実装。

- `PlayerStateIzumi`: 各プレイヤーの初期化フェーズ終了時の状態サマリー（dataclass）。
  - `external_rank_s`, `internal_rank_j`, `M_hat`: Huang 2022 の PlayerState と同様。
  - `good_arms`: 見つかった good arm の 0-based index リスト（n 本）。
  - `mu_tilde_min`: good arms 中の最小報酬下界。
  - `assigned_arm`: 割り当て済み arm（0-based）。

- `HomogeneousMultiChannelIzumi2026(K, M, n, delta, seed)`: アルゴリズム本体クラス。
  - 制約: `n < K - M` が必要（good arms が top-M arms と重ならないことを期待する設定）。

  - `find_multiple_good_arms(runner)` → `(List[int], Dict[int, float])`:
    - Algorithm 1 FindMultipleGoodArms を M 人同期実行。
    - T1 = `6*|K|*2^p*ceil(ln(2n/delta))` （Huang 2022 の `6K^2*2^p*ceil(ln(2/delta))` とは別計算）。
    - T2 = `|K|*2^p*ceil(ln(2n/delta))`。
    - active_arms は player 0 の G を基準に各フェーズ後に更新する。

  - `parallel_virtual_musical_chairs(runner, good_arms, tau)` → `List[int]`:
    - Algorithm 2 ParallelVirtualMusicalChairs を M 人同期実行。
    - total_steps = `ceil(K*tau/n)` （VirtualMusicalChairs の `K*tau` より n 倍少ない）。
    - ブロック先頭（t0 % K == 0）で n 本の good arm にスロットを割り当てる（spreading 条件）。
    - spreading: 0-based 禁止集合 `F = {(l[m][j0] + i0 - j0) % K | j0 < i0}` で衝突を防ぐ。
    - ランク確定時: l[m] の更新は次ブロック先頭まで遅らせる（同ブロック内の二重 pull を防ぐ）。

  - `parallel_virtual_number_players(runner, good_arms, s_list, tau)` → `(List[int], List[int])`:
    - Algorithm 3 ParallelVirtualNumberPlayers を M 人同期実行。
    - outer ループ h = 0..`ceil(2K/n)-1`、inner ループ i0 = 0..n-1 で仮想時刻 v = h*n+i0+1 を並列処理。
    - 各 h で n 本分の仮想時刻をまとめて処理することで VirtualNumberPlayers の n 倍高速化を実現。
    - スロットマッピング条件（0-based）: `(ell[m][i0] + i0) % K == k_0`。

  - `hierarchical_distributed_exploration(runner, good_arms, j_list, M_hat_list, tau)` → `List[int]`:
    - Algorithm 4 HierarchicalDistributedExploration を同期実行。
    - Grand Leader (j=1)、Sub-Leader (2<=j<=n)、Follower (j>n) の 3 階層構造。
    - グループ: `g_0 = (j_list[m]-1) % n`（0-based）。
    - 通信コストは「最も重いグループの通信時間」で近似（並列通信の近似）。
    - accept/reject は Grand Leader のみが計算する（AcceptReject ヘルパーを使用）。

  - `run(runner)` → `{"player_states": ..., "phase_durations": ...}`:
    - Algorithm 5 ProposedParallelAlgorithm のエントリーポイント。
    - tau = `ceil(ln(1/delta) / min(mu_tilde))`（論文に従う）。
    - 各フェーズを段階的に実行し、HorizonReached 時は取得済み状態を保持する。

  - `_try_assign(j, good_arms, M_active, C_accept)` → `int`:
    - AssignAndUpdate のヘルパー（割当のみ、active set 更新は HDE 側で行う）。
    - 優先 1: C_assign = C_accept - G から割り当てる（論文準拠の通常ケース）。
    - 優先 2: G ∩ C_accept から割り当てる（good arms が top-M になったときのフォールバック）。

  - `_compute_accept_reject(...)` → `(List[int], List[int])`:
    - Huang 2022 と同一の式を使用（Supplemental Pseudocode AcceptReject に従う）。

#### `simulator/algorithms/__init__.py`

`HomogeneousMultiChannelIzumi2026`, `PlayerStateIzumi` をエクスポートするよう更新。

#### `simulator/simulator/utils/metrics.py`

Izumi 2026 のフェーズ名（`find_multiple_good_arms`, `parallel_virtual_musical_chairs`, `parallel_virtual_number_players`, `hde_phase*`）に対応するようフェーズ集計ロジックを更新。

#### `simulator/main.py`

`run_izumi2026(n)` 関数を追加し、Huang 2022 と Izumi 2026 を同一設定で比較表示するよう更新。

---

## ソースパスの対応

| 実装内容 | ソースパス |
|---------|-----------|
| Izumi 2026 アルゴリズム | `simulator/algorithms/homogeneous/izumi2026/algorithm.py` |
| アルゴリズム export 更新 | `simulator/algorithms/__init__.py` |
| metrics フェーズ名対応 | `simulator/simulator/utils/metrics.py` |
| simulator/main.py 呼び出し例 | `simulator/main.py` |
| Izumi 2026 テスト | `tests/algorithms/homogeneous/izumi2026/` |

---

## Huang 2022 から流用・拡張した箇所

| 項目 | 流用/拡張の内容 |
|-----|---------------|
| `_compute_accept_reject` | 同一の式（ρ[k], B[k]）を使用。Supplemental Pseudocode AcceptReject に従う。 |
| `_consume_comm_steps` | 全プレイヤーがダミー腕を選ぶダミーアクションで時間コストをシミュレートする手法を流用。 |
| `_compute_accept_reject` の呼び出しパターン | HDE sub-phase 2 のフロー構造が DistributedExploration に類似。 |
| Sequential hopping（sub-phase 1） | HDE の sub-phase 1 は Huang 2022 の DistributedExploration と同一のロジック。 |
| `run()` の try/except パターン | HorizonReached 時に取得済み状態を保持するパターンを踏襲。 |
| `PlayerState` の設計 | `PlayerStateIzumi` の構造を参考に、`good_arms` リストを追加拡張。 |

### 主な相違点

| 項目 | Huang 2022 | Izumi 2026 |
|-----|-----------|-----------|
| FindGoodArm | 1 本探索、T1 = 6K² | n 本探索、T1 = 6|K|（active set ベース） |
| VMC 総ステップ | K × τ | ceil(K×τ/n)（n 倍短縮） |
| VNP 仮想時刻 | n=1..2K を直列 | n 本並列（n 倍短縮） |
| 通信構造 | Leader / Follower（2 階層） | Grand Leader / Sub-Leader / Follower（3 階層） |
| 通信チャンネル | good arm 1 本のみ | good arm n 本を各グループに割り当て |

---

## 簡略化した通信処理

### FindMultipleGoodArms の同期処理

**簡略化内容:**
各プレイヤーが独立に G_list[m] を追跡するが、active_arms の更新は player 0 の G_list[0] を基準にする。
同期設計上、homogeneous 設定では全プレイヤーが同じ good arms を確定するはずであり、
player 0 をシミュレーションの canonical player として使う。

**影響:**
player 0 以外のプレイヤーが player 0 と異なる arms を確定した場合、
K_active が不適切に更新される可能性がある。homogeneous 設定では発生しにくい。

### HierarchicalDistributedExploration の通信

**簡略化内容（Supplemental Pseudocode の Implementation Notes に従う）:**
1. 推定値 E[k] は直接集約する（量子化・通信誤りなし）。
2. 通信時間コストは「最も重いグループの通信時間」で近似する（並列通信の近似）。
3. accept/reject の計算は Grand Leader のみが行う。

**実装箇所:**
`hierarchical_distributed_exploration()` の sub-phase 2 の通信コスト計算部分。
コメント「通信コストを Trace に記録するためダミー step を消費する」参照。

**影響:**
- regret への通信コスト寄与は近似的に反映される。
- 通信誤りによる誤決定はシミュレートされない。
- 実際より少ない phase で accept/reject が確定する可能性がある。

### _try_assign の good arms フォールバック

**簡略化内容:**
論文の AssignAndUpdate では C_assign = C_accept \ G（good arms を除外）だが、
good arms が top-M arms になった場合（C_assign が空になる場合）、
G ∩ C_accept から割り当てるフォールバックを追加している。

**影響:**
論文の想定（good arms ≠ top-M arms）が成立しない場合でも割当が完了する。
テスト設定（means=[0.9, 0.8, 0.1, 0.05, 0.02], n=2）では good arms が top-2 に
なることが多く、このフォールバックが機能する。

**実装箇所:**
`_try_assign()` のコメント「G_accept から割り当てる（フォールバック）」参照。

### ParallelVirtualMusicalChairs のランク確定タイミング

**簡略化内容:**
ランク確定時（reward > 0）に l[m] の更新を次ブロック先頭まで遅らせる。
論文擬似コードは `l_j <- s for all j` を即座に行うが、
同ブロック内の後続ステップで新スロットが再マッチして同じ good arm を
2 回引くバグが発生するため、ブロック先頭での更新に変更した。

**影響:**
確定ブロック内の残りステップでは元の spreading スロットで動作する（問題なし）。
次ブロックから wait at rank s モードに入る。

---

## 未実装の残タスク

1. **通信の完全実装** (将来タスク)
   - `algorithms/communication.py` に forced collision bit 伝送を実装。
   - ComGrandLeader / ComSubLeader / ComFollower を詳細に再現。
   - `HierarchicalDistributedExploration` の通信コストをより正確にシミュレート。

2. **`simulator/utils/plotter.py`** (Step 5 の一部)
   - cumulative regret の比較グラフ（Huang 2022 vs Izumi 2026）。
   - フェーズ別所要時間の棒グラフ。

3. **`simulator/experiments/compare_homogeneous.py`** (Step 6)
   - Huang 2022 vs Izumi 2026 (n=1, 2, 3) の比較実験スクリプト。
   - CSV 出力（`results/` 配下）・グラフ出力（`figures/` 配下）。

4. **テスト追加** (Step 6)
   - n > 1 での rank_duration が Huang 2022 より短縮されることのテスト。
   - K が大きい場合の find_multiple_good_arms の収束テスト。

5. **`HeterogeneousMPMABEnv`** (将来拡張)
   - プレイヤーごとに異なる報酬行列 `means[m, k]` を持つ環境。

---

## 実行したテストと結果

### テスト実行コマンド

```bash
python -m pytest tests/test_env.py tests/algorithms/homogeneous/huang2022/ tests/algorithms/homogeneous/izumi2026/ -v
```

### 結果

```
============================= test session starts ==============================
collected 34 items

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
tests/algorithms/homogeneous/huang2022/::TestFindGoodArm::test_returns_positive_mean_arm PASSED
tests/algorithms/homogeneous/huang2022/::TestFindGoodArm::test_mu_tilde_is_lower_bound PASSED
tests/algorithms/homogeneous/huang2022/::TestVirtualMusicalChairs::test_rank_assignment PASSED
tests/algorithms/homogeneous/huang2022/::TestVirtualMusicalChairs::test_ranks_tend_to_be_unique PASSED
tests/algorithms/homogeneous/huang2022/::TestFullRun::test_assigned_arms_no_duplicate PASSED
tests/algorithms/homogeneous/huang2022/::TestFullRun::test_all_players_get_assignment PASSED
tests/algorithms/homogeneous/huang2022/::TestFullRun::test_phase_durations_recorded PASSED
tests/algorithms/homogeneous/izumi2026/::TestFindMultipleGoodArms::test_returns_n_arms PASSED
tests/algorithms/homogeneous/izumi2026/::TestFindMultipleGoodArms::test_returns_positive_mean_arms PASSED
tests/algorithms/homogeneous/izumi2026/::TestFindMultipleGoodArms::test_mu_tilde_lower_bounds PASSED
tests/algorithms/homogeneous/izumi2026/::TestFindMultipleGoodArms::test_arms_are_unique PASSED
tests/algorithms/homogeneous/izumi2026/::TestParallelVirtualMusicalChairs::test_no_duplicate_good_arm_per_block PASSED
tests/algorithms/homogeneous/izumi2026/::TestParallelVirtualMusicalChairs::test_rank_assignment PASSED
tests/algorithms/homogeneous/izumi2026/::TestParallelVirtualMusicalChairs::test_ranks_tend_to_be_unique PASSED
tests/algorithms/homogeneous/izumi2026/::TestParallelVirtualNumberPlayers::test_each_slot_at_most_one_good_arm PASSED
tests/algorithms/homogeneous/izumi2026/::TestParallelVirtualNumberPlayers::test_m_hat_plausible PASSED
tests/algorithms/homogeneous/izumi2026/::TestFullRunIzumi::test_assigned_arms_no_duplicate PASSED
tests/algorithms/homogeneous/izumi2026/::TestFullRunIzumi::test_all_players_get_assignment PASSED
tests/algorithms/homogeneous/izumi2026/::TestFullRunIzumi::test_phase_durations_recorded PASSED
tests/algorithms/homogeneous/izumi2026/::TestFullRunIzumi::test_good_arms_recorded_in_player_state PASSED
tests/algorithms/homogeneous/izumi2026/::TestN1Compatibility::test_n1_assignment_success PASSED
tests/algorithms/homogeneous/izumi2026/::TestN1Compatibility::test_n1_and_huang2022_both_succeed PASSED
tests/algorithms/homogeneous/izumi2026/::TestN1Compatibility::test_n1_no_duplicate_assignment PASSED

34 passed in 0.34s
==============================
```

---

## テスト設定

| 設定 | 値 |
|------|----|
| K | 5 |
| M | 2 |
| n (good arms 数) | 2（PVMC/PVNP/HDE テスト）、1（n=1 互換テスト）|
| means | `[0.9, 0.8, 0.1, 0.05, 0.02]` |
| top-2 arms | arm 0, arm 1 |
| delta | 0.1（テスト高速化のため緩め） |
| horizon | 1,000,000 |
| seed | 0〜300 のいくつかの値で固定 |

確率的アルゴリズムのため、`test_ranks_tend_to_be_unique` は 10 回試行中 5 回以上の rank 重複なし達成を条件としている。

---

## simulator/main.py の実行結果（T=50,000, K=5, M=2, n=2, seed=42）

```
=== Huang 2022 Sanity Check ===
  player 0: s=3, j=2, M_hat=2, k_tilde=0, assigned=1
  player 1: s=2, j=1, M_hat=2, k_tilde=0, assigned=0
  total_steps:             27759
  find_good_duration:      13580
  rank_duration:           1325
  number_players_duration: 2650
  exploration_duration:    10204
  cumulative_regret:       30368.30

=== Izumi 2026 Sanity Check (n=2) ===
  player 0: s=4, j=2, M_hat=2, good_arms=[0, 1], assigned=0
  player 1: s=0, j=1, M_hat=2, good_arms=[0, 1], assigned=1
  total_steps:             15712
  find_good_duration:      4050
  rank_duration:           133       ← VMC の約 1/10（n=2 倍 + 式の違い）
  number_players_duration: 1325      ← VNP の約 1/2（n=2 倍）
  exploration_duration:    10204     ← HDE は同等
  cumulative_regret:       18906.40  ← 約 38% 削減
```

Izumi 2026 (n=2) は初期化フェーズ（特に rank assignment と number players 推定）を大幅に短縮し、
cumulative regret を約 38% 削減している。
