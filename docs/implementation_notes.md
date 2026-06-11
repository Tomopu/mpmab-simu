# MPMAB 実装メモ

Homogeneous / Heterogeneous 全 4 モデルの実装内容をまとめたドキュメント。
ソースパスの対応、各コンポーネントの概要、簡略化箇所、実験コマンドを記録する。

---

## ソースパス対応

### Homogeneous

| 実装内容 | ソースパス |
|---------|-----------|
| 環境（MPMAB env） | `simulator/envs/bernoulli_mpmab.py` |
| 履歴記録 | `simulator/core/trace.py` |
| 実行ランナー | `simulator/core/runner.py` |
| アルゴリズム基底クラス | `simulator/algorithms/base.py` |
| Huang 2022 | `simulator/algorithms/homogeneous/huang2022/algorithm.py` |
| Izumi 2026 | `simulator/algorithms/homogeneous/izumi2026/algorithm.py` |
| 評価指標（Homogeneous） | `simulator/utils/metrics.py` — `compute_metrics` |
| プロット | `simulator/utils/plotter.py` |
| 比較実験スクリプト | `simulator/experiments/compare_homogeneous.py` |
| 動作確認スクリプト | `simulator/main.py` |
| テスト: 環境 | `tests/envs/test_env.py` |
| テスト: Huang 2022 | `tests/algorithms/homogeneous/huang2022/` |
| テスト: Izumi 2026 | `tests/algorithms/homogeneous/izumi2026/` |

### Heterogeneous

| 実装内容 | ソースパス |
|---------|-----------|
| 環境（Heterogeneous env） | `simulator/envs/heterogeneous_mpmab.py` |
| 実行ランナー | `simulator/algorithms/heterogeneous/shi2021/runner_hetero.py` |
| Orthogonalization + Rank Assignment（Wang 2020） | `simulator/algorithms/heterogeneous/shi2021/wang2020_orthogonalization.py` |
| BEACON Send/Receive（forced collision 通信） | `simulator/algorithms/heterogeneous/shi2021/shi2021_beacon_communication.py` |
| BEACON エポックループ | `simulator/algorithms/heterogeneous/shi2021/shi2021_beacon_epoch.py` |
| Matching Oracle（Hungarian 法） | `simulator/algorithms/heterogeneous/shi2021/oracle.py` |
| Shi 2021 BEACON メインクラス | `simulator/algorithms/heterogeneous/shi2021/algorithm.py` |
| ParallelBEACON エポックループ | `simulator/algorithms/heterogeneous/izumi2026/parallel_beacon_epoch.py` |
| Izumi 2026 ParallelBEACON メインクラス | `simulator/algorithms/heterogeneous/izumi2026/algorithm.py` |
| 評価指標（Heterogeneous） | `simulator/utils/metrics.py` — `compute_hetero_metrics` |
| 比較実験スクリプト | `simulator/experiments/compare_heterogeneous.py` |
| テスト: 環境・初期化・BEACON | `tests/algorithms/heterogeneous/shi2021/` |
| テスト: ParallelBEACON | `tests/algorithms/heterogeneous/izumi2026/` |

---

## コンポーネント概要

### BernoulliMPMABEnv

`simulator/envs/bernoulli_mpmab.py`

- `step(actions)`: M 人の action を同時に受け取り、collision 判定と報酬生成を行う。
  - 同一 arm を 2 人以上が選んだ場合（collision）: 全員の報酬 = 0。
  - collision なし: `Bernoulli(means[k])` の報酬。
- `reset(seed)`: 乱数シードをリセットして再現可能な状態に戻す。
- `optimal_total_reward`: top-M arm の平均報酬和。初期化時に計算。
- `StepResult.collisions`: 環境と評価指標での記録専用。no-sensing アルゴリズムには渡さない。

### Runner / Trace

`simulator/core/runner.py`, `simulator/core/trace.py`

- `Trace`: シミュレーション全体の履歴を保持。`to_dataframe()` で pandas DataFrame として取得可能。
- `Runner`: 環境との同期インターフェース。`step(actions, phase)` で 1 ステップ実行し、horizon 到達時に `HorizonReached` を送出。

### HomogeneousHuang2022

`simulator/algorithms/homogeneous/huang2022/algorithm.py`

| メソッド | 対応アルゴリズム |
|---------|----------------|
| `find_good_arm(runner)` | Algorithm 1 FindGoodArm |
| `virtual_musical_chairs(runner, k_tilde, tau)` | Algorithm 2 VirtualMusicalChairs |
| `virtual_number_players(runner, k_tilde, s_list, tau)` | Algorithm 3 VirtualNumberPlayers |
| `distributed_exploration(runner, k_tilde, j_list, M_hat_list, tau)` | Algorithm 4 DistributedExploration |
| `run(runner)` | Algorithm 5 Proposed algorithm |

`tau` の計算:
```python
tau_rank = ceil(K * ln(1/δ) / μ̃)   # VMC 用
tau_comm = ceil(ln(1/δ) / μ̃)        # VNP・DE 用
```

### HomogeneousMultiChannelIzumi2026

`simulator/algorithms/homogeneous/izumi2026/algorithm.py`

| メソッド | 対応アルゴリズム |
|---------|----------------|
| `find_multiple_good_arms(runner)` | Algorithm 1 FindMultipleGoodArms |
| `parallel_virtual_musical_chairs(runner, good_arms, tau)` | Algorithm 2 ParallelVirtualMusicalChairs |
| `parallel_virtual_number_players(runner, good_arms, s_list, tau)` | Algorithm 3 ParallelVirtualNumberPlayers |
| `hierarchical_distributed_exploration(runner, good_arms, j_list, M_hat_list, tau)` | Algorithm 4 HierarchicalDistributedExploration |
| `run(runner)` | Algorithm 5 ProposedParallelAlgorithm |

`tau` はフェーズごとに分ける（`20260518_bugfix_report.md` 修正 4 参照）:
```python
tau_rank = ceil(K * ln(1/δ) / μ̃)   # ParallelVMC 用
tau_comm = ceil(ln(1/δ) / μ̃)        # VNP・HDE 用
```

#### Huang 2022 との比較

| 項目 | Huang 2022 | Izumi 2026 |
|-----|-----------|-----------|
| FindGoodArm | 1 本探索、T1 = 6K | n 本探索、T1 = 6\|K\|（active set ベース） |
| VMC 総ステップ | K × τ_rank | ceil(K × τ_rank / n)（n 倍短縮） |
| VNP 仮想時刻 | 2K を直列 | n 本並列（n 倍短縮） |
| 通信構造 | Leader / Follower（2 階層） | Grand Leader / Sub-Leader / Follower（3 階層） |

---

## 簡略化した箇所

### 通信の簡略実装（DistributedExploration / HierarchicalDistributedExploration）

**内容:** 論文の forced collision bit 伝送（EncoderSendFloat / DecoderReceiveFloat、ComGrandLeader / ComSubLeader / ComFollower）は完全には再現していない。

**実際の実装:**
- 推定値 E[k] は直接集約する（量子化・通信誤りなし）。
- 通信時間コストとして `Q * tau * Ka ステップ` を Trace に記録するため、ダミー arm を選ぶ `runner.step()` を消費する。
- HDE では「最も重いグループの通信時間」で並列通信を近似する。

**影響:**
- regret には通信コストが近似的に反映される。
- 通信誤りによる誤決定はシミュレートされない。
- 集約精度が論文の理論解析（量子化誤差込み）より高いため、B[k] が過大評価になる。

### FindGoodArm の T1 式

**内容:** T1 = `6K * 2^p * ceil(ln(2/δ))` を使用する。旧実装の `6K^2` は誤りで修正済み（`20260518_bugfix_report.md` 修正 1 参照）。

### FindMultipleGoodArms の active_arms 同期

**内容:** active_arms の更新は player 0 の G_list[0] を基準にする。homogeneous 設定では全プレイヤーが同じ good arms を確定するはずであり、player 0 を canonical player として使う。

### HierarchicalDistributedExploration の n>1 問題（未解決）

**内容:** `C_assign = C_accept - G` の設計により、n>1 で good arms が top-M arms と重なる典型ケースで割当が完了しないデッドロックが起きる可能性がある。`20260518_bugfix_report.md` 問題 1 を参照。

---

## Heterogeneous の簡略化した箇所

### ParallelBEACON の通信実装

**内容:** Heterogeneous 版（Izumi 2026 ParallelBEACON）では、アップリンク・ダウンリンクの統計集約を直接行い、forced collision bit 伝送を再現しない。

**実際の実装:**
- follower → sub-leader → grand leader の統計は Python 辞書で直接集約する。
- 通信時間コストは BEACON と同じ式（`Q * Ka ステップ`）のダミー `runner.step()` で消費する。
- ダウンリンク（割当通知）は `ceil(log2(K))` ビット相当の dummy ステップで近似する。

**影響:**
- regret に通信コストが近似的に反映される点は BEACON（Shi 2021）と同じ。
- 通信誤りはシミュレートされない。

### BEACON（Shi 2021）の通信実装

**内容:** BEACON の `Send` / `Receive` は 1 bit = 1 ステップで実際に forced collision をシミュレートする（`shi2021_beacon_communication.py`）。報酬推定値の量子化（EncoderSendFloat / DecoderReceiveFloat）は実装済み。

---

## 比較実験の実行

### Homogeneous

```bash
# 小規模 sanity check（K=5, M=2, T=50000）
python -m simulator.experiments.compare_homogeneous --experiment small --trials 20

# multi-channel speedup 確認（K=10, M=5）
python -m simulator.experiments.compare_homogeneous --experiment speedup --trials 50

# good arm 探索コストとの tradeoff（K=20, M=5）
python -m simulator.experiments.compare_homogeneous --experiment tradeoff --trials 50
```

### Heterogeneous

```bash
# BEACON vs ParallelBEACON（K=5, M=2, T=500000）
python -m simulator.experiments.compare_heterogeneous --experiment small --trials 10

# 非対称な報酬行列（K=6, M=3）
python -m simulator.experiments.compare_heterogeneous --experiment asym --trials 20

# ParallelBEACON のみ（高速デバッグ）
python -m simulator.experiments.compare_heterogeneous \
    --experiment small --trials 2 --horizon 50000 --no-plots --no-beacon
```

主なオプション（詳細は `20260522_experiment_guide.md` 参照）:

| Option | 内容 |
|--------|------|
| `--trials N` | trial 数 |
| `--horizon T` | horizon |
| `--K K --M M` | arm 数・player 数 |
| `--n-values 1,2` | ParallelBEACON の n の値 |
| `--means-matrix JSON` | 報酬行列を JSON で直接指定 |
| `--retry-on-failure` | 割当失敗 trial を seed を変えて再実行 |
| `--no-plots` | CSV のみ出力 |
| `--no-beacon` | BEACON を省略し ParallelBEACON のみ実行 |

出力は `outputs/runs/YYYYMMDD_HHMMSS_<experiment>/` 以下に保存（`.gitignore` 済み）。
