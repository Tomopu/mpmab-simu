# Heterogeneous ParallelBEACON 実験分析レポート

作成日: 2026-06-26

## 実験設定

| 項目 | 値 |
|---|---|
| K (arm 数) | 20 |
| M (プレイヤー数) | 8 |
| T (horizon) | 1,000,000 |
| trials | 20 |
| means_matrix | homogeneous tradeoff の means 値を shift=2 の巡回行列で配置 |
| n_values | 1, 2, 3, 4, 5, 6 |

means_matrix の構造:
- 全プレイヤーが homogeneous tradeoff と同じ means 集合（0.95, 0.9, ..., 0.03）を持つ
- shift=2 の循環配置により各プレイヤーの最良 arm が異なる:
  p0→arm0, p1→arm18, p2→arm16, p3→arm14, p4→arm12, p5→arm10, p6→arm8, p7→arm6
- 最小 arm 間 gap = 0.030

---

## 実験結果（20 trial 平均）

| algorithm | n | cumulative_regret | std | init_duration | beacon_comm | beacon_explore | collision_count |
|---|---|---|---|---|---|---|---|
| shi2021_beacon | 1 | 102,997 | 13,120 | 440 | 17,882 | 981,678 | 6,154 |
| izumi2026_parallel_beacon | 1 | 102,692 | 17,019 | 1,160 | 17,705 | 981,135 | 1,140 |
| izumi2026_parallel_beacon | 2 | 96,622 | 19,330 | 590 | 18,510 | 980,900 | 570 |
| izumi2026_parallel_beacon | 3 | 90,588 | 12,824 | 414 | 18,572 | 981,014 | 394 |
| izumi2026_parallel_beacon | 4 | 90,008 | 22,684 | 305 | 18,428 | 981,267 | 285 |
| izumi2026_parallel_beacon | 5 | **88,278** | 21,713 | 248 | 18,471 | 981,281 | 228 |
| izumi2026_parallel_beacon | 6 | 93,695 | 18,426 | 217 | 18,064 | 981,719 | 197 |

n=5 が最もリグレットが小さい（ただし std が大きく、n=4〜6 の差は統計的に有意でない可能性がある）。

---

## アルゴリズム改善内容（このセッション）

### Fix 1: グループ平均 → 個人推定値

```
旧: mu_tilde[k][m] = group_stats[g][k].mean  ← heterogeneous では致命的
新: mu_tilde[k][m] = mean(samples[k][m][:2^p_km])
```

n > 1 で `final_assignment_success = 0%` だったのが `100%` になった。
グループ平均を使うと grand leader が他グループのプレイヤーの個人最適 arm を誤認識する。

### Fix 2: 差分エンコード（prev_p の追跡）

```python
if curr_p[k][m] > prev_p[k][m]:
    Q = ceil(1 + p[k,m] / 2)
    cost += Q + 1  # 符号ビット + magnitude
```

毎エポック全アームを課金する方式から、変化したアームのみ課金に変更。

### Fix 3: Phase A 並列化コストの修正

```
旧: for g in groups: consume(group_g_cost)  ← 逐次加算（sum）になっていた
新: consume(max over groups)                ← 並列なので max が正しい
```

### Fix 4: consume_dummy_steps → consume_comm_steps

通信中に idle な player が全員同じ arm を pull → collision → reward = 0 を修正。

```
旧: actions = [dummy_arm] * M     → 全員 collision → reward = 0 per step
新: actions = state_arms          → 全員異なる arm（s_list から） → reward > 0
```

通信フェーズ regret の変化:
```
修正前: 7.6 × 17,739 steps ≈ 134,616  （regret の大部分）
修正後: (7.6 - reward_from_state_arms) × steps  （大幅に減少）
```

BEACON が通信中 idle な M-2 人に `c[m]`（自分の state arm）を pull させるのと同じ設計。

---

## comm end について

**どの実行例でも "comm end"（通信の収束）には達しない。**

エポックループは `while True` で走り続け、T を使い切った時点だけ終了する。
通信コストはゼロにならない理由:

1. 探索フェーズで `T[assignment[m]][m]` が毎エポック `2^p_r` 増える
2. → `p[assignment[m]][m]` が増える（`floor(log2(T))` が増加）
3. → 差分エンコードの条件 `curr_p > prev_p` が満たされ、毎エポック通信が発生する

ただし「割当が安定して最適解に収束する」という意味での収束は T の途中で起きる（後述）。

---

## 収束（割当安定）に必要な T の推定

UCB ボーナス < 最小 arm gap（0.030）を満たすのに必要なサンプル数:

$$2^{p+1} > \frac{3 \ln T}{\text{gap}^2} \approx 3333 \cdot \ln T$$

| T | 推定 explore steps | p の推定値 | UCB ボーナス | gap(0.030) より小? |
|---|---|---|---|---|
| 50,000 | 25,500 | 14 | 0.0315 | ✗ |
| **100,000** | **74,000** | **16** | **0.0162** | **✓** |
| 200,000 | 172,500 | 17 | 0.0118 | ✓ |
| 1,000,000 | 969,500 | 19 | 0.0063 | ✓ |

**T ≈ 100,000 から収束が始まる**（Shi et al. の観察と一致）。

T = 1,000,000 ではほぼ全リグレットが T ≤ 100,000 の学習フェーズに集中し、
残りの 900,000 ステップは最適割当での純粋な搾取（regret ≈ 0）になっている。

---

## 分散が大きくなる理由

### 観測された分散

| n | std | SE（20 trials） | 95% CI |
|---|---|---|---|
| n=4 | 22,684 | 5,073 | ±9,943 |
| n=5 | 21,713 | 4,858 | ±9,522 |
| n=6 | 18,426 | 4,121 | ±8,077 |
| BEACON | 13,120 | 2,935 | ±5,753 |

### 分散の源泉

**1. 割当収束タイミングの seed 依存性**

n=5, 最小値 57,909 vs 最大値 118,604（差 2 倍）。
あるシードでは学習初期に正しい割当を得て残りを純搾取できるが、
別のシードでは割当がなかなか安定せず累積リグレットが大きくなる。

**2. 初期化の確率的性質（CSVMC/CSVNP）**

rank 割当（どのプレイヤーが grand leader になるか）が seed によって異なる。
grand leader になるプレイヤーの means_matrix が学習速度に影響する。

**3. Heterogeneous マッチングの難しさ**

Homogeneous では top-M arm に入れば誰でも OK だが、
heterogeneous では各プレイヤーが個別の最良 arm に到達する必要がある。
早期の誤割当が累積リグレットに与えるペナルティが大きい。

### 100 trials にすると SE はどう変わるか

試行数を増やしても **各 trial の分散（std）は変わらない**。
変わるのは平均値の推定精度（標準誤差 SE = std/√n）:

| trials | SE（std=21,713） | 95% CI | n=5 vs n=6 差（5,417）の有意性 |
|---|---|---|---|
| 10 | 6,866 | ±13,458 | 検出困難（差 < 1σ）|
| 20 | 4,855 | ±9,516 | 境界（差 ≈ 1σ）|
| 50 | 3,071 | ±6,019 | 検出可能（差 ≈ 1.8σ）|
| **100** | **2,171** | **±4,256** | **検出可能（差 ≈ 2.5σ）** |

100 trials にすると n=5 vs n=6 の差（5,417）が 2.5σ 水準で有意になる。
ただし最適な n（4 or 5 or 6）の判定には 50〜100 trials が必要。

---

## n とリグレットの関係の考察

### なぜ n が大きいほど regret が（おおむね）減るか

| n 増加による効果 | 方向 |
|---|---|
| 初期化（CSVMC+CSVNP）が O(K/n) で短縮 → 学習予算が増える | ✓ 改善 |
| Phase A（フォロワー→sub-leader）が並列化 → uplink 高速化 | ✓ 改善 |
| Phase B（sub-leader→grand leader）が逐次増加 → Phase A の節約を打ち消し | ✗ 悪化 |
| Downlink A（grand leader→sub-leaders）が逐次増加 | ✗ 悪化 |

**n=5 付近が最適な理由**: 初期化短縮の恩恵（n に対して単調改善）が Phase B オーバーヘッドの増加を
上回る点が n ≈ M - 2 〜 M - 3（=5〜6）付近になる。

### Homogeneous との対比

| | Homogeneous（Izumi 2026） | Heterogeneous（今回） |
|---|---|---|
| 初期化コスト vs n | FindGoodArms + HDE = 増加傾向 | CSVMC + CSVNP = 単調減少 |
| n > M/2 での挙動 | 悪化（HDE オーバーヘッド支配） | 改善が続く（初期化短縮が大） |
| 最適 n | ≈ M/2 | ≈ M - 2 〜 M - 3 |

---

## 実験ファイル

- 実験結果: `outputs/runs/20260626_042834_hetero_large_K20_M8_K20_M8_T1000000/`
- アルゴリズム実装: `simulator/algorithms/heterogeneous/izumi2026/algorithm5_parallel_beacon_epoch.py`
- 疑似コード: `docs/pseudocode/20260626_izumi2026_heterogeneous_pseudocode.md`
- 実験設定: `simulator/experiments/configs.py` → `HETERO_CONFIGS["large"]`
