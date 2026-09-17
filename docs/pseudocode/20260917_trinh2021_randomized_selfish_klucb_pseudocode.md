# Randomized Selfish KL-UCB（Trinh and Combes, 2021）

出典: C. Trinh and R. Combes, "A High Performance, Low Complexity Algorithm for Multi-Player Bandits Without Collision Sensing Information", arXiv:2102.10200, 2021. Section 2.3 と Algorithm 2。

## 統計量（player m、arm k、時刻 t）

- `N_{m,k}(t)`: 時刻 `t-1` までに player m が arm k を選んだ回数
- `mu_{m,k}(t) = (1 / max(N_{m,k}(t), 1)) * sum_{s<t} r_m(s) 1{pi_m(s) = k}`: 観測報酬の平均。衝突で得た報酬 0 も 1 回の試行として数える。

## 指数

```
b_{m,k}(t) = max{ q in [0,1] : N_{m,k}(t) d(mu_{m,k}(t), q) <= f(t) }
f(t) = log t + c log log t    （論文の実験では c = 0）
d(p, q) = p log(p/q) + (1-p) log((1-p)/(1-q))
```

## Algorithm 2（player m ごとに独立に実行）

```
for t = 1, ..., T do
  for k = 1, ..., K do
    b_{m,k}(t) を計算する
    Z_{m,k}(t) ~ N(0, 1) を引く
  end for
  pi_m(t) = argmax_k { b_{m,k}(t) + Z_{m,k}(t) / t }
  報酬 r_m(t) を観測して統計量を更新する
end for
```

## 実装との対応

| 論文 | 実装 |
|---|---|
| 指数 `b_{m,k}(t)` | `trinh2021/klucb.py` の `klucb_bernoulli`（区間を保つ Newton 法。前ステップの指数を初期値に使う） |
| 行動選択 | `trinh2021/policy.py` の `select_actions` |
| 1 ステップずつ Runner で実行 | `trinh2021/algorithm.py` の `HomogeneousRandomizedSelfishKLUCB` |
| 比較実験用の高速版 | `trinh2021/batch.py` の `simulate_rskl_batch`（環境も含めて trial 方向にベクトル化） |

- 未試行（`N = 0`）の arm は制約がないので指数 1。経験平均が 1 の arm も指数 1。
- `t = 1` では `f(1) = 0` なので、試行済みの arm の指数は経験平均になる。
- 衝突フラグは意思決定に使わない（no-sensing）。

## 論文の本文と著者の公開実装の違い

著者の公開実装（https://github.com/ctrnh/multi_player_multi_armed_bandit_algorithms 、論文の実験で使われた版）は、論文の本文と次の点が違う。
そこで 2 つの版を用意し、比較実験では algorithm 名で選ぶ。

| 項目 | 論文の本文（`variant="paper"`、`trinh2021_rskl`） | 著者の公開実装（`variant="authors_code"`、`trinh2021_rskl_c3`） |
|---|---|---|
| 探索関数 | `log t + c log log t`。本文に「実用上は通常 c = 0」とあるので c = 0 | `log t0 + 3 log log t0`（c = 3）。`t0 = t - 1` は 0 始まりの時刻 |
| 指数の精度 | 厳密（Newton 法で 1e-12） | 二分法で幅 1e-3 まで絞り、区間の中点（`cklucb.pyx` の `computeKLUCB`） |
| 経験平均 | 報酬和 / max(N, 1) | successes / (1e-7 + pulls) |
| 未試行 arm の指数 | 1 | +inf（同点は一様ランダム）。実装では 1e6 を使い、正規乱数で 1 本を選ぶ（同じ分布） |
| 乱数の大きさ | Z / t | 標準偏差 1/(self.t + 1) の正規乱数（1 始まりの t なら Z / t と同じ） |

## 検証

- 論文の式をスカラーで書き写したものと、ランダムな 3000 状態で行動が全件一致した。
- 著者実装（`computeKLUCB` と `Selfishucb.run` の行動選択）を書き写したものと、ランダムな 3000 状態で行動が全件一致し、指数の差は最大 1.1e-16 だった。
- `tests/algorithms/homogeneous/trinh2021/` に、指数の計算とバッチ版の指標のテストがある。
