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
