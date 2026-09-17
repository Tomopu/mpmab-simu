"""
Bernoulli 分布の KL-UCB 指数を numpy 配列でまとめて計算する。

Trinh and Combes (2021) の式:
    b = max{ q in [0, 1] : N * d(mu_hat, q) <= f(t) }
    d(p, q) = p log(p/q) + (1-p) log((1-p)/(1-q))   （Bernoulli の KL ダイバージェンス）

d(p, q) は q >= p の範囲で q について単調増加かつ凸なので、
区間 [mu_hat, 上界] を保ったまま Newton 法で根を求める（Newton が区間外に出たら二分法に切り替える）。
Randomized Selfish KL-UCB では毎ステップ全 (プレイヤー, arm) の指数が要るため、
前ステップの指数を初期値に使うと多くの要素が 1〜2 回の反復で収束する。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# log(0) を避けるための下限・上限
_EPS = 1e-15


def bernoulli_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Bernoulli 分布どうしの KL ダイバージェンス d(p, q) を要素ごとに計算する。

    Args:
        p: 期待値 p の配列（0 以上 1 以下）
        q: 期待値 q の配列（0 以上 1 以下）

    Returns:
        d(p, q) の配列。0 log 0 = 0 の約束に合わせるため、p と q を [_EPS, 1-_EPS] に丸めて計算する。
    """
    p = np.clip(p, _EPS, 1.0 - _EPS)
    q = np.clip(q, _EPS, 1.0 - _EPS)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def klucb_bernoulli(
    mu_hat: np.ndarray,
    counts: np.ndarray,
    f: float,
    init: Optional[np.ndarray] = None,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> np.ndarray:
    """
    KL-UCB 指数 b = max{q in [0,1] : counts * d(mu_hat, q) <= f} を要素ごとに求める。

    Args:
        mu_hat: 経験平均の配列（任意の形）
        counts: 試行回数の配列（mu_hat と同じ形）
        f: 探索関数の値 f(t)（全要素で共通）
        init: Newton 法の初期値（前ステップの指数など）。None なら区間の中点から始める。
        tol: 反復を止める更新幅
        max_iter: 反復回数の上限

    Returns:
        mu_hat と同じ形の指数配列。
        - counts = 0 の要素は制約がないので 1
        - mu_hat = 1 の要素は d(1, q) = log(1/q) より q = 1 まで許されるので 1
        - f <= 0 の要素は q = mu_hat
    """
    mu = np.clip(np.asarray(mu_hat, dtype=float), 0.0, 1.0)
    n = np.asarray(counts, dtype=float)
    out = np.ones_like(mu)

    if f <= 0.0:
        # f(1) = log 1 = 0 のとき: 試行済みの arm は指数 = 経験平均、未試行は 1
        return np.where(n > 0, mu, 1.0)

    # 1. Newton 法で解く必要がある要素を選ぶ（未試行と mu_hat = 1 は指数 1 で確定）
    active = (n > 0) & (mu < 1.0 - 1e-12)
    if not np.any(active):
        return out
    p = mu[active]
    c = f / n[active]  # 解くべき式: d(p, q) = c

    # 2. 根を挟む区間 [lo, hi] を作る
    #    Pinsker の不等式 d(p, q) >= 2 (q - p)^2 より、根は p + sqrt(c / 2) 以下
    lo = p.copy()
    hi = np.minimum(1.0 - _EPS, p + np.sqrt(c / 2.0))

    # 3. 初期値（前ステップの指数を区間内に丸めて使う）
    if init is None:
        q = 0.5 * (lo + hi)
    else:
        q = np.clip(np.asarray(init, dtype=float)[active], lo, hi)

    # d(p, q) のうち q に依存しない部分 p log p + (1-p) log(1-p) を先に計算しておく
    pc = np.clip(p, _EPS, 1.0 - _EPS)
    neg_entropy = pc * np.log(pc) + (1.0 - pc) * np.log1p(-pc)

    # 未収束の要素だけを反復する（前ステップの指数を初期値にすると、2 回目以降はごく一部だけが残る）
    idx = np.arange(q.size)
    for _ in range(max_iter):
        qi, pi, ci, li, hi_i = q[idx], pc[idx], c[idx], lo[idx], hi[idx]
        qc = np.clip(qi, _EPS, 1.0 - _EPS)
        g = neg_entropy[idx] - pi * np.log(qc) - (1.0 - pi) * np.log1p(-qc) - ci
        # 4. g <= 0 なら q は制約を満たす（根は q 以上）、g > 0 なら根は q 未満
        feasible = g <= 0.0
        li = np.where(feasible, qi, li)
        hi_i = np.where(feasible, hi_i, qi)
        # 5. Newton ステップ g'(q) = (q - p) / (q (1 - q))。区間外や微分 0 なら二分法に切り替える
        dg = (qc - pi) / (qc * (1.0 - qc))
        with np.errstate(divide="ignore", invalid="ignore"):
            q_newton = qi - g / dg
        use_newton = (dg > 0.0) & (q_newton > li) & (q_newton < hi_i)
        q_next = np.where(use_newton, q_newton, 0.5 * (li + hi_i))
        # 6. 更新して、更新幅が tol 以上の要素だけを次の反復に残す
        q[idx], lo[idx], hi[idx] = q_next, li, hi_i
        moving = np.abs(q_next - qi) >= tol
        if not np.any(moving):
            break
        idx = idx[moving]

    out[active] = q
    return out
