"""
KL-UCB 指数（klucb_bernoulli）のテスト。

確認内容:
- 高精度の二分法（スカラー版）と一致すること
- 未試行・経験平均 1・f = 0 の特別な場合
- 前ステップの指数を初期値に渡しても結果が変わらないこと
"""

import numpy as np

from simulator.algorithms.homogeneous.trinh2021.klucb import bernoulli_kl, klucb_bernoulli


def _reference_index(p: float, n: float, f: float) -> float:
    """定義 max{q : n d(p, q) <= f} をスカラーの二分法で 200 回反復して求める（比較用）。"""
    if n == 0 or p >= 1.0:
        return 1.0
    lo, hi = p, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if n * float(bernoulli_kl(np.array(p), np.array(mid))) <= f:
            lo = mid
        else:
            hi = mid
    return lo


def test_matches_reference_bisection():
    # Given: ランダムな経験平均と試行回数（0 と 1 の端点を含む）
    rng = np.random.default_rng(0)
    mu = rng.random(300)
    mu[:10] = 0.0
    mu[10:20] = 1.0
    n = rng.integers(0, 5000, 300).astype(float)

    for f in [0.3, 2.0, 13.8]:
        # When: まとめて計算する
        got = klucb_bernoulli(mu, n, f)
        # Then: 二分法の結果と 1e-10 以内で一致する
        expected = np.array([_reference_index(p, k, f) for p, k in zip(mu, n)])
        assert np.max(np.abs(got - expected)) < 1e-10


def test_special_cases():
    # Given: 未試行、経験平均 1、通常の arm
    mu = np.array([0.3, 1.0, 0.4])
    n = np.array([0.0, 10.0, 10.0])

    # When / Then: f > 0 なら未試行と経験平均 1 は指数 1
    index = klucb_bernoulli(mu, n, 1.0)
    assert index[0] == 1.0
    assert index[1] == 1.0
    assert 0.4 < index[2] < 1.0

    # When / Then: f = 0（t = 1）なら試行済みは経験平均、未試行は 1
    index0 = klucb_bernoulli(mu, n, 0.0)
    assert np.allclose(index0, [1.0, 1.0, 0.4])


def test_warm_start_does_not_change_result():
    # Given: 前ステップの指数（別の f で計算した値）
    rng = np.random.default_rng(1)
    mu = rng.random((4, 3, 6))
    n = rng.integers(1, 1000, (4, 3, 6)).astype(float)
    prev = klucb_bernoulli(mu, n, 5.0)

    # When: その値を初期値にして f = 5.01 で計算する
    warm = klucb_bernoulli(mu, n, 5.01, init=prev)
    cold = klucb_bernoulli(mu, n, 5.01)

    # Then: 初期値の有無で結果は変わらない
    assert np.max(np.abs(warm - cold)) < 1e-10
