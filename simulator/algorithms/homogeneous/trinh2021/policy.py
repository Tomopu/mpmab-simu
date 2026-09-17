"""
Randomized Selfish KL-UCB の 1 ステップ分の行動選択。

Trinh and Combes (2021), "A High Performance, Low Complexity Algorithm for Multi-Player
Bandits Without Collision Sensing Information", arXiv:2102.10200, Algorithm 2 に対応する。

各プレイヤー m は他のプレイヤーを考えず（Selfish）、自分の観測だけから KL-UCB 指数を計算する:
    N_{m,k}(t)   : 時刻 t-1 までに player m が arm k を選んだ回数
    mu_{m,k}(t)  : 時刻 t-1 までに player m が arm k から得た報酬の平均（衝突時の報酬 0 も含む）
    b_{m,k}(t)   = max{q in [0,1] : N_{m,k}(t) d(mu_{m,k}(t), q) <= f(t)},  f(t) = log t + c log log t
対称性を崩すため、標準正規乱数 Z_{m,k}(t) を t で割って加えた値の argmax を選ぶ:
    pi_m(t) = argmax_k { b_{m,k}(t) + Z_{m,k}(t) / t }

no-sensing 設定なので、衝突フラグは一切使わない（観測した報酬 r_m(t) だけで統計を更新する）。
論文の実験と同じく c = 0（f(t) = log t）を既定値とする。
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from simulator.algorithms.homogeneous.trinh2021.klucb import klucb_bernoulli


def exploration_function(t: int, c: float = 0.0) -> float:
    """
    探索関数 f(t) = log t + c log log t を返す。

    Args:
        t: 時刻（1-based）
        c: log log t の係数。論文の実験に合わせて既定は 0。

    Returns:
        f(t) の値。log log t が定義されない t < 3 では log log t の項を 0 とする。
    """
    value = math.log(t)
    if c > 0.0 and t >= 3:
        value += c * math.log(math.log(t))
    return value


def select_actions(
    counts: np.ndarray,
    reward_sums: np.ndarray,
    t: int,
    noise: np.ndarray,
    prev_index: np.ndarray | None = None,
    c: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    全プレイヤーの行動を 1 ステップ分まとめて選ぶ。

    Args:
        counts: N_{m,k}(t)。形は (..., M, K)。先頭に試行の次元を付けてもよい。
        reward_sums: player m が arm k から得た報酬の合計。counts と同じ形。
        t: 時刻（1-based）
        noise: 標準正規乱数 Z_{m,k}(t)。counts と同じ形。
        prev_index: 前ステップの KL-UCB 指数（Newton 法の初期値。結果は変わらず速くなるだけ）
        c: 探索関数の log log t の係数

    Returns:
        (actions, index)
        - actions: 形 (..., M) の arm index（0-based）
        - index: 今ステップの KL-UCB 指数（次ステップの初期値に使う）
    """
    # 1. 経験平均 mu_hat = 報酬和 / max(N, 1)（論文の定義どおり分母を 1 以上にする）
    mu_hat = reward_sums / np.maximum(counts, 1.0)
    # 2. KL-UCB 指数（未試行 arm は 1）
    index = klucb_bernoulli(mu_hat, counts, exploration_function(t, c), init=prev_index)
    # 3. 標準正規乱数を t で割って加え、argmax を選ぶ（連続分布なので同点は確率 0）
    actions = np.argmax(index + noise / t, axis=-1)
    return actions, index
