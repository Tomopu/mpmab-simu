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

variant で次の版を選べる（"selfish_klucb" と "selfish_ts" は比較用の基準で、Trinh and Combes の手法ではない）:
    - "paper": 論文の本文の定義。指数は厳密に計算し、未試行 arm の指数は 1。c は引数で与える（既定 0。
      論文の本文に「実用上は通常 c = 0 とする」とある）。
    - "authors_code": 著者の公開実装（論文の実験で使われた版）と同じ計算。c = 3、指数は幅 1e-3 の二分法、
      探索関数には 0 始まりの時刻 t - 1 を使い、未試行 arm の指数は +inf とする。
    - "selfish_klucb": 乱数の項のない Selfish KL-UCB（Besson and Kaufmann, 2018。原論文の Algorithm 1）。
      指数は "paper" と同じ（c は引数）。同点の arm の中からは一様に 1 本を選ぶ（noise に 1e-9 を掛けて加える。
      同点でない arm の順位は変わらない）。
    - "selfish_ts": Selfish Thompson Sampling（比較用）。player m は自分の観測（衝突の報酬 0 も含む）から
      Beta(1 + 成功数, 1 + 失敗数) を作り、そこから引いた標本の argmax を選ぶ。標本は batch.py が
      trial ごとの乱数系列から作り、select_actions には noise として渡す。
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from simulator.algorithms.homogeneous.trinh2021.klucb import (
    klucb_bernoulli,
    klucb_bernoulli_authors_code,
)

# 著者実装の +inf の代わりに使う値。正規乱数 Z/t（標準偏差 1 以下）を足しても、試行済み arm の指数（1 以下）を
# 必ず上回るので、未試行 arm の中から正規乱数の最大で 1 本を選ぶことになる。著者実装の「+inf の arm の中から
# 一様に選ぶ」と、同じ分布になる。
_UNPULLED_INDEX = 1e6
VARIANTS = ("paper", "authors_code", "selfish_klucb", "selfish_ts")
# selfish_klucb で同点だけを崩すための乱数の大きさ（KL-UCB 指数の差がこれより小さくなることは実質ない）
_TIE_BREAK_SCALE = 1e-9


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
    variant: str = "paper",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    全プレイヤーの行動を 1 ステップ分まとめて選ぶ。

    Args:
        counts: N_{m,k}(t)。形は (..., M, K)。先頭に試行の次元を付けてもよい。
        reward_sums: player m が arm k から得た報酬の合計。counts と同じ形。
        t: 時刻（1-based）
        noise: 標準正規乱数 Z_{m,k}(t)。counts と同じ形。
        prev_index: 前ステップの KL-UCB 指数（Newton 法の初期値。結果は変わらず速くなるだけ）
        c: 探索関数の log log t の係数（variant="paper" のときだけ使う）
        variant: "paper"（論文の本文）または "authors_code"（著者の公開実装）

    Returns:
        (actions, index)
        - actions: 形 (..., M) の arm index（0-based）
        - index: 今ステップの KL-UCB 指数（次ステップの初期値に使う）
    """
    if variant == "selfish_ts":
        # Thompson Sampling: noise に事後分布からの標本が入っている（batch.py が作る）。そのまま argmax を取る
        return np.argmax(noise, axis=-1), noise
    if variant in ("paper", "selfish_klucb"):
        # 1. 経験平均 mu_hat = 報酬和 / max(N, 1)（論文の定義どおり分母を 1 以上にする）
        mu_hat = reward_sums / np.maximum(counts, 1.0)
        # 2. KL-UCB 指数（未試行 arm は 1）
        index = klucb_bernoulli(mu_hat, counts, exploration_function(t, c), init=prev_index)
    elif variant == "authors_code":
        # 1. 著者実装の経験平均 successes / (1e-7 + pulls)（未試行なら 0）
        mu_hat = reward_sums / (1e-7 + counts)
        # 2. 著者実装の指数。時刻は 0 始まり（self.t = t - 1）で計算し、未試行 arm は +inf 相当にする
        index = klucb_bernoulli_authors_code(mu_hat, counts, t - 1)
        index = np.where(counts < 1, _UNPULLED_INDEX, index)
    else:
        raise ValueError(f"unknown variant: {variant}")
    if variant == "selfish_klucb":
        # 3'. 乱数の項はない。同点の arm だけを一様に崩す
        actions = np.argmax(index + _TIE_BREAK_SCALE * noise, axis=-1)
        return actions, index
    # 3. 標準正規乱数を t で割って加え、argmax を選ぶ（連続分布なので同点は確率 0）
    actions = np.argmax(index + noise / t, axis=-1)
    return actions, index
