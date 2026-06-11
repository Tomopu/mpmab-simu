"""
HeterogeneousMPMABEnv: Heterogeneous Multi-Player Multi-Armed Bandit 環境。

各プレイヤーが arm ごとに異なる報酬分布を持つ heterogeneous 設定。

論文との対応:
- means[m][k]: player m が arm k を引いたときの期待報酬（Bernoulli パラメータ）
- collision_sensing=True: Shi et al. (2021) BEACON が前提とする collision 観測モード。
  collision した場合、関与プレイヤーは reward=0 かつ collision フラグ=True を受け取る。
- optimal_total_reward: 最大重み二部マッチング（Hungarian 法）で求めた最適割当の期待報酬和。
  homogeneous の top-M arm 和とは異なり、player-arm 対の利益を最大化する割当を使う。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class HeterogeneousStepResult:
    """
    1 ステップの結果を保持するデータクラス（Heterogeneous 版）。

    Attributes:
        actions: 各プレイヤーが選んだ arm index（0-based, 長さ M）
        rewards: 各プレイヤーが得た報酬（0/1, 長さ M）
        collisions: 各プレイヤーの collision フラグ（True/False, 長さ M）
            collision_sensing=True の場合のみ意味を持つ。
        total_reward: このステップの報酬合計
        optimal_total_reward: 最適マッチングの期待報酬和
        instant_regret: optimal_total_reward - total_reward
    """

    actions: List[int]
    rewards: List[float]
    collisions: List[bool]
    total_reward: float
    optimal_total_reward: float
    instant_regret: float


def _hungarian_max_weight(cost: List[List[float]]) -> float:
    """
    M×K コスト行列から最大重み二部マッチング値を返す（M<=K を仮定）。

    scipy が利用可能な場合は scipy.optimize.linear_sum_assignment を使う。
    利用不可の場合は単純な総当たり（M が小さい場合のみ正確）。

    Args:
        cost: cost[m][k] = player m が arm k に割り当てられたときの利益

    Returns:
        最大マッチング値
    """
    M = len(cost)
    K = len(cost[0]) if M > 0 else 0

    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np

        cost_np = np.array(cost)
        # linear_sum_assignment は最小化を解くので符号反転
        row_ind, col_ind = linear_sum_assignment(-cost_np)
        return float(cost_np[row_ind, col_ind].sum())
    except ImportError:
        pass

    # scipy がない場合の fallback: 全通りの M 要素部分集合を試す（M<=8 程度まで）
    from itertools import permutations

    best = 0.0
    arms = list(range(K))
    for perm in permutations(arms, M):
        val = sum(cost[m][perm[m]] for m in range(M))
        if val > best:
            best = val
    return best


class HeterogeneousMPMABEnv:
    """
    Bernoulli 報酬を持つ Heterogeneous Multi-Player Multi-Armed Bandit 環境。

    各プレイヤーは arm ごとに異なる期待報酬を持つ。
    collision_sensing=True のとき、BEACON などの collision sensing アルゴリズムが
    collision フラグを意思決定に使える。

    Args:
        means: 報酬行列。means[m][k] = player m の arm k の期待報酬。
            shape (M, K)。
        collision_sensing: True なら collision を観測可能（BEACON 用）。
        seed: 乱数シード
    """

    def __init__(
        self,
        means: Sequence[Sequence[float]],
        collision_sensing: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.means: List[List[float]] = [list(row) for row in means]
        self.M = len(self.means)
        self.K = len(self.means[0]) if self.M > 0 else 0
        self.collision_sensing = collision_sensing
        self._seed = seed
        self.rng = random.Random(seed)
        self.t = 0

        if self.M < 1:
            raise ValueError("M (num_players) は 1 以上でなければならない。")
        if self.K < 1:
            raise ValueError("K (num_arms) は 1 以上でなければならない。")
        if self.K < self.M:
            raise ValueError(f"K={self.K} は M={self.M} 以上でなければならない（各プレイヤーに arm を割り当てるため）。")

        # 最適マッチングの期待報酬和を事前計算
        self._optimal_total_reward = _hungarian_max_weight(self.means)

    def reset(self, seed: Optional[int] = None) -> None:
        """
        環境をリセットする。

        Args:
            seed: 新しい乱数シード。None なら初期 seed を再利用。
        """
        self.t = 0
        if seed is not None:
            self._seed = seed
        self.rng = random.Random(self._seed)

    def step(self, actions: List[int]) -> HeterogeneousStepResult:
        """
        1 ステップ進める。全プレイヤーの action を同時に受け取り、結果を返す。

        Args:
            actions: 各プレイヤーが選んだ arm index（0-based, 長さ M）

        Returns:
            HeterogeneousStepResult: このステップの結果

        Raises:
            ValueError: actions の長さが M と異なる、または arm index が範囲外の場合
        """
        if len(actions) != self.M:
            raise ValueError(f"actions の長さは M={self.M} でなければならない。got {len(actions)}")
        for a in actions:
            if not (0 <= a < self.K):
                raise ValueError(f"arm index {a} は範囲外。0 以上 K-1={self.K - 1} 以下でなければならない。")

        self.t += 1

        # 1. 各 arm が何人に選ばれたかカウント
        counts = [0] * self.K
        for a in actions:
            counts[a] += 1

        # 2. collision フラグを決定（2 人以上が同じ arm を選んだ場合）
        collisions = [counts[a] >= 2 for a in actions]

        # 3. 各プレイヤーの報酬を生成（heterogeneous: player m は means[m][a] を使う）
        rewards: List[float] = []
        for m, a in enumerate(actions):
            if collisions[m]:
                # collision: 報酬 0
                rewards.append(0.0)
            else:
                # collision なし: Bernoulli(means[m][a]) をサンプリング
                r = 1.0 if self.rng.random() < self.means[m][a] else 0.0
                rewards.append(r)

        total_reward = sum(rewards)
        instant_regret = self._optimal_total_reward - total_reward

        return HeterogeneousStepResult(
            actions=list(actions),
            rewards=rewards,
            collisions=collisions,
            total_reward=total_reward,
            optimal_total_reward=self._optimal_total_reward,
            instant_regret=instant_regret,
        )

    @property
    def optimal_total_reward(self) -> float:
        """最適マッチングの期待報酬和（heterogeneous での最適値）。"""
        return self._optimal_total_reward
