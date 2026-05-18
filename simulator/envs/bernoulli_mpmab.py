"""
BernoulliMPMABEnv: Bernoulli Multi-Player Multi-Armed Bandit 環境。

同期シミュレーションを前提に、全プレイヤーの action を同時に受け取り、
collision 判定と報酬生成を行う。

論文との対応:
- homogeneous 設定: means は shape (K,) の 1 次元配列。全プレイヤーが同じ報酬分布を持つ。
- collision: 2 人以上が同じ arm を選んだ場合、関与した全プレイヤーの報酬は 0。
- no-sensing: プレイヤーは collision の有無を観測できない（rewards のみ観測）。
- optimal_total_reward: top-M arm の平均報酬和（homogeneous 設定での最適値）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class StepResult:
    """
    1 ステップの結果を保持するデータクラス。

    Attributes:
        actions: 各プレイヤーが選んだ arm index（0-based, 長さ M）
        rewards: 各プレイヤーが得た報酬（0/1, 長さ M）
        collisions: 各プレイヤーの collision フラグ（True/False, 長さ M）
            ※ no-sensing アルゴリズムには渡さない。環境と評価指標での記録専用。
        total_reward: このステップの報酬合計
        optimal_total_reward: top-M arm の平均報酬和（各ステップの最適値）
        instant_regret: optimal_total_reward - total_reward
    """

    actions: List[int]
    rewards: List[float]
    collisions: List[bool]
    total_reward: float
    optimal_total_reward: float
    instant_regret: float


class BernoulliMPMABEnv:
    """
    Bernoulli 報酬を持つ Multi-Player Multi-Armed Bandit 環境。

    各時刻に M 人のプレイヤーが同時に arm を 1 本選ぶ。
    - 同じ arm を 2 人以上が選んだ場合（collision）: 関与全員の報酬 = 0
    - collision がない場合: Bernoulli(means[k]) の報酬

    Args:
        means: 各 arm の平均報酬（shape (K,), 0-based index）
        num_players: プレイヤー数 M
        collision_sensing: False なら collision を観測不可（no-sensing モード）。
            True でも rewards には影響しない（collision は StepResult.collisions に記録）。
        seed: 乱数シード
    """

    def __init__(
        self,
        means: Sequence[float],
        num_players: int,
        collision_sensing: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.means = list(means)
        self.K = len(self.means)
        self.M = num_players
        self.collision_sensing = collision_sensing
        self._seed = seed
        self.rng = random.Random(seed)
        self.t = 0

        if self.K < 1:
            raise ValueError("K (number of arms) must be >= 1")
        if self.M < 1:
            raise ValueError("num_players must be >= 1")

        # top-M arm の平均報酬和を事前計算（homogeneous での最適値）
        sorted_means = sorted(self.means, reverse=True)
        self._optimal_total_reward = float(sum(sorted_means[: self.M]))

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

    def step(self, actions: List[int]) -> StepResult:
        """
        1 ステップ進める。全プレイヤーの action を同時に受け取り、結果を返す。

        Args:
            actions: 各プレイヤーが選んだ arm index（0-based, 長さ M）

        Returns:
            StepResult: このステップの結果

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

        # 3. 各プレイヤーの報酬を生成
        rewards: List[float] = []
        for i, a in enumerate(actions):
            if collisions[i]:
                # collision: 報酬 0
                rewards.append(0.0)
            else:
                # collision なし: Bernoulli(means[a]) をサンプリング
                r = 1.0 if self.rng.random() < self.means[a] else 0.0
                rewards.append(r)

        total_reward = sum(rewards)
        instant_regret = self._optimal_total_reward - total_reward

        return StepResult(
            actions=list(actions),
            rewards=rewards,
            collisions=collisions,
            total_reward=total_reward,
            optimal_total_reward=self._optimal_total_reward,
            instant_regret=instant_regret,
        )

    @property
    def optimal_total_reward(self) -> float:
        """top-M arm の平均報酬和（homogeneous での最適値）。"""
        return self._optimal_total_reward
