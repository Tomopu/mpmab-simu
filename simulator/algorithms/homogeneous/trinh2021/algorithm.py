"""
Randomized Selfish KL-UCB（Trinh and Combes, 2021）の Runner 版。

他のアルゴリズムと同じく Runner.step() を通して環境と対話する。
1 ステップずつ Trace に記録するため、T が大きい比較実験では batch.py の高速版を使う。
行動選択は batch.py と同じ policy.select_actions を使う。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from simulator.algorithms.base import BaseAlgorithm
from simulator.algorithms.homogeneous.trinh2021.batch import trial_generators
from simulator.algorithms.homogeneous.trinh2021.policy import select_actions
from simulator.core.runner import HorizonReached, Runner


class HomogeneousRandomizedSelfishKLUCB(BaseAlgorithm):
    """
    Randomized Selfish KL-UCB を M 人分同期して実行する。

    各プレイヤーは協調も通信もせず、自分の観測だけで KL-UCB 指数に小さな正規乱数を足した
    argmax を選ぶ。割当を確定する手続きはないので、horizon まで動き続ける。

    Args:
        K: arm 数
        M: プレイヤー数
        seed: 行動選択の乱数 seed（batch.py と同じ系列を使う）
        c: 探索関数 f(t) = log t + c log log t の係数（既定 0）
    """

    def __init__(self, K: int, M: int, seed: Optional[int] = None, c: float = 0.0) -> None:
        if K < 1 or M < 1 or M > K:
            raise ValueError("1 <= M <= K が必要。")
        self.K = K
        self.M = M
        self.c = c
        self._policy_rng, _ = trial_generators(0 if seed is None else seed)

    def run(self, runner: Runner) -> Dict[str, object]:
        """
        horizon に達するまで行動選択と統計の更新を繰り返す。

        Args:
            runner: Runner インスタンス

        Returns:
            {"counts": 各 (player, arm) の選択回数, "reward_sums": 報酬の合計, "last_actions": 最後の行動}
        """
        runner.set_phase("randomized_selfish_klucb")
        counts = np.zeros((self.M, self.K))
        reward_sums = np.zeros((self.M, self.K))
        prev_index = None
        last_actions = [-1] * self.M
        t = 0
        while True:
            t += 1
            noise = self._policy_rng.standard_normal((self.M, self.K))
            actions, prev_index = select_actions(counts, reward_sums, t, noise, prev_index, self.c)
            try:
                result = runner.step([int(a) for a in actions])
            except HorizonReached:
                break
            # 観測した報酬だけで統計を更新する（result.collisions は使わない）
            for m, a in enumerate(result.actions):
                counts[m, a] += 1.0
                reward_sums[m, a] += result.rewards[m]
            last_actions = list(result.actions)
        return {"counts": counts, "reward_sums": reward_sums, "last_actions": last_actions}
