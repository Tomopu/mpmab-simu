"""
Wang 2020 Rank Assignment。

orthogonalization で得た state から rank と M_hat を推定する。
"""

from __future__ import annotations

from typing import List, Tuple

from simulator.algorithms.heterogeneous.shi2021.runner_hetero import (
    HeterogeneousRunner,
)


class Wang2020RankAssignmentMixin:
    """Wang et al. (2020) の rank assignment を実行する補助クラス。"""

    def rank_assignment(
        self,
        runner: HeterogeneousRunner,
        state_list: List[int],
    ) -> Tuple[List[int], List[int]]:
        """
        Rank Assignment フェーズを M 人同期実行する。

        Args:
            runner: HeterogeneousRunner
            state_list: 各プレイヤーの state（0-based, 未確定は -1）

        Returns:
            (M_hat_list, rank_list):
                M_hat_list: 各プレイヤーの推定プレイヤー数（1 以上）
                rank_list: 各プレイヤーの内部 rank（1-based, leader = 1）
        """
        runner.set_phase("rank_assignment")
        K = self.K
        M = self.M

        M_hat = [1] * M
        rank = [1] * M

        for k in range(K - 1):
            collision_seen_by_others = [False] * M

            for q in range(K - 1):
                actions = []
                for m in range(M):
                    if state_list[m] == k:
                        actions.append(q)
                    else:
                        s = state_list[m] if state_list[m] != -1 else K - 1
                        actions.append(s)

                result = runner.step(actions)

                for m in range(M):
                    if state_list[m] != k and result.collisions[m]:
                        collision_seen_by_others[m] = True

            for m in range(M):
                if collision_seen_by_others[m]:
                    M_hat[m] += 1
                    if state_list[m] != -1 and k < state_list[m]:
                        rank[m] += 1

        return M_hat, rank
