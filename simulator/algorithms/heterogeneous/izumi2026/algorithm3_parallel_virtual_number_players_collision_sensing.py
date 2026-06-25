"""collision-sensing 版 ParallelVirtualNumberPlayers。"""

from __future__ import annotations

from typing import List, Tuple

from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner


class Izumi2026CollisionSensingParallelVirtualNumberPlayersMixin:
    """直接観測した衝突フラグで M_hat と internal rank を推定する補助クラス。"""

    def parallel_virtual_number_players_collision_sensing(
        self,
        runner: HeterogeneousRunner,
        good_arms: List[int],
        s_list: List[int],
    ) -> Tuple[List[int], List[int]]:
        """
        直接観測した衝突フラグを使う ParallelVirtualNumberPlayers。

        各テストを 1 回だけ実行し、観測した衝突フラグを直接使う。
        """
        runner.set_phase("parallel_virtual_number_players_collision_sensing")
        K = self.K
        M = self.M
        n = len(good_arms)

        M_hat = [1] * M
        j = [1] * M

        good_set = set(good_arms)
        non_good_arms = [a for a in range(K) if a not in good_set]
        dummy = non_good_arms[0] if non_good_arms else good_arms[0]

        H = _ceil(2 * K / n)
        for h_0 in range(H):
            ell = [[-1] * n for _ in range(M)]
            for m in range(M):
                s_1based = s_list[m] + 1
                for i0 in range(n):
                    v = h_0 * n + i0 + 1
                    if v > 2 * K:
                        continue
                    if v > 2 * s_1based:
                        ell[m][i0] = (v - s_1based - 1) % K
                    else:
                        ell[m][i0] = s_list[m]

            for k_0 in range(K):
                actions: List[int] = []
                matched_i0_per_player = [-1] * M
                for m in range(M):
                    found = False
                    for i0 in range(n):
                        if ell[m][i0] != -1 and (ell[m][i0] + i0) % K == k_0:
                            matched_i0_per_player[m] = i0
                            found = True
                            break
                    actions.append(good_arms[matched_i0_per_player[m]] if found else dummy)

                result = runner.step(actions)
                for m in range(M):
                    i0 = matched_i0_per_player[m]
                    if i0 >= 0 and result.collisions[m]:
                        M_hat[m] += 1
                        v_collision = h_0 * n + i0 + 1
                        s_1based = s_list[m] + 1
                        if v_collision <= 2 * s_1based:
                            j[m] += 1

        return M_hat, j
