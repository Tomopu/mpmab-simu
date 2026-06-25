"""collision-sensing 版 ParallelVirtualMusicalChairs。"""

from __future__ import annotations

from typing import List

from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.algorithms.homogeneous.math_helpers import checked_log as _ln
from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner


class Izumi2026CollisionSensingParallelVirtualMusicalChairsMixin:
    """直接観測した衝突フラグで external rank を割り当てる補助クラス。"""

    def parallel_virtual_musical_chairs_collision_sensing(
        self,
        runner: HeterogeneousRunner,
        good_arms: List[int],
    ) -> List[int]:
        """
        直接観測した衝突フラグを使う ParallelVirtualMusicalChairs。

        プレイヤーがチャネル腕を引いて衝突なしを観測した時点で external rank を確定する。
        """
        runner.set_phase("parallel_virtual_musical_chairs_collision_sensing")
        K = self.K
        M = self.M
        n = len(good_arms)
        delta = self.delta

        s_list = [-1] * M
        slots = [[-1] * n for _ in range(M)]

        good_set = set(good_arms)
        non_good_arms = [a for a in range(K) if a not in good_set]
        dummy = non_good_arms[0] if non_good_arms else good_arms[0]

        tau_rank_cs = max(1, _ceil(_ln(1.0 / delta)))
        total_steps = max(1, _ceil(K * tau_rank_cs / max(1, n)))

        for t0 in range(total_steps):
            if t0 % K == 0:
                for m in range(M):
                    if s_list[m] == -1:
                        forbidden_set = set()
                        for i0 in range(n):
                            available = [slot for slot in range(K) if slot not in forbidden_set]
                            if not available:
                                available = list(range(K))
                            slots[m][i0] = self._player_rngs[m].choice(available)
                            forbidden_set = {
                                (slots[m][j0] + (i0 + 1 - j0)) % K
                                for j0 in range(i0 + 1)
                            }
                    else:
                        for i0 in range(n):
                            slots[m][i0] = s_list[m]

            actions: List[int] = []
            pulled_i0_per_player = [-1] * M
            for m in range(M):
                pulled = False
                for i0 in range(n):
                    if slots[m][i0] >= 0 and (t0 + i0) % K == slots[m][i0]:
                        actions.append(good_arms[i0])
                        pulled_i0_per_player[m] = i0
                        pulled = True
                        break
                if not pulled:
                    actions.append(dummy)

            result = runner.step(actions)
            for m in range(M):
                i0 = pulled_i0_per_player[m]
                if i0 >= 0 and s_list[m] == -1 and not result.collisions[m]:
                    s_list[m] = slots[m][i0]
                    for j0 in range(n):
                        slots[m][j0] = s_list[m]

        return s_list
