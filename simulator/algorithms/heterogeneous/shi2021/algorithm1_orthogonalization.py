"""
Wang 2020 Orthogonalization。

BEACON の初期化フェーズとして使う。
collision-sensing 設定を利用してプレイヤーごとに一意な state を割り当てる。
"""

from __future__ import annotations

from typing import List

from simulator.algorithms.heterogeneous.shi2021.runner_hetero import (
    HeterogeneousRunner,
)


class Wang2020OrthogonalizationMixin:
    """Wang et al. (2020) の orthogonalization procedure を実行する補助クラス。"""

    def orthogonalization(
        self,
        runner: HeterogeneousRunner,
        max_blocks: int = 0,
    ) -> List[int]:
        """
        Orthogonalization フェーズを M 人同期実行する。

        各プレイヤーが衝突なしで arm を確保できるまで繰り返すことで、
        一意な外部 state を割り当てる。

        論文の変数対応:
            state (0-based) {0,...,K-2} ← 論文 1-based {1,...,K-1}
            broadcast arm (0-based) K-1 ← 論文 1-based arm K
            ブロックは「選択ラウンド 1 + broadcast ラウンド K」= K+1 ラウンド

        Args:
            runner: HeterogeneousRunner
            max_blocks: 最大ブロック数。0 なら制限なし（horizon に委ねる）。

        Returns:
            state_list: 各プレイヤーの state（0-based, 未確定は -1）
        """
        runner.set_phase("orthogonalization")
        K = self.K
        M = self.M

        broadcast_arm = K - 1
        num_states = K - 1

        state_list = [-1] * M

        block = 0
        while True:
            block += 1
            if max_blocks > 0 and block > max_blocks:
                break

            actions = []
            candidates = []
            for m in range(M):
                if state_list[m] == -1:
                    c = self._player_rngs[m].randrange(num_states)
                    candidates.append(c)
                    actions.append(c)
                else:
                    candidates.append(state_list[m])
                    actions.append(state_list[m])

            result = runner.step(actions)

            for m in range(M):
                if state_list[m] == -1 and not result.collisions[m]:
                    state_list[m] = candidates[m]

            any_collision_in_broadcast = False
            for q in range(K):
                actions = []
                for m in range(M):
                    if state_list[m] == -1:
                        actions.append(broadcast_arm)
                    elif q == state_list[m]:
                        actions.append(broadcast_arm)
                    else:
                        actions.append(state_list[m])

                result = runner.step(actions)

                if any(result.collisions):
                    any_collision_in_broadcast = True

            if not any_collision_in_broadcast:
                break

        return state_list
