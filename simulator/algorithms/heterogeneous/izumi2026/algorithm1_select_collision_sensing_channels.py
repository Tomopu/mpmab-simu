"""collision-sensing 版の通信チャネル選択。"""

from __future__ import annotations

from typing import List, Tuple

from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner


class Izumi2026SelectCollisionSensingChannelsMixin:
    """n 本の通信チャネルを選択する補助クラス。"""

    def select_collision_sensing_channels(
        self,
        runner: HeterogeneousRunner,
    ) -> Tuple[List[int], dict[int, float]]:
        """
        n 本の通信チャネルを選択する。

        collision-sensing 設定では任意の腕で衝突ビットを伝送できるため、
        このフェーズにサンプリングコストはかからない。
        """
        runner.set_phase("select_collision_sensing_channels")
        channels = list(range(self.n))
        return channels, {k: 1.0 for k in channels}
