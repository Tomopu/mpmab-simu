from __future__ import annotations

import random
from typing import Dict, List, Optional

from simulator.algorithms.homogeneous.izumi2026.algorithm1_find_multiple_good_arms import (
    Izumi2026FindMultipleGoodArmsMixin,
)
from simulator.algorithms.homogeneous.izumi2026.algorithm2_parallel_virtual_musical_chairs import (
    Izumi2026ParallelVirtualMusicalChairsMixin,
)
from simulator.algorithms.homogeneous.izumi2026.algorithm3_parallel_virtual_number_players import (
    Izumi2026ParallelVirtualNumberPlayersMixin,
)
from simulator.algorithms.homogeneous.izumi2026.algorithm4_hierarchical_distributed_exploration import (
    Izumi2026HierarchicalDistributedExplorationMixin,
)
from simulator.algorithms.homogeneous.izumi2026.results import build_result
from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.algorithms.homogeneous.math_helpers import checked_log as _ln
from simulator.core.runner import HorizonReached, Runner


class HomogeneousMultiChannelIzumi2026(
    Izumi2026FindMultipleGoodArmsMixin,
    Izumi2026ParallelVirtualMusicalChairsMixin,
    Izumi2026ParallelVirtualNumberPlayersMixin,
    Izumi2026HierarchicalDistributedExplorationMixin,
):
    """
    Izumi et al. (2026) Homogeneous Multi-Channel MPMAB without Collision Sensing の実装。

    使い方:
        algo = HomogeneousMultiChannelIzumi2026(K=5, M=2, n=2, delta=1e-3, seed=42)
        result = algo.run(runner)

    Args:
        K: 腕の数
        M: プレイヤー数（真値）
        n: FindMultipleGoodArms で探す good arm の数。n < K-M が必要。
        delta: 信頼度パラメータ δ
        seed: 乱数シード（再現性のため）
    """

    def __init__(
        self,
        K: int,
        M: int,
        n: int,
        delta: float,
        seed: Optional[int] = None,
    ) -> None:
        if K < 2:
            raise ValueError("K は 2 以上でなければならない。")
        if M < 1:
            raise ValueError("M は 1 以上でなければならない。")
        if n < 1:
            raise ValueError("n は 1 以上でなければならない。")
        if n >= K - M:
            raise ValueError(f"n < K-M が必要。n={n}, K-M={K - M}")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta は (0, 1) の範囲でなければならない。")

        self.K = K
        self.M = M
        self.n = n
        self.delta = delta

        # プレイヤーごとに独立な乱数生成器を派生させる
        master_rng = random.Random(seed)
        self._player_rngs: List[random.Random] = [random.Random(master_rng.randrange(1 << 30)) for _ in range(M)]

    def run(self, runner: Runner) -> Dict[str, object]:
        """
        Algorithm 5 ProposedParallelAlgorithm を実行する。

        手順:
            1. FindMultipleGoodArms で n 本の good arm G と下界 mu_tilde を得る。
            2. tau = ceil(ln(1/delta) / min(mu_tilde))
            3. ParallelVirtualMusicalChairs で各プレイヤーに external rank s を割り当てる。
            4. ParallelVirtualNumberPlayers で内部 rank j と推定プレイヤー数 M_hat を得る。
            5. HierarchicalDistributedExploration で各プレイヤーに top-M arm を割り当てる。

        HorizonReached について:
            各フェーズを個別に try/except で囲み、取得済み状態を保持する。
            horizon に達した時点でそれ以前に完了したフェーズの結果を返す。

        Args:
            runner: Runner インスタンス

        Returns:
            {
                "player_states": List[PlayerStateIzumi],
                "phase_durations": Dict[str, int],
            }
        """
        M = self.M
        delta = self.delta

        # フェーズ間で保持する状態（HorizonReached 時の部分結果）
        good_arms: List[int] = []
        mu_tilde_map: Dict[int, float] = {}
        s_list = [-1] * M
        j_list = [1] * M
        M_hat_list = [1] * M
        f_list = [-1] * M

        # 1. FindMultipleGoodArms
        try:
            good_arms, mu_tilde_map = self.find_multiple_good_arms(runner)
        except HorizonReached:
            pass

        if not good_arms:
            # FindMultipleGoodArms すら完了しなかった場合はそのまま返す
            return build_result(M, good_arms, mu_tilde_map, s_list, j_list, M_hat_list, f_list, runner)

        # tau = ceil(ln(1/delta) / min(mu_tilde)) の計算
        # 論文: tilde_mu_min <- min_{k in G} tilde_mu[k]
        mu_min = min(mu_tilde_map.get(k, 1e-12) for k in good_arms)
        mu_safe = max(mu_min, 1e-12)
        tau = _ceil(_ln(1.0 / delta) / mu_safe)

        # 2. ParallelVirtualMusicalChairs
        try:
            s_list = self.parallel_virtual_musical_chairs(runner, good_arms, tau)
        except HorizonReached:
            pass

        # 3. ParallelVirtualNumberPlayers
        try:
            M_hat_list, j_list = self.parallel_virtual_number_players(runner, good_arms, s_list, tau)
        except HorizonReached:
            pass

        # 4. HierarchicalDistributedExploration
        try:
            f_list = self.hierarchical_distributed_exploration(runner, good_arms, j_list, M_hat_list, tau)
        except HorizonReached:
            pass

        return build_result(M, good_arms, mu_tilde_map, s_list, j_list, M_hat_list, f_list, runner)
