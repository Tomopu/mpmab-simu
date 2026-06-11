"""
HeterogeneousMultiChannelIzumi2026: Izumi et al. (2026) の Heterogeneous 拡張実装。

Heterogeneous Multi-Channel MPMAB with Collision Sensing のモデル。

初期化フェーズ:
    Izumi et al. (2026) の no-sensing 版アルゴリズムを collision sensing 環境に適用する。
    FindMultipleGoodArms / ParallelVMC / ParallelVNP は collision flag を参照せず rewards のみ使う。

学習フェーズ:
    ParallelBEACON = Shi et al. (2021) BEACON を n チャネル・階層グループ構造に拡張したもの。
    grand leader (j=1) が Oracle で最適マッチングを決定し、
    sub-leader (j=2..n) が階層的に割当を中継する。

実装上の注意:
    - 初期化 mixin (Izumi 2026 homogeneous) は Runner 型を期待するが、
      HeterogeneousRunner も同じインターフェース（step, set_phase, elapsed, trace）を持つため
      Python のダックタイピングで動作する。
    - 初期化 mixin が発生させる HorizonReached は内部で catch されないため、
      HeterogeneousRunner が raise する HorizonReachedHetero がそのまま
      run() の except に到達する。

使い方:
    env = HeterogeneousMPMABEnv(means=means_matrix, collision_sensing=True)
    runner = HeterogeneousRunner(env, horizon=T)
    algo = HeterogeneousMultiChannelIzumi2026(K=K, M=M, n=2, delta=1e-3, seed=42)
    result = algo.run(runner)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from simulator.algorithms.heterogeneous.izumi2026.parallel_beacon_epoch import (
    ParallelBeaconEpochMixin,
    ParallelBeaconExploreState,
)
from simulator.algorithms.heterogeneous.izumi2026.results import build_parallel_beacon_result
from simulator.algorithms.heterogeneous.shi2021.runner_hetero import (
    HeterogeneousRunner,
    HorizonReachedHetero,
)
from simulator.algorithms.homogeneous.izumi2026.algorithm1_find_multiple_good_arms import (
    Izumi2026FindMultipleGoodArmsMixin,
)
from simulator.algorithms.homogeneous.izumi2026.algorithm2_parallel_virtual_musical_chairs import (
    Izumi2026ParallelVirtualMusicalChairsMixin,
)
from simulator.algorithms.homogeneous.izumi2026.algorithm3_parallel_virtual_number_players import (
    Izumi2026ParallelVirtualNumberPlayersMixin,
)
from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.algorithms.homogeneous.math_helpers import checked_log as _ln


class HeterogeneousMultiChannelIzumi2026(
    Izumi2026FindMultipleGoodArmsMixin,
    Izumi2026ParallelVirtualMusicalChairsMixin,
    Izumi2026ParallelVirtualNumberPlayersMixin,
    ParallelBeaconEpochMixin,
):
    """
    Izumi et al. (2026) Heterogeneous Multi-Channel MPMAB の実装。

    初期化フェーズに Izumi 2026 の並列初期化（FindMultipleGoodArms + ParallelVMC +
    ParallelVNP）を使い、学習フェーズに ParallelBEACON（BEACON の n チャネル拡張）
    を使う。

    Args:
        K: arm の数
        M: プレイヤー数（真値）
        n: good arm の数（通信チャネル数 = グループ数）。n < K-M かつ n >= 1。
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
            raise ValueError(
                f"n < K-M が必要（good arm を探索する余地が必要）。n={n}, K-M={K - M}"
            )
        if not (0.0 < delta < 1.0):
            raise ValueError("delta は (0, 1) の範囲でなければならない。")

        self.K = K
        self.M = M
        self.n = n
        self.delta = delta

        # プレイヤーごとに独立な乱数生成器
        master_rng = random.Random(seed)
        self._player_rngs: List[random.Random] = [
            random.Random(master_rng.randrange(1 << 30)) for _ in range(M)
        ]

    def run(self, runner: HeterogeneousRunner) -> Dict[str, object]:
        """
        ParallelBEACON (Izumi 2026 Heterogeneous) を実行する。

        手順:
            1. FindMultipleGoodArms: n 本の good arm G と下界 mu_tilde を取得
            2. tau_rank = ceil(K * ln(1/delta) / mu_min) を計算
            3. tau_comm = ceil(ln(1/delta) / mu_min) を計算
            4. ParallelVirtualMusicalChairs: external rank s を割り当てる
            5. ParallelVirtualNumberPlayers: internal rank j と M_hat を推定する
            6. ParallelBEACON Initial Sampling: 全 arm を一通り探索する
            7. ParallelBEACON Epoch Loop: 通信 + 探索を繰り返す（horizon まで）

        Args:
            runner: HeterogeneousRunner インスタンス

        Returns:
            {
                "player_states": List[ParallelBeaconPlayerState],
                "phase_durations": Dict[str, int],
            }
        """
        M = self.M
        delta = self.delta

        # フェーズ間で保持する状態
        good_arms: List[int] = []
        mu_tilde_map: Dict[int, float] = {}
        s_list = [-1] * M
        j_list = [1] * M
        M_hat_list = [1] * M
        assigned_arms = list(range(M))

        # 1. FindMultipleGoodArms
        try:
            good_arms, mu_tilde_map = self.find_multiple_good_arms(runner)
        except HorizonReachedHetero:
            return build_parallel_beacon_result(
                M, self.n, good_arms, s_list, j_list, M_hat_list, assigned_arms, runner
            )

        if not good_arms:
            return build_parallel_beacon_result(
                M, self.n, good_arms, s_list, j_list, M_hat_list, assigned_arms, runner
            )

        # tau の計算
        mu_min = min(mu_tilde_map.get(k, 1e-12) for k in good_arms)
        mu_safe = max(mu_min, 1e-12)
        tau_rank = _ceil(self.K * _ln(1.0 / delta) / mu_safe)
        tau_comm = _ceil(_ln(1.0 / delta) / mu_safe)

        # 2. ParallelVirtualMusicalChairs
        try:
            s_list = self.parallel_virtual_musical_chairs(runner, good_arms, tau_rank)
        except HorizonReachedHetero:
            return build_parallel_beacon_result(
                M, self.n, good_arms, s_list, j_list, M_hat_list, assigned_arms, runner
            )

        # s が未確定のまま残った場合は 0 にフォールバックする
        s_list = [s if s >= 0 else 0 for s in s_list]

        # 3. ParallelVirtualNumberPlayers
        try:
            M_hat_list, j_list = self.parallel_virtual_number_players(
                runner, good_arms, s_list, tau_comm
            )
        except HorizonReachedHetero:
            return build_parallel_beacon_result(
                M, self.n, good_arms, s_list, j_list, M_hat_list, assigned_arms, runner
            )

        # 4. ParallelBEACON Initial Sampling
        explore_state: Optional[ParallelBeaconExploreState] = None
        try:
            explore_state = self.parallel_beacon_initial_sampling(
                runner, good_arms, j_list
            )
        except HorizonReachedHetero:
            return build_parallel_beacon_result(
                M, self.n, good_arms, s_list, j_list, M_hat_list, assigned_arms, runner
            )

        # 5. ParallelBEACON Epoch Loop
        try:
            assigned_arms = self.parallel_beacon_run_epochs(runner, explore_state)
        except HorizonReachedHetero:
            assigned_arms = explore_state.last_assigned_arms

        return build_parallel_beacon_result(
            M, self.n, good_arms, s_list, j_list, M_hat_list, assigned_arms, runner
        )
