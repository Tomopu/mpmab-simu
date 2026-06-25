"""
HeterogeneousMultiChannelIzumi2026: Izumi et al. (2026) の Heterogeneous 拡張実装。

Heterogeneous Multi-Channel MPMAB with Collision Sensing のモデル。

初期化フェーズ:
    collision sensing 環境では衝突を 1 回の pull で観測できるため、
    no-sensing 版の「正報酬が出るまで繰り返す」判定を collision flag 判定で置き換える。
    通信用 arm は正報酬 arm である必要がないため、FindMultipleGoodArms は
    sampling なしの通信チャネル選択に簡略化する。

学習フェーズ:
    ParallelBEACON = Shi et al. (2021) BEACON を n チャネル・階層グループ構造に拡張したもの。
    grand leader (j=1) が Oracle で最適マッチングを決定し、
    sub-leader (j=2..n) が階層的に割当を中継する。

実装上の注意:
    - 初期化は Izumi2026CollisionSensingInitializationMixin が担当し、
      HeterogeneousRunner のインターフェース（step, set_phase, elapsed, trace）を使う。
    - 初期化補助クラスが発生させる HorizonReachedHetero は内部で catch されないため、
      run() の except HorizonReachedHetero に直接伝播する。

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
from simulator.algorithms.heterogeneous.izumi2026.collision_sensing_initialization import (
    Izumi2026CollisionSensingInitializationMixin,
)
from simulator.algorithms.heterogeneous.izumi2026.results import build_parallel_beacon_result
from simulator.algorithms.heterogeneous.shi2021.runner_hetero import (
    HeterogeneousRunner,
    HorizonReachedHetero,
)


class HeterogeneousMultiChannelIzumi2026(
    Izumi2026CollisionSensingInitializationMixin,
    ParallelBeaconEpochMixin,
):
    """
    Izumi et al. (2026) Heterogeneous Multi-Channel MPMAB の実装。

    初期化フェーズに collision-sensing 版の並列初期化（通信チャネル選択 +
    ParallelVMC + ParallelVNP）を使い、学習フェーズに ParallelBEACON
    （BEACON の n チャネル拡張）を使う。

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
            1. n 本の通信チャネル G を選択する
            2. collision-sensing ParallelVirtualMusicalChairs で external rank s を割り当てる
            3. collision-sensing ParallelVirtualNumberPlayers で internal rank j と M_hat を推定する
            4. ParallelBEACON Initial Sampling: 全 arm を一通り探索する
            5. ParallelBEACON Epoch Loop: 通信 + 探索を繰り返す（horizon まで）

        Args:
            runner: HeterogeneousRunner インスタンス

        Returns:
            {
                "player_states": List[ParallelBeaconPlayerState],
                "phase_durations": Dict[str, int],
            }
        """
        M = self.M
        # フェーズ間で保持する状態
        good_arms: List[int] = []
        s_list = [-1] * M
        j_list = [1] * M
        M_hat_list = [1] * M
        assigned_arms = list(range(M))

        # 1. 通信チャネル選択
        try:
            good_arms, _ = self.select_collision_sensing_channels(runner)
        except HorizonReachedHetero:
            return build_parallel_beacon_result(
                M, self.n, good_arms, s_list, j_list, M_hat_list, assigned_arms, runner
            )

        if not good_arms:
            return build_parallel_beacon_result(
                M, self.n, good_arms, s_list, j_list, M_hat_list, assigned_arms, runner
            )

        # 2. collision-sensing ParallelVirtualMusicalChairs
        try:
            s_list = self.parallel_virtual_musical_chairs_collision_sensing(
                runner, good_arms
            )
        except HorizonReachedHetero:
            return build_parallel_beacon_result(
                M, self.n, good_arms, s_list, j_list, M_hat_list, assigned_arms, runner
            )

        # s が未確定のまま残った場合は 0 にフォールバックする
        s_list = [s if s >= 0 else 0 for s in s_list]

        # 3. collision-sensing ParallelVirtualNumberPlayers
        try:
            M_hat_list, j_list = self.parallel_virtual_number_players_collision_sensing(
                runner, good_arms, s_list
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
