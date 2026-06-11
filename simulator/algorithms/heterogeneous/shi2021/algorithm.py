"""
HeterogeneousShiBeacon2021: Shi et al. (2021) BEACON の実装。

Heterogeneous MPMAB with Collision Sensing に対応するアルゴリズム。
初期化フェーズに Wang et al. (2020) の orthogonalization + rank assignment を使い、
学習フェーズに BEACON Leader/Follower エポックループを実行する。

使い方:
    env = HeterogeneousMPMABEnv(means=means_matrix, collision_sensing=True)
    runner = HeterogeneousRunner(env, horizon=T)
    algo = HeterogeneousShiBeacon2021(K=K, M=M, seed=42)
    result = algo.run(runner)

参考文献:
    Shi et al. (2021) Heterogeneous Multi-player Multi-armed Bandits: Closing the Gap
        and Generalization. NeurIPS 2021.
    Wang et al. (2020) An Optimal Algorithm for Multiplayer Multi-Armed Bandits.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from simulator.algorithms.heterogeneous.shi2021.runner_hetero import (
    HeterogeneousRunner,
    HorizonReachedHetero,
)
from simulator.algorithms.heterogeneous.shi2021.wang2020_orthogonalization import (
    Wang2020OrthogonalizationMixin,
)
from simulator.algorithms.heterogeneous.shi2021.shi2021_beacon_epoch import (
    BeaconEpochMixin,
    BeaconExploreState,
)
from simulator.algorithms.heterogeneous.shi2021.results import build_beacon_result


class HeterogeneousShiBeacon2021(
    Wang2020OrthogonalizationMixin,
    BeaconEpochMixin,
):
    """
    Shi et al. (2021) BEACON の同期シミュレーション実装。

    各フェーズ:
        1. Orthogonalization (Wang 2020): 各プレイヤーが一意な state を取得
        2. Rank Assignment (Wang 2020): state から rank と M_hat を推定
        3. BEACON Initial Sampling: 全プレイヤーが全 arm を一通り探索
        4. BEACON Epoch Loop: 通信フェーズ + 探索フェーズを繰り返す

    HorizonReached の扱い:
        各フェーズを個別に try/except で囲み、horizon に達した時点で
        取得済みの状態をそのまま返す。未完了フェーズの出力は初期値のまま。

    Args:
        K: arm の数
        M: プレイヤー数（真値）
        seed: 乱数シード（再現性のため）
        max_ortho_blocks: orthogonalization の最大ブロック数（0 = 無制限）
    """

    def __init__(
        self,
        K: int,
        M: int,
        seed: Optional[int] = None,
        max_ortho_blocks: int = 0,
    ) -> None:
        if K < M + 1:
            # state {0,...,K-2} が M 個必要 + broadcast arm 1 個で K >= M+1
            raise ValueError(
                f"K={K} は M+1={M + 1} 以上でなければならない（state {M} 個 + broadcast arm 1 個）。"
            )

        self.K = K
        self.M = M
        self.max_ortho_blocks = max_ortho_blocks

        # プレイヤーごとに独立な乱数生成器を派生させる
        master_rng = random.Random(seed)
        self._player_rngs: List[random.Random] = [
            random.Random(master_rng.randrange(1 << 30)) for _ in range(M)
        ]

    def run(self, runner: HeterogeneousRunner) -> Dict[str, object]:
        """
        BEACON を実行する。

        手順:
            1. Orthogonalization: 各プレイヤーに state s を割り当てる
            2. Rank Assignment: state から rank j と M_hat を推定する
            3. Initial Sampling: 各プレイヤーが全 arm を一通り探索する
            4. Epoch Loop: 通信 + 探索を繰り返す（horizon まで）

        Args:
            runner: HeterogeneousRunner インスタンス

        Returns:
            {
                "player_states": List[BeaconPlayerState],  各プレイヤーの最終状態
                "phase_durations": Dict[str, int],         フェーズごとの所要ステップ数
            }
        """
        M = self.M

        # フェーズ間で保持する状態（HorizonReachedHetero 時の部分結果）
        state_list = [-1] * M
        rank_list = [1] * M
        M_hat_list = [1] * M
        assigned_arms = list(range(M))  # デフォルト: rank 順に arm を割当

        # 1. Orthogonalization
        try:
            state_list = self.orthogonalization(
                runner, max_blocks=self.max_ortho_blocks
            )
        except HorizonReachedHetero:
            return build_beacon_result(
                M, state_list, rank_list, M_hat_list, assigned_arms, runner
            )

        # 未確定の state を broadcast arm (K-1) に固定（フォールバック）
        for m in range(M):
            if state_list[m] == -1:
                state_list[m] = self.K - 1

        # 2. Rank Assignment
        try:
            M_hat_list, rank_list = self.rank_assignment(runner, state_list)
        except HorizonReachedHetero:
            return build_beacon_result(
                M, state_list, rank_list, M_hat_list, assigned_arms, runner
            )

        # 3. Initial Sampling
        explore_state: Optional[BeaconExploreState] = None
        try:
            explore_state = self.beacon_initial_sampling(runner, state_list, rank_list)
        except HorizonReachedHetero:
            return build_beacon_result(
                M, state_list, rank_list, M_hat_list, assigned_arms, runner
            )

        # 4. Epoch Loop
        try:
            assigned_arms = self.beacon_run_epochs(
                runner, state_list, rank_list, explore_state
            )
        except HorizonReachedHetero:
            # horizon 到達時は last_assigned_arms を使う
            assigned_arms = explore_state.last_assigned_arms

        return build_beacon_result(
            M, state_list, rank_list, M_hat_list, assigned_arms, runner
        )
