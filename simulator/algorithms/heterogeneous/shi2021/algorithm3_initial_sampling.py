"""BEACON の初期サンプリングフェーズ。"""

from __future__ import annotations

from typing import List

from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner


class BeaconExploreState:
    """
    BEACON エポックループで共有される探索状態。

    Attributes:
        T[k][m]: player m による arm k の exploratory サンプル数
        R[k][m]: player m による arm k の累積報酬
        samples[k][m]: player m による arm k の個別サンプル列（UCB に先頭 2^p 個を使う）
        K: arm 数
        M: プレイヤー数
        leader_pid: leader の player index（0-based）
        last_assigned_arms: 最終エポックの Oracle 割当（HorizonReached 時の取得用）
    """

    def __init__(
        self,
        T: List[List[int]],
        R: List[List[float]],
        samples: List[List[List[float]]],
        K: int,
        M: int,
        leader_pid: int,
    ) -> None:
        self.T = T
        self.R = R
        self.samples = samples
        self.K = K
        self.M = M
        self.leader_pid = leader_pid
        self.last_assigned_arms: List[int] = list(range(M))


class BeaconInitialSamplingMixin:
    """BEACON の初期サンプリングを実行する補助クラス。"""

    def beacon_initial_sampling(
        self,
        runner: HeterogeneousRunner,
        state_list: List[int],
        rank_list: List[int],
    ) -> BeaconExploreState:
        """
        BEACON の初期サンプリングフェーズ（メインループ前の K ステップ）。

        Args:
            runner: HeterogeneousRunner
            state_list: 各プレイヤーの state（0-based）
            rank_list: 各プレイヤーの rank（1-based）

        Returns:
            BeaconExploreState: 初期サンプリング後の状態
        """
        runner.set_phase("beacon_initial_sampling")
        K = self.K
        M = self.M

        T = [[0] * M for _ in range(K)]
        R = [[0.0] * M for _ in range(K)]
        samples: List[List[List[float]]] = [[[] for _ in range(M)] for _ in range(K)]

        leader_pid = next((m for m in range(M) if rank_list[m] == 1), 0)

        for k in range(K):
            actions = []
            for m in range(M):
                if rank_list[m] == 1:
                    actions.append(k)
                else:
                    r_zero = rank_list[m] - 1
                    actions.append((r_zero + k) % K)

            result = runner.step(actions)

            for m in range(M):
                a = actions[m]
                T[a][m] += 1
                R[a][m] += result.rewards[m]
                samples[a][m].append(result.rewards[m])

        return BeaconExploreState(
            T=T, R=R, samples=samples, K=K, M=M, leader_pid=leader_pid
        )
