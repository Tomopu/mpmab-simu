"""ParallelBEACON の初期サンプリングフェーズ。"""

from __future__ import annotations

from typing import Dict, List, Optional

from simulator.algorithms.heterogeneous.izumi2026.helpers import build_group_map
from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner


class ParallelBeaconExploreState:
    """
    ParallelBEACON エポックループで共有される探索状態。

    Attributes:
        T[k][m]: player m による arm k の exploratory サンプル数
        R[k][m]: player m による arm k の累積報酬
        samples[k][m]: player m による arm k の個別サンプル列
        K, M, n: arm 数・プレイヤー数・good arm 数
        good_arms: 通信チャネルとして使う arm の 0-based index リスト（長さ n）
        group_map: group_map[g] = そのグループに所属するプレイヤー index リスト (1-based g)
        last_assigned_arms: 最終エポックの Oracle 割当（HorizonReached 保険）
    """

    def __init__(
        self,
        T: List[List[int]],
        R: List[List[float]],
        samples: List[List[List[float]]],
        K: int,
        M: int,
        n: int,
        good_arms: List[int],
        rank_list: List[int],
        state_arms: Optional[List[int]] = None,
    ) -> None:
        self.T = T
        self.R = R
        self.samples = samples
        self.K = K
        self.M = M
        self.n = n
        self.good_arms = good_arms
        self.rank_list = rank_list
        # state_arms[m]: player m が通信中 idle のときに pull する arm（CSVMC の s_list から設定）。
        # 全員が異なる arm を持つため collision が起きず、non-zero reward が発生する。
        # None の場合は player index ベースのフォールバックを使う。
        self.state_arms: List[int] = state_arms if state_arms is not None else list(range(M))

        self.group_map: Dict[int, List[int]] = build_group_map(rank_list, n)
        self.grand_leader_pid: int = next(
            (m for m in range(M) if rank_list[m] == 1), 0
        )
        self.subleader_pids: Dict[int, int] = {
            g: next((m for m in range(M) if rank_list[m] == g), -1)
            for g in range(2, n + 1)
        }

        self.last_assigned_arms: List[int] = list(range(M))


class ParallelBeaconInitialSamplingMixin:
    """ParallelBEACON の初期サンプリングを実行する補助クラス。"""

    def parallel_beacon_initial_sampling(
        self,
        runner: HeterogeneousRunner,
        good_arms: List[int],
        rank_list: List[int],
        s_list: Optional[List[int]] = None,
    ) -> ParallelBeaconExploreState:
        """
        ParallelBEACON の初期サンプリングフェーズ（K ステップ）。

        Args:
            runner: HeterogeneousRunner
            good_arms: FindMultipleGoodArms の出力 G（0-based, 長さ n）
            rank_list: 各プレイヤーの rank（1-based）
            s_list: CSVMC が返す external rank（各プレイヤーの通信 idle arm）

        Returns:
            ParallelBeaconExploreState
        """
        runner.set_phase("parallel_beacon_initial_sampling")
        K = self.K
        M = self.M
        n = len(good_arms)

        T = [[0] * M for _ in range(K)]
        R = [[0.0] * M for _ in range(K)]
        samples: List[List[List[float]]] = [[[] for _ in range(M)] for _ in range(K)]

        for k in range(K):
            actions = []
            for m in range(M):
                actions.append((rank_list[m] - 1 + k) % K)
            result = runner.step(actions)
            for m in range(M):
                a = actions[m]
                T[a][m] += 1
                R[a][m] += result.rewards[m]
                samples[a][m].append(result.rewards[m])

        return ParallelBeaconExploreState(
            T=T, R=R, samples=samples,
            K=K, M=M, n=n,
            good_arms=good_arms,
            rank_list=rank_list,
            state_arms=s_list,
        )
