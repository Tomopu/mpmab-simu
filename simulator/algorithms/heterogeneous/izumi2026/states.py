"""ParallelBEACON (Izumi 2026 Heterogeneous) で使うプレイヤー状態の dataclass。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ParallelBeaconPlayerState:
    """
    各プレイヤーの初期化フェーズ終了時の状態サマリー（ParallelBEACON 版）。

    Attributes:
        external_rank_s: ParallelVirtualMusicalChairs の出力 s（0-based, 未確定は -1）
        internal_rank_j: ParallelVirtualNumberPlayers の出力 j（1-based）
        M_hat: 推定プレイヤー数
        good_arms: FindMultipleGoodArms の出力 G（0-based arm index リスト）
        assigned_arm: 最終エポックの Oracle 割当（0-based, 未割当は -1）
        group: 所属グループ（1-based, grand leader=1, sub-leader=2..n, follower>n は mod で分配）
        is_grand_leader: rank==1 なら True
        is_subleader: 2 <= rank <= n なら True
    """

    external_rank_s: int
    internal_rank_j: int
    M_hat: int
    good_arms: List[int]
    assigned_arm: int
    group: int
    is_grand_leader: bool = field(init=False)
    is_subleader: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_grand_leader = self.internal_rank_j == 1
        n = len(self.good_arms)
        self.is_subleader = 2 <= self.internal_rank_j <= n
