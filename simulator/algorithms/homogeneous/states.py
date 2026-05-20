"""Player state dataclasses for homogeneous algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PlayerState:
    """
    各プレイヤーの初期化フェーズ終了時の状態サマリー。

    Attributes:
        external_rank_s: VirtualMusicalChairs の出力 s（0-based, 論文の s-1 に対応）
        internal_rank_j: VirtualNumberPlayers の出力 j（1-based, 論文と同じ）
        M_hat: 推定プレイヤー数
        good_arm: FindGoodArm の出力 k_tilde（0-based）
        mu_tilde: good_arm の報酬下界
        assigned_arm: DistributedExploration の出力（0-based, 未割当は -1）
    """

    external_rank_s: int
    internal_rank_j: int
    M_hat: int
    good_arm: int
    mu_tilde: float
    assigned_arm: int


@dataclass
class PlayerStateIzumi:
    """
    各プレイヤーの初期化フェーズ終了時の状態サマリー（Izumi 2026 版）。

    Attributes:
        external_rank_s: ParallelVirtualMusicalChairs の出力 s（0-based, 論文 s-1 に対応）
        internal_rank_j: ParallelVirtualNumberPlayers の出力 j（1-based, 論文と同じ）
        M_hat: 推定プレイヤー数
        good_arms: FindMultipleGoodArms の出力（0-based arm index のリスト）
        mu_tilde_min: good_arms 中の最小報酬下界
        assigned_arm: HierarchicalDistributedExploration の出力（0-based, 未割当は -1）
    """

    external_rank_s: int
    internal_rank_j: int
    M_hat: int
    good_arms: List[int]
    mu_tilde_min: float
    assigned_arm: int

