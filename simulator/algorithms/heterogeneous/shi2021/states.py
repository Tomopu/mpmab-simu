"""BEACON (Shi 2021) で使うプレイヤー状態の dataclass。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class BeaconPlayerState:
    """
    各プレイヤーの初期化フェーズ終了時の状態サマリー（BEACON 版）。

    Attributes:
        state_s: Orthogonalization の出力 s（0-based, 未確定は -1）
        rank: Rank Assignment の出力 rank（1-based, leader = 1）
        M_hat: 推定プレイヤー数
        assigned_arm: 最終エポックの Oracle 割当（0-based, 未割当は -1）
        is_leader: rank==1 なら True
    """

    state_s: int
    rank: int
    M_hat: int
    assigned_arm: int
    is_leader: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_leader = self.rank == 1
