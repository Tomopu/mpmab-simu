"""Result builder for BEACON (Shi 2021)."""

from __future__ import annotations

from typing import Dict, List

from simulator.algorithms.heterogeneous.shi2021.states import BeaconPlayerState
from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner


def build_beacon_result(
    M: int,
    state_list: List[int],
    rank_list: List[int],
    M_hat_list: List[int],
    assigned_arms: List[int],
    runner: HeterogeneousRunner,
) -> Dict[str, object]:
    """
    BEACON の実行結果を辞書形式で返す。

    Args:
        M: プレイヤー数
        state_list: 各プレイヤーの state（0-based）
        rank_list: 各プレイヤーの rank（1-based）
        M_hat_list: 各プレイヤーの推定プレイヤー数
        assigned_arms: 最終割当（0-based）
        runner: HeterogeneousRunner

    Returns:
        {
            "player_states": List[BeaconPlayerState],
            "phase_durations": Dict[str, int],
        }
    """
    player_states = [
        BeaconPlayerState(
            state_s=state_list[m],
            rank=rank_list[m],
            M_hat=M_hat_list[m],
            assigned_arm=assigned_arms[m],
        )
        for m in range(M)
    ]
    return {
        "player_states": player_states,
        "phase_durations": runner.trace.phase_durations,
    }
