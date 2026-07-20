"""ParallelBEACON (Izumi 2026 Heterogeneous) の実行結果を構築するヘルパー。"""

from __future__ import annotations

from typing import Dict, List

from simulator.algorithms.heterogeneous.izumi2026.states import ParallelBeaconPlayerState
from simulator.algorithms.heterogeneous.izumi2026.helpers import player_group
from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner


def build_parallel_beacon_result(
    M: int,
    n: int,
    good_arms: List[int],
    s_list: List[int],
    j_list: List[int],
    M_hat_list: List[int],
    assigned_arms: List[int],
    runner: HeterogeneousRunner,
) -> Dict[str, object]:
    """
    ParallelBEACON の実行結果を辞書形式で返す。

    Args:
        M: プレイヤー数
        n: good arm 数（グループ数）
        good_arms: FindMultipleGoodArms の出力 G（0-based）
        s_list: 各プレイヤーの external rank（0-based）
        j_list: 各プレイヤーの internal rank（1-based）
        M_hat_list: 各プレイヤーの推定プレイヤー数
        assigned_arms: 最終割当（0-based）
        runner: HeterogeneousRunner

    Returns:
        {
            "player_states": List[ParallelBeaconPlayerState],
            "phase_durations": Dict[str, int],
        }
    """
    player_states = [
        ParallelBeaconPlayerState(
            external_rank_s=s_list[m],
            internal_rank_j=j_list[m],
            M_hat=M_hat_list[m],
            good_arms=list(good_arms),
            assigned_arm=assigned_arms[m],
            group=player_group(j_list[m], n) if n > 0 else 1,
        )
        for m in range(M)
    ]
    return {
        "player_states": player_states,
        "phase_durations": runner.trace.phase_durations,
    }
