from __future__ import annotations

from typing import Dict, List

from simulator.algorithms.homogeneous.states import PlayerState
from simulator.core.runner import Runner

def build_result(
    M: int,
    s_list: List[int],
    j_list: List[int],
    M_hat_list: List[int],
    k_tilde: int,
    mu_tilde: float,
    f_list: List[int],
    runner: Runner,
) -> Dict[str, object]:
    """PlayerState のリストと phase_durations を返す。"""
    player_states = [
        PlayerState(
            external_rank_s=s_list[m],
            internal_rank_j=j_list[m],
            M_hat=M_hat_list[m],
            good_arm=k_tilde,
            mu_tilde=mu_tilde,
            assigned_arm=f_list[m],
        )
        for m in range(M)
    ]
    return {
        "player_states": player_states,
        "phase_durations": runner.trace.phase_durations,
    }
