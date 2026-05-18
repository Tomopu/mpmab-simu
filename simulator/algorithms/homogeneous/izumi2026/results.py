from __future__ import annotations

from typing import Dict, List

from simulator.algorithms.homogeneous.states import PlayerStateIzumi
from simulator.core.runner import Runner

def build_result(
    M: int,
    good_arms: List[int],
    mu_tilde_map: Dict[int, float],
    s_list: List[int],
    j_list: List[int],
    M_hat_list: List[int],
    f_list: List[int],
    runner: Runner,
) -> Dict[str, object]:
    """PlayerStateIzumi のリストと phase_durations を返す。"""
    mu_tilde_min = min(mu_tilde_map.values()) if mu_tilde_map else 0.0
    player_states = [
        PlayerStateIzumi(
            external_rank_s=s_list[m],
            internal_rank_j=j_list[m],
            M_hat=M_hat_list[m],
            good_arms=list(good_arms),
            mu_tilde_min=mu_tilde_min,
            assigned_arm=f_list[m],
        )
        for m in range(M)
    ]
    return {
        "player_states": player_states,
        "phase_durations": runner.trace.phase_durations,
    }
