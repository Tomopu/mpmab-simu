from __future__ import annotations

from typing import Dict, List, Optional

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
    player_k_tilde: Optional[List[int]] = None,
    player_mu_tilde: Optional[List[float]] = None,
    init_failure_reason: Optional[str] = None,
) -> Dict[str, object]:
    """
    PlayerState のリストと phase_durations を返す。

    player_k_tilde を渡すと、各プレイヤーの good_arm に自分の値を入れる（合意の検証用）。
    渡さなければ従来どおり player 0 の値を全員に入れる。
    """
    player_states = []
    for m in range(M):
        if player_k_tilde is not None and m < len(player_k_tilde):
            k_m = player_k_tilde[m]
            mu_m = player_mu_tilde[m] if player_mu_tilde is not None and m < len(player_mu_tilde) else mu_tilde
        else:
            k_m, mu_m = k_tilde, mu_tilde
        player_states.append(
            PlayerState(
                external_rank_s=s_list[m],
                internal_rank_j=j_list[m],
                M_hat=M_hat_list[m],
                good_arm=k_m,
                mu_tilde=mu_m,
                assigned_arm=f_list[m],
            )
        )
    return {
        "player_states": player_states,
        "phase_durations": runner.trace.phase_durations,
        "good_arm": k_tilde,
        "init_failure_reason": init_failure_reason,
    }
