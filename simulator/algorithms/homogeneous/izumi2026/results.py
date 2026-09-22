from __future__ import annotations

from typing import Dict, List, Optional

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
    player_good_arms: Optional[List[List[int]]] = None,
    player_mu_tilde: Optional[List[Dict[int, float]]] = None,
    init_failure_reason: Optional[str] = None,
) -> Dict[str, object]:
    """
    PlayerStateIzumi のリストと phase_durations を返す。

    player_good_arms を渡すと、各プレイヤーの good_arms に自分の G を入れる（合意の検証用）。
    渡さなければ従来どおり player 0 の G を全員に入れる。
    init_failure_reason は、初期化の確率的な失敗を例外にせず記録するための文字列（成功なら None）。
    """
    player_states = []
    for m in range(M):
        if player_good_arms is not None and m < len(player_good_arms):
            g_m = list(player_good_arms[m])
            mt = player_mu_tilde[m] if player_mu_tilde is not None and m < len(player_mu_tilde) else {}
            mu_min_m = min(mt.values()) if mt else 0.0
        else:
            g_m = list(good_arms)
            mu_min_m = min(mu_tilde_map.values()) if mu_tilde_map else 0.0
        player_states.append(
            PlayerStateIzumi(
                external_rank_s=s_list[m],
                internal_rank_j=j_list[m],
                M_hat=M_hat_list[m],
                good_arms=g_m,
                mu_tilde_min=mu_min_m,
                assigned_arm=f_list[m],
            )
        )
    return {
        "player_states": player_states,
        "phase_durations": runner.trace.phase_durations,
        "good_arms": list(good_arms),
        "init_failure_reason": init_failure_reason,
    }
