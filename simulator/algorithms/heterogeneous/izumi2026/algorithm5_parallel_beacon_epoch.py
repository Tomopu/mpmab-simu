"""ParallelBEACON のメインエポックループ。"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.algorithms.heterogeneous.izumi2026.algorithm4_parallel_beacon_initial_sampling import (
    ParallelBeaconExploreState,
)
from simulator.algorithms.heterogeneous.izumi2026.helpers import (
    consume_dummy_steps,
    floor_log2,
    player_group,
)
from simulator.algorithms.heterogeneous.shi2021.oracle import matching_oracle
from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner


class ParallelBeaconEpochMixin:
    """ParallelBEACON のエポックループを実行する補助クラス。"""

    def parallel_beacon_run_epochs(
        self,
        runner: HeterogeneousRunner,
        state: ParallelBeaconExploreState,
    ) -> List[int]:
        """
        ParallelBEACON のメインエポックループを実行する。

        Args:
            runner: HeterogeneousRunner
            state: ParallelBeaconExploreState

        Returns:
            最終割当 arm（0-based, 長さ M）
        """
        K = state.K
        M = state.M
        n = state.n
        good_arms = state.good_arms
        rank_list = state.rank_list
        grand_leader = state.grand_leader_pid

        good_set = set(good_arms)
        non_good = [a for a in range(K) if a not in good_set]
        dummy = non_good[0] if non_good else good_arms[0]

        mu_tilde = [[0.0] * M for _ in range(K)]
        arm_bits_needed = max(1, math.ceil(math.log2(max(2, K))))

        r = 0
        while True:
            r += 1
            runner.set_phase(f"parallel_beacon_epoch_{r}_comm")

            curr_p = [
                [floor_log2(state.T[k][m]) for m in range(M)] for k in range(K)
            ]

            group_stats: Dict[int, Dict[int, Tuple[float, int]]] = {}

            max_followers_per_group = 0
            for g in range(1, n + 1):
                group_pids = state.group_map.get(g, [])
                group_stats[g] = {}
                for k in range(K):
                    wsum = 0.0
                    tn = 0
                    for m in group_pids:
                        n_use = max(1, 1 << curr_p[k][m]) if curr_p[k][m] >= 0 else 1
                        samps = state.samples[k][m]
                        mu_hat_km = (
                            sum(samps[: min(n_use, len(samps))]) / min(n_use, len(samps))
                            if samps else 0.0
                        )
                        wsum += mu_hat_km * state.T[k][m]
                        tn += state.T[k][m]
                    group_stats[g][k] = (wsum, tn)

                Ka = K
                _p_vals_g = [
                    curr_p[kk][mm]
                    for mm in group_pids for kk in range(K)
                    if curr_p[kk][mm] >= 0
                ]
                Q = _ceil(max(0, max(_p_vals_g) if _p_vals_g else 0) / 2.0 + 3)

                n_followers_g = max(0, len(group_pids) - 1)
                max_followers_per_group = max(max_followers_per_group, n_followers_g)
                comm_steps_up = n_followers_g * Q * Ka
                consume_dummy_steps(runner, comm_steps_up, dummy, M)

            n_subleaders = max(0, n - 1)
            _p_vals_gl = [
                curr_p[kk][grand_leader]
                for kk in range(K)
                if curr_p[kk][grand_leader] >= 0
            ]
            Q_gl = _ceil(max(0, max(_p_vals_gl) if _p_vals_gl else 0) / 2.0 + 3)
            comm_steps_gl = n_subleaders * Q_gl * K
            consume_dummy_steps(runner, comm_steps_gl, dummy, M)

            for k in range(K):
                for m in range(M):
                    g = player_group(rank_list[m], n)
                    gs_wsum, gs_n = group_stats.get(g, {}).get(k, (0.0, 0))
                    if gs_n > 0 and g == player_group(rank_list[grand_leader], n):
                        n_use = max(1, 1 << curr_p[k][m]) if curr_p[k][m] >= 0 else 1
                        samps = state.samples[k][m]
                        mu_tilde[k][m] = (
                            sum(samps[: min(n_use, len(samps))]) / min(n_use, len(samps))
                            if samps else 0.0
                        )
                    elif gs_n > 0:
                        mu_tilde[k][m] = gs_wsum / gs_n
                    else:
                        mu_tilde[k][m] = 0.0

            t_r = runner.elapsed
            ln_tr = math.log(max(2, t_r))
            mu_bar = [[0.0] * M for _ in range(K)]
            for k in range(K):
                for m in range(M):
                    p_km = curr_p[k][m] if curr_p[k][m] >= 0 else 0
                    n_eff = max(1, 1 << (p_km + 1))
                    mu_bar[k][m] = mu_tilde[k][m] + math.sqrt(3.0 * ln_tr / n_eff)

            assignment = matching_oracle(mu_bar, M)

            comm_steps_down = (n_subleaders + max_followers_per_group) * arm_bits_needed
            consume_dummy_steps(runner, comm_steps_down, dummy, M)

            state.last_assigned_arms = list(assignment)

            runner.set_phase(f"parallel_beacon_epoch_{r}_explore")
            p_r = min(
                (curr_p[assignment[m]][m] if curr_p[assignment[m]][m] >= 0 else 0)
                for m in range(M)
            )
            n_explore = max(1, 1 << p_r)

            for _ in range(n_explore):
                actions = [assignment[m] for m in range(M)]
                result = runner.step(actions)
                for m in range(M):
                    k_a = assignment[m]
                    state.T[k_a][m] += 1
                    state.R[k_a][m] += result.rewards[m]
                    state.samples[k_a][m].append(result.rewards[m])

        return state.last_assigned_arms  # type: ignore[return-value]
