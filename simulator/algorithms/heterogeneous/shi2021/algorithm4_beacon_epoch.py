"""BEACON のメインエポックループ。"""

from __future__ import annotations

import math
from typing import List

from simulator.algorithms.heterogeneous.shi2021.algorithm3_initial_sampling import (
    BeaconExploreState,
)
from simulator.algorithms.heterogeneous.shi2021.communication import (
    BeaconCommunicationMixin,
)
from simulator.algorithms.heterogeneous.shi2021.oracle import matching_oracle
from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner


class BeaconEpochMixin(BeaconCommunicationMixin):
    """BEACON のエポックループを実行する補助クラス。"""

    def beacon_run_epochs(
        self,
        runner: HeterogeneousRunner,
        state_list: List[int],
        rank_list: List[int],
        explore_state: BeaconExploreState,
    ) -> List[int]:
        """
        BEACON のメインエポックループを実行する。

        Args:
            runner: HeterogeneousRunner
            state_list: 各プレイヤーの state（通信 arm として使う, 0-based）
            rank_list: 各プレイヤーの rank（1-based）
            explore_state: BeaconExploreState（更新される）

        Returns:
            最終エポック終了時の割当 arm（0-based, 長さ M）
        """
        K = self.K
        M = self.M
        es = explore_state
        c = state_list
        leader_pid = es.leader_pid

        prev_p = [[-1] * M for _ in range(K)]
        mu_tilde = [[0.0] * M for _ in range(K)]
        prev_mu_tilde_q = [[0] * M for _ in range(K)]

        arm_bits_needed = max(1, math.ceil(math.log2(max(2, K))))

        r = 0
        while True:
            r += 1
            runner.set_phase(f"beacon_epoch_{r}_comm")

            curr_p = [
                [_floor_log2(es.T[k][m]) for m in range(M)] for k in range(K)
            ]

            for m in range(M):
                if rank_list[m] == 1:
                    continue

                for k in range(K):
                    if curr_p[k][m] <= prev_p[k][m]:
                        continue

                    Q = math.ceil(1.0 + curr_p[k][m] / 2.0)
                    n_use = 1 << curr_p[k][m]
                    samps = es.samples[k][m]
                    if samps:
                        mu_hat_km = (
                            sum(samps[: min(n_use, len(samps))])
                            / min(n_use, len(samps))
                        )
                    else:
                        mu_hat_km = 0.0

                    new_q = self._quantize_mean(mu_hat_km, Q)
                    delta_q = new_q - prev_mu_tilde_q[k][m]
                    prev_mu_tilde_q[k][m] = new_q

                    sign_bit = 0 if delta_q >= 0 else 1
                    abs_val = abs(delta_q)
                    bits_to_send = [sign_bit] + self._int_to_bits(abs_val, Q)

                    self._send_receive(
                        runner, bits_to_send,
                        sender=m, receiver=leader_pid,
                        state_list=c,
                    )

                    mu_tilde[k][m] = self._dequantize_mean(prev_mu_tilde_q[k][m], Q)

            for k in range(K):
                n_use = max(1, 1 << curr_p[k][leader_pid])
                samps = es.samples[k][leader_pid]
                if samps:
                    mu_tilde[k][leader_pid] = (
                        sum(samps[: min(n_use, len(samps))])
                        / min(n_use, len(samps))
                    )

            t_r = runner.elapsed
            ln_tr = math.log(max(2, t_r))
            mu_bar = [[0.0] * M for _ in range(K)]
            for k in range(K):
                for m in range(M):
                    n_eff = max(1, 1 << (curr_p[k][m] + 1))
                    mu_bar[k][m] = mu_tilde[k][m] + math.sqrt(3.0 * ln_tr / n_eff)

            assignment = matching_oracle(mu_bar, M)

            for m in range(M):
                if rank_list[m] == 1:
                    continue
                arm_bits = self._int_to_bits(assignment[m], arm_bits_needed)
                self._send_receive(
                    runner, arm_bits,
                    sender=leader_pid, receiver=m,
                    state_list=c,
                )

            es.last_assigned_arms = list(assignment)

            runner.set_phase(f"beacon_epoch_{r}_explore")
            p_r = min(curr_p[assignment[m]][m] for m in range(M))
            n_explore = max(1, 1 << p_r)

            for _ in range(n_explore):
                actions = [assignment[m] for m in range(M)]
                result = runner.step(actions)

                for m in range(M):
                    k_a = assignment[m]
                    es.T[k_a][m] += 1
                    es.R[k_a][m] += result.rewards[m]
                    es.samples[k_a][m].append(result.rewards[m])

            prev_p = [row[:] for row in curr_p]

        return es.last_assigned_arms  # type: ignore[return-value]


def _floor_log2(n: int) -> int:
    """floor(log2(n)) を返す。n<=0 なら -1。"""
    if n <= 0:
        return -1
    return int(math.floor(math.log2(n)))
