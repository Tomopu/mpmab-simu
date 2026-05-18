"""
論文再現に必要な初期化不変条件のテスト。

ここでは fallback 後の run() ではなく、VMC/VNP 系の出力そのものを見る。
delta は論文保証が働きやすいよう、既存の高速テストより小さめにする。
"""

import math

from simulator.algorithms.homogeneous import (
    HomogeneousHuang2022,
    HomogeneousMultiChannelIzumi2026,
)
from simulator.core.runner import Runner
from simulator.envs.bernoulli_mpmab import BernoulliMPMABEnv


def test_huang_vnp_outputs_permutation_rank_without_normalization():
    """Huang 2022 VNP が M_hat=M と j=1..M の permutation を返すこと。"""
    # Given
    K = 6
    M = 3
    delta = 0.01
    means = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]

    env = BernoulliMPMABEnv(means=means, num_players=M, seed=0)
    runner = Runner(env=env, horizon=2_000_000)
    algo = HomogeneousHuang2022(K=K, M=M, delta=delta, seed=0)

    # When
    k_tilde, mu_tilde = algo.find_good_arm(runner)
    tau_rank = math.ceil(K * math.log(1.0 / delta) / max(mu_tilde, 1e-12))
    tau_comm = math.ceil(math.log(1.0 / delta) / max(mu_tilde, 1e-12))
    s_list = algo.virtual_musical_chairs(runner, k_tilde, tau_rank)
    M_hat_list, j_list = algo.virtual_number_players(runner, k_tilde, s_list, tau_comm)

    # Then
    assert all(s >= 0 for s in s_list)
    assert len(set(s_list)) == M
    assert M_hat_list == [M] * M
    assert sorted(j_list) == list(range(1, M + 1))


def test_izumi_parallel_vnp_outputs_permutation_rank_without_normalization():
    """Izumi 2026 ParallelVNP が M_hat=M と j=1..M の permutation を返すこと。"""
    # Given
    K = 6
    M = 3
    n = 2
    delta = 0.01
    means = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]

    env = BernoulliMPMABEnv(means=means, num_players=M, seed=0)
    runner = Runner(env=env, horizon=2_000_000)
    algo = HomogeneousMultiChannelIzumi2026(K=K, M=M, n=n, delta=delta, seed=0)

    # When
    good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)
    tau = math.ceil(
        math.log(1.0 / delta) / max(min(mu_tilde_map.values()), 1e-12)
    )
    s_list = algo.parallel_virtual_musical_chairs(runner, good_arms, tau)
    M_hat_list, j_list = algo.parallel_virtual_number_players(
        runner, good_arms, s_list, tau
    )

    # Then
    assert all(s >= 0 for s in s_list)
    assert len(set(s_list)) == M
    assert M_hat_list == [M] * M
    assert sorted(j_list) == list(range(1, M + 1))
