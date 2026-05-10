"""
main.py: MPMAB シミュレーターのエントリーポイント。

小規模な sanity check を実行して、アルゴリズムが正常に動作することを確認する。
"""

from __future__ import annotations

import math

from envs.bernoulli_mpmab import BernoulliMPMABEnv
from algorithms.homogeneous_huang2022 import HomogeneousHuang2022
from core.runner import Runner
from utils.metrics import compute_metrics


def main() -> None:
    # 実験設定
    K = 5
    M = 2
    T = 50_000
    MEANS = [0.9, 0.8, 0.5, 0.2, 0.1]
    DELTA = 1.0 / (T * math.log(T))
    SEED = 42

    print("=== Huang 2022 Sanity Check ===")
    print(f"K={K}, M={M}, T={T}, delta={DELTA:.2e}, seed={SEED}")
    print(f"means: {MEANS}")
    print(f"top-{M} arms: arm 0 (μ={MEANS[0]}), arm 1 (μ={MEANS[1]})")
    print()

    env = BernoulliMPMABEnv(means=MEANS, num_players=M, seed=SEED)
    runner = Runner(env=env, horizon=T)
    algo = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=SEED)

    result = algo.run(runner)
    player_states = result["player_states"]

    # 結果表示
    for i, ps in enumerate(player_states):
        print(f"player {i}: s={ps.external_rank_s}, j={ps.internal_rank_j}, "
              f"M_hat={ps.M_hat}, k_tilde={ps.good_arm}, assigned={ps.assigned_arm}")

    print()
    metrics = compute_metrics(
        trace=runner.trace,
        player_states=player_states,
        means=MEANS,
        M=M,
    )
    print(f"total_steps:            {metrics['total_steps']}")
    print(f"find_good_duration:     {metrics['find_good_duration']}")
    print(f"rank_duration:          {metrics['rank_duration']}")
    print(f"number_players_duration:{metrics['number_players_duration']}")
    print(f"exploration_duration:   {metrics['exploration_duration']}")
    print(f"collision_count:        {metrics['collision_count']}")
    print(f"cumulative_regret:      {metrics['cumulative_regret']:.2f}")
    print(f"rank_assignment_success:{metrics['rank_assignment_success']}")
    print(f"player_count_success:   {metrics['player_count_success']}")
    print(f"final_assignment_success:{metrics['final_assignment_success']}")
    print(f"assignment_duplicate:   {metrics['assignment_duplicate']}")


if __name__ == "__main__":
    main()
