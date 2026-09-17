"""
arm の並びを入れ替える shuffle_means と、実験での trial_means のテスト。
"""

from simulator.envs import shuffle_means
from simulator.experiments.configs import CONFIGS
from simulator.experiments.run_homogeneous import trial_means
from dataclasses import replace


def test_shuffle_means_is_deterministic_permutation():
    # Given
    means = [0.95, 0.9, 0.86, 0.82, 0.78, 0.7, 0.64, 0.58]

    # When
    shuffled_a, order_a = shuffle_means(means, seed=20260510)
    shuffled_b, order_b = shuffle_means(means, seed=20260510)
    shuffled_c, _ = shuffle_means(means, seed=20260511)

    # Then: 同じ seed なら同じ並び、期待値の多重集合は保たれ、shuffled[k] = means[order[k]]
    assert order_a == order_b and shuffled_a == shuffled_b
    assert sorted(shuffled_a) == sorted(means)
    assert all(shuffled_a[k] == means[order_a[k]] for k in range(len(means)))
    assert shuffled_c != shuffled_a


def test_trial_means_respects_shuffle_flag():
    # Given
    base = CONFIGS["tradeoff"]

    # When
    plain_means, plain_order = trial_means(base, seed=base.seed_base)
    shuffled_means, shuffled_order = trial_means(replace(base, shuffle_arms=True), seed=base.seed_base)

    # Then: フラグなしなら元の並び、ありなら並べ替えた列
    assert plain_means == list(base.means) and plain_order == list(range(base.K))
    assert sorted(shuffled_means) == sorted(base.means) and shuffled_order != plain_order
