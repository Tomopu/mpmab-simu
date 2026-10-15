"""
比較用の基準 selfish_klucb（乱数の項のない Selfish KL-UCB）と selfish_ts（Selfish Thompson Sampling）のテスト。

どちらも simulate_rskl_batch の variant として R-SKL と同じ高速版で動く。
"""

import numpy as np
import pytest

from simulator.algorithms.homogeneous.trinh2021 import simulate_rskl_batch
from simulator.algorithms.homogeneous.trinh2021.policy import select_actions


@pytest.mark.parametrize("variant", ["selfish_klucb", "selfish_ts"])
def test_baselines_do_not_depend_on_batch_or_block(variant):
    # Given: 同じ trial（seed 22）を、別の trial と一緒に回す場合と単独で回す場合
    means = [0.8, 0.5, 0.3, 0.1]
    M, T = 2, 500

    # When
    together = simulate_rskl_batch([means, means], M, T, seeds=[21, 22], sample_times=[0, T], block=1024, variant=variant)
    alone = simulate_rskl_batch([means], M, T, seeds=[22], sample_times=[0, T], block=7, variant=variant)

    # Then: seed 22 の trial の結果は完全に一致する（trial ごとの乱数系列を使っている）
    assert together.cumulative_regret[1] == alone.cumulative_regret[0]
    assert together.collision_steps[1] == alone.collision_steps[0]


@pytest.mark.parametrize("variant", ["selfish_klucb", "selfish_ts"])
def test_baselines_metrics_match_recorded_actions(variant):
    # Given: 小さな環境
    means = [[0.9, 0.2, 0.6, 0.1, 0.4], [0.4, 0.1, 0.9, 0.6, 0.2]]
    M, T = 3, 300

    # When: 行動を記録しながら実行する
    result = simulate_rskl_batch(means, M, T, seeds=[5, 6], sample_times=[0, T], record_actions=True, variant=variant)

    # Then: 行動列から数え直した衝突ステップ数が一致し、行動はすべて有効な arm index
    actions = result.actions
    assert actions.min() >= 0 and actions.max() < 5
    for b in range(len(means)):
        steps = sum(int(any(np.sum(actions[t, b] == a) >= 2 for a in actions[t, b])) for t in range(T))
        assert result.collision_steps[b] == steps


def test_selfish_klucb_breaks_only_exact_ties():
    # Given: 指数がはっきり違う arm（0 回と多数回）と、同じ統計の 2 arm
    counts = np.array([[100.0, 0.0, 0.0]])
    sums = np.array([[10.0, 0.0, 0.0]])
    rng = np.random.default_rng(0)

    # When: 乱数を変えて何度も選ぶ
    picks = [int(select_actions(counts, sums, 50, rng.standard_normal((1, 3)), variant="selfish_klucb")[0][0])
             for _ in range(400)]

    # Then: 未試行の 2 arm（指数 1）だけが選ばれ、その中では両方が選ばれる
    assert set(picks) == {1, 2}


def test_selfish_ts_two_symmetric_players_eventually_separate():
    # Given: M = K = 2、μ = (0.9, 0.1)（原論文の図 1 の設定）
    means = [[0.9, 0.1]] * 20
    seeds = list(range(100, 120))

    # When
    res = simulate_rskl_batch(means, 2, 3000, seeds=seeds, sample_times=[0, 3000], variant="selfish_ts")

    # Then: 標本の乱数で対称性が崩れ、ほとんどの試行で最後は 2 人が別々の腕にいる
    assert res.final_distinct_top_m.mean() >= 0.8
