"""
動的な設定（プレイヤーが入る・出る）の高速版 simulate_dynamic_batch のテスト。
"""

import numpy as np
import pytest

from simulator.algorithms.homogeneous.trinh2021.dynamic import sample_arrivals, simulate_dynamic_batch


@pytest.mark.parametrize("variant", ["paper", "selfish_ts"])
def test_single_player_has_no_collisions_and_consistent_totals(variant):
    # Given: 最初の 1 人だけが参加し、だれも入らない（入る時刻はすべて horizon の外）
    means = [0.9, 0.5, 0.1]
    T = 400

    # When
    r = simulate_dynamic_batch(means, T, seeds=[1, 2], entries_per_trial=[[0, T + 5, T + 9]] * 2,
                               sample_times=[T], variant=variant)

    # Then: 衝突は起きず、最適な期待報酬は毎ステップ 0.9、regret = 最適 - 実際
    assert (r.collision_steps == 0).all()
    np.testing.assert_allclose(r.optimal_reward, 0.9 * T)
    np.testing.assert_allclose(r.cumulative_regret, r.optimal_reward - r.system_reward)
    np.testing.assert_allclose(r.curve[:, -1], r.cumulative_regret)
    assert (r.n_entered == 1).all()


def test_entries_follow_the_authors_timing_and_block_when_full():
    # Given: K = 2 の席。1 人目は時刻 0、2 人目は時刻 9、3 人目は時刻 20 に来る（出ないモデル）
    means = [0.8, 0.3]
    T = 50

    # When
    r = simulate_dynamic_batch(means, T, seeds=[3], entries_per_trial=[[0, 9, 20]], variant="paper")

    # Then: 2 人目は手番 10（著者の時刻 9）のあとに着く。3 人目は席が埋まっていて入れない
    #       最適な期待報酬 = 手番 1〜10 は 0.8、手番 11〜50 は 0.8 + 0.3
    assert r.n_entered[0] == 2
    np.testing.assert_allclose(r.optimal_reward[0], 0.8 * 10 + 1.1 * 40)


def test_leaving_frees_a_slot_and_ratio_is_at_most_one():
    # Given: K = 2。1 人目は時刻 4 に出て、2 人目（時刻 2 に来る）と 3 人目（時刻 6 に来る）が入る
    means = [0.7, 0.2]
    T = 30

    # When
    r = simulate_dynamic_batch(means, T, seeds=[4], entries_per_trial=[[0, 2, 6]],
                               leaving_per_trial=[[4, 30, 30]], variant="selfish_ts")

    # Then: 3 人とも席に着ける。参加人数は 1（手番 1〜3）→ 2（4〜5）→ 1（6〜7）→ 2（8〜30）
    assert r.n_entered[0] == 3
    np.testing.assert_allclose(r.optimal_reward[0], 0.7 * 3 + 0.9 * 2 + 0.7 * 2 + 0.9 * 23)
    assert 0.0 <= r.ratio[0] <= 1.0


def test_results_do_not_depend_on_batch():
    # Given: 同じ trial を単独で回す場合と、別の trial と一緒に回す場合
    means = [0.9, 0.6, 0.3, 0.1]
    rng = np.random.default_rng(0)
    e1, l1 = sample_arrivals(rng, K=4, T=300, lam=1 / 30, mean_stay=60)
    e2, l2 = sample_arrivals(rng, K=4, T=300, lam=1 / 30, mean_stay=60)

    # When
    both = simulate_dynamic_batch(means, 300, seeds=[7, 8], entries_per_trial=[e1, e2], leaving_per_trial=[l1, l2])
    alone = simulate_dynamic_batch(means, 300, seeds=[8], entries_per_trial=[e2], leaving_per_trial=[l2])

    # Then
    assert both.cumulative_regret[1] == pytest.approx(alone.cumulative_regret[0])
    assert both.collision_steps[1] == alone.collision_steps[0]


def test_sample_arrivals_matches_the_authors_conventions():
    # Given / When
    e, l = sample_arrivals(np.random.default_rng(1), K=10, T=10_000, lam=1 / 1000, mean_stay=500)

    # Then: 少なくとも K 人、先頭は 0、昇順、出る時刻は入る時刻より後で horizon 以下
    assert len(e) >= 10 and e[0] == 0 and (np.diff(e) >= 0).all()
    assert (l > e).all() and (l <= 10_000 + 1).all()
