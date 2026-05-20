"""
BernoulliMPMABEnv のテスト。

確認内容:
- 同じ arm を 2 人以上が選ぶと全員の報酬が 0 になること
- collision していないプレイヤーは Bernoulli 報酬を得ること
- seed 固定時に結果が再現すること
- optimal_total_reward が正しく計算されること
- StepResult の各フィールドが正しいこと
"""

import pytest
from simulator.envs.bernoulli_mpmab import BernoulliMPMABEnv


class TestCollision:
    """collision 処理のテスト。"""

    def test_collision_gives_zero_reward(self):
        """同じ arm を 2 人が選んだとき、両者の報酬が 0 になること。"""
        # Given
        # means[0] = 1.0（必ず報酬が出るはず）だが、collision なので 0 になる
        env = BernoulliMPMABEnv(means=[1.0, 0.5], num_players=2, seed=0)

        # When
        result = env.step([0, 0])  # 両プレイヤーが arm 0 を選ぶ

        # Then
        assert result.rewards == [0.0, 0.0], "collision 時は両者の報酬が 0"
        assert result.collisions == [True, True], "collision フラグが True"

    def test_collision_three_players(self):
        """3 人全員が同じ arm を選んだとき、全員の報酬が 0 になること。"""
        # Given
        env = BernoulliMPMABEnv(means=[1.0, 0.5, 0.5], num_players=3, seed=0)

        # When
        result = env.step([0, 0, 0])

        # Then
        assert all(r == 0.0 for r in result.rewards)
        assert all(c for c in result.collisions)

    def test_partial_collision(self):
        """一部の arm に collision があり、他は collision なしの場合。"""
        # Given
        # arm 0: players 0,1 が collision → 両者 reward=0
        # arm 1: player 2 だけ → collision なし
        env = BernoulliMPMABEnv(means=[1.0, 1.0, 0.5], num_players=3, seed=0)

        # When
        result = env.step([0, 0, 1])

        # Then
        assert result.collisions[0] and result.collisions[1], "arm 0 の 2 人は collision"
        assert not result.collisions[2], "arm 1 の 1 人は collision なし"
        assert result.rewards[0] == 0.0 and result.rewards[1] == 0.0
        # arm 1 は means=1.0 なので必ず 1.0
        assert result.rewards[2] == 1.0

    def test_no_collision(self):
        """各プレイヤーが異なる arm を選んだとき、collision なし。"""
        # Given
        env = BernoulliMPMABEnv(means=[1.0, 1.0], num_players=2, seed=0)

        # When
        result = env.step([0, 1])

        # Then
        assert not any(result.collisions)
        # means が 1.0 なので両者とも報酬 1
        assert result.rewards[0] == 1.0
        assert result.rewards[1] == 1.0


class TestReproducibility:
    """seed 固定による再現性のテスト。"""

    def test_same_seed_same_results(self):
        """同じ seed でリセットすると同じ報酬列を得ること。"""
        # Given
        env1 = BernoulliMPMABEnv(means=[0.5, 0.3], num_players=2, seed=42)
        env2 = BernoulliMPMABEnv(means=[0.5, 0.3], num_players=2, seed=42)

        # When
        results1 = [env1.step([0, 1]).rewards for _ in range(10)]
        results2 = [env2.step([0, 1]).rewards for _ in range(10)]

        # Then
        assert results1 == results2, "同じ seed では同じ結果が出る"

    def test_reset_reproducibility(self):
        """reset() 後も同じ seed なら同じ結果が出ること。"""
        # Given
        env = BernoulliMPMABEnv(means=[0.5, 0.3], num_players=2, seed=42)

        # When
        results1 = [env.step([0, 1]).rewards for _ in range(5)]
        env.reset(seed=42)
        results2 = [env.step([0, 1]).rewards for _ in range(5)]

        # Then
        assert results1 == results2

    def test_different_seeds_different_results(self):
        """異なる seed では結果が異なること（高確率で）。"""
        # Given
        env1 = BernoulliMPMABEnv(means=[0.5, 0.5], num_players=2, seed=1)
        env2 = BernoulliMPMABEnv(means=[0.5, 0.5], num_players=2, seed=999)

        # When
        results1 = [env1.step([0, 1]).rewards for _ in range(20)]
        results2 = [env2.step([0, 1]).rewards for _ in range(20)]

        # Then
        assert results1 != results2, "異なる seed では結果が異なる（高確率）"


class TestOptimalReward:
    """optimal_total_reward の計算テスト。"""

    def test_optimal_is_top_m_sum(self):
        """optimal_total_reward が top-M arm の平均報酬和であること。"""
        # Given
        means = [0.9, 0.8, 0.5, 0.2, 0.1]
        M = 2

        # When
        env = BernoulliMPMABEnv(means=means, num_players=M, seed=0)

        # Then
        # top-2: 0.9 + 0.8 = 1.7
        assert abs(env.optimal_total_reward - 1.7) < 1e-9

    def test_instant_regret_nonneg_on_average(self):
        """期待値ベースでは instant_regret >= 0 のはず（enough samples）。"""
        # Given
        means = [0.9, 0.8, 0.1, 0.05]
        M = 2
        env = BernoulliMPMABEnv(means=means, num_players=M, seed=0)

        # When
        # top-2 以外の arm を選ぶと regret > 0 になりやすい
        total_regret = sum(env.step([2, 3]).instant_regret for _ in range(100))

        # Then
        # 平均 regret は正になるはず
        assert total_regret > 0


class TestValidation:
    """入力バリデーションのテスト。"""

    def test_invalid_arm_index(self):
        """範囲外の arm index で ValueError が発生すること。"""
        # Given
        env = BernoulliMPMABEnv(means=[0.5, 0.5], num_players=2, seed=0)

        # When / Then
        with pytest.raises(ValueError):
            env.step([0, 5])  # arm 5 は存在しない（K=2）

    def test_invalid_actions_length(self):
        """actions の長さが M と異なる場合に ValueError が発生すること。"""
        # Given
        env = BernoulliMPMABEnv(means=[0.5, 0.5], num_players=2, seed=0)

        # When / Then
        with pytest.raises(ValueError):
            env.step([0])  # 長さが 1（M=2 ではない）
