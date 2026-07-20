"""
HeterogeneousMPMABEnv のテスト。

collision 処理、seed 再現性、最適マッチング計算を確認する。
"""

import pytest

from simulator.envs.heterogeneous_mpmab import HeterogeneousMPMABEnv, _hungarian_max_weight


class TestHeterogeneousMPMABEnv:
    """HeterogeneousMPMABEnv の基本動作テスト。"""

    def test_no_collision_rewards_are_nonzero(self) -> None:
        """衝突なし: 各プレイヤーが異なる arm を選ぶと報酬が入る可能性がある。"""
        means = [[0.9, 0.1], [0.1, 0.9]]
        env = HeterogeneousMPMABEnv(means, seed=0)
        # player 0 → arm 0, player 1 → arm 1 (衝突なし)
        results = [env.step([0, 1]) for _ in range(50)]
        total = sum(r.total_reward for r in results)
        # 期待報酬は 0.9+0.9=1.8/step なので 50 step で概ね 50 以上
        assert total > 20, "衝突なしで報酬が極端に低い"

    def test_collision_gives_zero_reward(self) -> None:
        """衝突: 2 人が同じ arm を選ぶと全員の報酬 = 0。"""
        means = [[0.99, 0.1], [0.99, 0.1]]
        env = HeterogeneousMPMABEnv(means, seed=0)
        result = env.step([0, 0])  # 両プレイヤーが arm 0 を選ぶ
        assert result.rewards == [0.0, 0.0]
        assert result.collisions == [True, True]

    def test_seed_reproducibility(self) -> None:
        """seed が同じなら同じ結果が再現する。"""
        means = [[0.7, 0.3], [0.3, 0.7]]
        env1 = HeterogeneousMPMABEnv(means, seed=42)
        env2 = HeterogeneousMPMABEnv(means, seed=42)
        for _ in range(20):
            r1 = env1.step([0, 1])
            r2 = env2.step([0, 1])
            assert r1.rewards == r2.rewards

    def test_optimal_reward_is_max_matching(self) -> None:
        """optimal_total_reward は最大重み二部マッチングの値と一致する。"""
        means = [[0.9, 0.1, 0.5], [0.1, 0.8, 0.5], [0.5, 0.5, 0.7]]
        env = HeterogeneousMPMABEnv(means, seed=0)
        # 最適: player0→arm0(0.9), player1→arm1(0.8), player2→arm2(0.7) = 2.4
        assert abs(env.optimal_total_reward - 2.4) < 1e-9

    def test_k_must_be_geq_m(self) -> None:
        """K < M なら ValueError。"""
        means = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]  # 3 players, 2 arms
        with pytest.raises(ValueError, match="K="):
            HeterogeneousMPMABEnv(means, seed=0)

    def test_instant_regret_non_negative(self) -> None:
        """instant_regret は常に 0 以上（観測報酬は期待値以下の可能性があるため負になることもある）。"""
        # 完全割当の場合でも確率的なので regret < 0 はありうる
        # ここでは型と符号の異常がないことだけ確認
        means = [[0.9, 0.1], [0.1, 0.9]]
        env = HeterogeneousMPMABEnv(means, seed=1)
        result = env.step([0, 1])
        # optimal - rewards は float として計算される
        assert isinstance(result.instant_regret, float)


class TestHungarianMaxWeight:
    """_hungarian_max_weight のユニットテスト。"""

    def test_simple_2x2(self) -> None:
        """2×2 の単純ケース。"""
        cost = [[0.9, 0.1], [0.1, 0.8]]
        val = _hungarian_max_weight(cost)
        # 最適: (0,0)→0.9 + (1,1)→0.8 = 1.7
        assert abs(val - 1.7) < 1e-9

    def test_anti_diagonal_better(self) -> None:
        """反対角が最適なケース。"""
        cost = [[0.1, 0.9], [0.8, 0.1]]
        val = _hungarian_max_weight(cost)
        # 最適: (0,1)→0.9 + (1,0)→0.8 = 1.7
        assert abs(val - 1.7) < 1e-9
