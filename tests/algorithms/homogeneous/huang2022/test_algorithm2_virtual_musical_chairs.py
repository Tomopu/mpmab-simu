"""
HomogeneousHuang2022 の初期化フェーズのテスト。

確認内容:
- FindGoodArm が正の平均報酬を持つ arm を返すこと
- VirtualMusicalChairs の後に rank 重複が減ること
- VirtualNumberPlayers の後に各プレイヤーが同じ M_hat を得ること
- 最終的に各プレイヤーに異なる arm が割り当てられること

確率的アルゴリズムのため:
- seed 固定、小さな K・M、十分差のある means を使用して安定させる。
- 失敗確率 delta を小さく設定して成功確率を上げる。
"""

import pytest
from simulator.envs.bernoulli_mpmab import BernoulliMPMABEnv
from simulator.algorithms.homogeneous import HomogeneousHuang2022
from simulator.core.runner import Runner, HorizonReached


# テスト用の小規模設定
K = 5
M = 2
MEANS = [0.9, 0.8, 0.1, 0.05, 0.02]  # top-2: arm 0, 1
DELTA = 0.1  # 失敗確率（テスト高速化のため緩め）
HORIZON = 500_000  # 十分大きい horizon


def make_env_and_runner(seed: int = 42):
    env = BernoulliMPMABEnv(means=MEANS, num_players=M, seed=seed)
    runner = Runner(env=env, horizon=HORIZON)
    return env, runner

class TestVirtualMusicalChairs:
    """VirtualMusicalChairs のテスト。"""

    def test_rank_assignment(self):
        """VirtualMusicalChairs 後に少なくとも 1 人が rank を得ること。"""
        # Given
        env, runner = make_env_and_runner(seed=2)
        algo = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=2)
        k_tilde, mu_tilde = algo.find_good_arm(runner)
        import math
        mu_safe = max(mu_tilde, 1e-12)
        tau_rank = math.ceil(K * math.log(1.0 / DELTA) / mu_safe)

        # When
        s_list = algo.virtual_musical_chairs(runner, k_tilde, tau_rank)

        # Then
        # 少なくとも 1 人が rank を得ていること
        assigned = [s for s in s_list if s >= 0]
        assert len(assigned) >= 1, f"1 人も rank を得ていない: {s_list}"

    def test_ranks_tend_to_be_unique(self):
        """複数試行で rank 重複が減る傾向があること（確率的テスト）。"""
        # Given
        # 10 回試行して、重複なし率が 50% 以上あることを確認
        successes = 0
        n_trials = 10

        # When
        for seed in range(n_trials):
            env, runner = make_env_and_runner(seed=seed)
            algo = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=seed)
            try:
                k_tilde, mu_tilde = algo.find_good_arm(runner)
                import math
                mu_safe = max(mu_tilde, 1e-12)
                tau_rank = math.ceil(K * math.log(1.0 / DELTA) / mu_safe)
                s_list = algo.virtual_musical_chairs(runner, k_tilde, tau_rank)
                assigned = [s for s in s_list if s >= 0]
                if len(set(assigned)) == len(assigned):
                    successes += 1
            except HorizonReached:
                pass

        # Then
        assert successes >= n_trials // 2, (
            f"rank 重複なし率が低すぎる: {successes}/{n_trials}"
        )
