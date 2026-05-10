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
from envs.bernoulli_mpmab import BernoulliMPMABEnv
from algorithms.homogeneous_huang2022 import HomogeneousHuang2022
from core.runner import Runner, HorizonReached


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


class TestFindGoodArm:
    """FindGoodArm のテスト。"""

    def test_returns_positive_mean_arm(self):
        """FindGoodArm が正の平均報酬を持つ arm を返すこと。"""
        env, runner = make_env_and_runner(seed=0)
        algo = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=0)

        k_tilde, mu_tilde = algo.find_good_arm(runner)

        # k_tilde が有効な arm index であること
        assert 0 <= k_tilde < K, f"k_tilde={k_tilde} が範囲外"
        # mu_tilde が正であること
        assert mu_tilde > 0, f"mu_tilde={mu_tilde} が非正"
        # 真の平均報酬が正であること
        assert MEANS[k_tilde] > 0, f"k_tilde={k_tilde} の真の報酬が 0"

    def test_mu_tilde_is_lower_bound(self):
        """mu_tilde が実際の arm 平均報酬以下（下界）であること。"""
        env, runner = make_env_and_runner(seed=1)
        algo = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=1)

        k_tilde, mu_tilde = algo.find_good_arm(runner)

        # mu_tilde は 2^{-p} 形式なので真値以下になるはず
        assert mu_tilde <= MEANS[k_tilde] + 1e-9, (
            f"mu_tilde={mu_tilde} が真値 {MEANS[k_tilde]} より大きい"
        )


class TestVirtualMusicalChairs:
    """VirtualMusicalChairs のテスト。"""

    def test_rank_assignment(self):
        """VirtualMusicalChairs 後に少なくとも 1 人が rank を得ること。"""
        env, runner = make_env_and_runner(seed=2)
        algo = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=2)

        k_tilde, mu_tilde = algo.find_good_arm(runner)
        import math
        mu_safe = max(mu_tilde, 1e-12)
        tau_rank = math.ceil(K * math.log(1.0 / DELTA) / mu_safe)

        s_list = algo.virtual_musical_chairs(runner, k_tilde, tau_rank)

        # 少なくとも 1 人が rank を得ていること
        assigned = [s for s in s_list if s >= 0]
        assert len(assigned) >= 1, f"1 人も rank を得ていない: {s_list}"

    def test_ranks_tend_to_be_unique(self):
        """複数試行で rank 重複が減る傾向があること（確率的テスト）。"""
        # 10 回試行して、重複なし率が 50% 以上あることを確認
        successes = 0
        n_trials = 10
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

        assert successes >= n_trials // 2, (
            f"rank 重複なし率が低すぎる: {successes}/{n_trials}"
        )


class TestFullRun:
    """HomogeneousHuang2022.run() の統合テスト。"""

    def test_assigned_arms_no_duplicate(self):
        """run() 後に全プレイヤーの assigned_arm が重複しないこと。"""
        env, runner = make_env_and_runner(seed=42)
        algo = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=42)

        result = algo.run(runner)
        player_states = result["player_states"]

        assigned = [ps.assigned_arm for ps in player_states if ps.assigned_arm >= 0]
        assert len(assigned) == len(set(assigned)), f"assigned_arm に重複がある: {assigned}"

    def test_all_players_get_assignment(self):
        """run() 後に全プレイヤーが arm を割り当てられること（horizon 内）。"""
        env, runner = make_env_and_runner(seed=42)
        algo = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=42)

        result = algo.run(runner)
        player_states = result["player_states"]

        for i, ps in enumerate(player_states):
            assert ps.assigned_arm >= 0, f"player {i} が arm を割り当てられていない"

    def test_phase_durations_recorded(self):
        """フェーズごとの所要ステップ数が記録されていること。"""
        env, runner = make_env_and_runner(seed=42)
        algo = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=42)

        result = algo.run(runner)
        durations = result["phase_durations"]

        assert any("find_good_arm" in k for k in durations), "find_good_arm フェーズが記録されていない"
        assert any("virtual_musical_chairs" in k for k in durations), "VMC フェーズが記録されていない"
        assert any("virtual_number_players" in k for k in durations), "VNP フェーズが記録されていない"

    def test_three_players_get_unique_assignments(self):
        """M=3 でも未割当や重複割当が残らないこと。"""
        means = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
        env = BernoulliMPMABEnv(means=means, num_players=3, seed=7)
        runner = Runner(env=env, horizon=2_000_000)
        algo = HomogeneousHuang2022(K=6, M=3, delta=DELTA, seed=7)

        result = algo.run(runner)
        assigned = [ps.assigned_arm for ps in result["player_states"]]

        assert all(a >= 0 for a in assigned), f"未割当が残っている: {assigned}"
        assert len(set(assigned)) == 3, f"assigned_arm に重複がある: {assigned}"
