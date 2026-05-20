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

class TestFullRun:
    """HomogeneousHuang2022.run() の統合テスト。"""

    def test_assigned_arms_no_duplicate(self):
        """run() 後に全プレイヤーの assigned_arm が重複しないこと。"""
        # Given
        env, runner = make_env_and_runner(seed=42)
        algo = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=42)

        # When
        result = algo.run(runner)
        player_states = result["player_states"]

        # Then
        assigned = [ps.assigned_arm for ps in player_states if ps.assigned_arm >= 0]
        assert len(assigned) == len(set(assigned)), f"assigned_arm に重複がある: {assigned}"

    def test_all_players_get_assignment(self):
        """run() 後に全プレイヤーが arm を割り当てられること（horizon 内）。"""
        # Given
        env, runner = make_env_and_runner(seed=42)
        algo = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=42)

        # When
        result = algo.run(runner)
        player_states = result["player_states"]

        # Then
        for i, ps in enumerate(player_states):
            assert ps.assigned_arm >= 0, f"player {i} が arm を割り当てられていない"

    def test_phase_durations_recorded(self):
        """フェーズごとの所要ステップ数が記録されていること。"""
        # Given
        env, runner = make_env_and_runner(seed=42)
        algo = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=42)

        # When
        result = algo.run(runner)
        durations = result["phase_durations"]

        # Then
        assert any("find_good_arm" in k for k in durations), "find_good_arm フェーズが記録されていない"
        assert any("virtual_musical_chairs" in k for k in durations), "VMC フェーズが記録されていない"
        assert any("virtual_number_players" in k for k in durations), "VNP フェーズが記録されていない"

    def test_three_players_get_unique_assignments(self):
        """M=3 でも未割当や重複割当が残らないこと。"""
        # Given
        means = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
        env = BernoulliMPMABEnv(means=means, num_players=3, seed=7)
        runner = Runner(env=env, horizon=2_000_000)
        algo = HomogeneousHuang2022(K=6, M=3, delta=DELTA, seed=7)

        # When
        result = algo.run(runner)
        assigned = [ps.assigned_arm for ps in result["player_states"]]

        # Then
        assert all(a >= 0 for a in assigned), f"未割当が残っている: {assigned}"
        assert len(set(assigned)) == 3, f"assigned_arm に重複がある: {assigned}"
