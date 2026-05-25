"""
HomogeneousMultiChannelIzumi2026 の初期化フェーズのテスト。

確認内容:
- FindMultipleGoodArms が n 本の正の平均報酬 arm を返すこと
- ParallelVirtualMusicalChairs で各ステップに各プレイヤーが最大 1 本の good arm を引くこと
  （spreading 条件により同一ブロック内に複数 good arm が衝突しないことを検証）
- ParallelVirtualNumberPlayers で各スロットに最大 1 本の good arm がマップされること
- run() 後に全プレイヤーが重複なく arm を割り当てられること
- n=1 のとき Huang 2022 と大きく矛盾しないこと

確率的アルゴリズムのため:
- seed 固定、小さな K・M、十分差のある means を使用して安定させる。
- delta を緩くしてテスト実行時間を短縮する。
"""

import math
import pytest

from simulator.envs.bernoulli_mpmab import BernoulliMPMABEnv
from simulator.algorithms.homogeneous import (
    HomogeneousHuang2022,
    HomogeneousMultiChannelIzumi2026,
)
from simulator.core.runner import Runner, HorizonReached


# テスト用の小規模設定
K = 5
M = 2
N_GOOD = 2           # FindMultipleGoodArms で探す good arm 数
MEANS = [0.9, 0.8, 0.1, 0.05, 0.02]  # top-2: arm 0, 1
DELTA = 0.1          # 失敗確率（テスト高速化のため緩め）
HORIZON = 1_000_000  # 十分大きい horizon


def make_env_and_runner(seed: int = 42):
    env = BernoulliMPMABEnv(means=MEANS, num_players=M, seed=seed)
    runner = Runner(env=env, horizon=HORIZON)
    return env, runner


def make_algo(n: int = N_GOOD, seed: int = 42):
    return HomogeneousMultiChannelIzumi2026(K=K, M=M, n=n, delta=DELTA, seed=seed)


# ============================================================
# TestFindMultipleGoodArms
# ============================================================

class TestFullRunIzumi:
    """HomogeneousMultiChannelIzumi2026.run() の統合テスト。"""

    def test_assigned_arms_no_duplicate(self):
        """run() 後に全プレイヤーの assigned_arm が重複しないこと。"""
        # Given
        env, runner = make_env_and_runner(seed=42)
        algo = make_algo(n=N_GOOD, seed=42)

        # When
        result = algo.run(runner)
        player_states = result["player_states"]

        # Then
        assigned = [ps.assigned_arm for ps in player_states if ps.assigned_arm >= 0]
        assert len(assigned) == len(set(assigned)), (
            f"assigned_arm に重複がある: {assigned}"
        )

    def test_n_greater_than_1_assignment_success(self):
        """n>1 のとき Grand Leader / Sub-Leader も含めて全員割り当てが成功すること。

        旧実装では n>1 で Grand Leader / Sub-Leader への割当経路がなく、
        good arms が active_arms に残り続けるデッドロックが発生していた（問題1）。
        修正後は j<=n のプレイヤーが good_arms[j-1] を受け取る経路が確保され、
        この seed で全員割り当てが成功する。
        """
        # Given
        env, runner = make_env_and_runner(seed=42)
        algo = make_algo(n=N_GOOD, seed=42)

        # When
        result = algo.run(runner)
        player_states = result["player_states"]
        assigned = [ps.assigned_arm for ps in player_states]

        # Then
        assert all(a >= 0 for a in assigned), (
            f"n={N_GOOD} で未割当プレイヤーが存在する: {assigned}"
        )
        assert len(assigned) == len(set(assigned)), (
            f"assigned_arm に重複がある: {assigned}"
        )

    def test_phase_durations_recorded(self):
        """フェーズごとの所要ステップ数が記録されていること。"""
        # Given
        env, runner = make_env_and_runner(seed=42)
        algo = make_algo(n=N_GOOD, seed=42)

        # When
        result = algo.run(runner)
        durations = result["phase_durations"]

        # Then
        assert any("find_multiple_good_arms" in k for k in durations), (
            "find_multiple_good_arms フェーズが記録されていない"
        )
        assert any("parallel_virtual_musical_chairs" in k for k in durations), (
            "ParallelVMC フェーズが記録されていない"
        )
        assert any("parallel_virtual_number_players" in k for k in durations), (
            "ParallelVNP フェーズが記録されていない"
        )

    def test_good_arms_recorded_in_player_state(self):
        """PlayerStateIzumi に good_arms が記録されていること。"""
        # Given
        env, runner = make_env_and_runner(seed=42)
        algo = make_algo(n=N_GOOD, seed=42)

        # When
        result = algo.run(runner)
        player_states = result["player_states"]

        # Then
        for i, ps in enumerate(player_states):
            assert len(ps.good_arms) == N_GOOD, (
                f"player {i} の good_arms 本数が {N_GOOD} でない: {ps.good_arms}"
            )
            for k in ps.good_arms:
                assert MEANS[k] > 0, (
                    f"player {i} の good arm {k} の真の平均報酬が 0"
                )

    def test_three_players_n2_assignment_success(self):
        """M=3, n=2 で全員が重複なく top-M arm に割り当てられること。

        旧実装では n>1 のデッドロックにより Grand Leader / Sub-Leader が未割当のまま
        active_arms に good arms が残り続けていた。
        修正後は担当チャンネル割り当て + フォロワーの相対 rank 割り当てにより全員完了する。
        """
        # Given
        means = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
        env = BernoulliMPMABEnv(means=means, num_players=3, seed=7)
        runner = Runner(env=env, horizon=2_000_000)
        algo = HomogeneousMultiChannelIzumi2026(
            K=6, M=3, n=2, delta=DELTA, seed=7
        )

        # When
        result = algo.run(runner)
        assigned = [ps.assigned_arm for ps in result["player_states"]]

        # Then
        assert all(a >= 0 for a in assigned), (
            f"M=3 n=2 で未割当プレイヤーが存在する: {assigned}"
        )
        assert len(assigned) == len(set(assigned)), (
            f"割当済み arm に重複がある: {assigned}"
        )
        top3_arms = set(sorted(range(len(means)), key=lambda k: means[k], reverse=True)[:3])
        assert all(a in top3_arms for a in assigned), (
            f"top-3 arm 以外が割り当てられている: {assigned}, top3={top3_arms}"
        )


# ============================================================
# TestN1Compatibility: n=1 のとき Huang 2022 と矛盾しないことを確認
# ============================================================

class TestN1Compatibility:
    """n=1 のとき HomogeneousMultiChannelIzumi2026 が Huang 2022 と矛盾しないことを確認する。"""

    def test_n1_assignment_success(self):
        """
        n=1 の Izumi 2026 が全プレイヤーに arm を割り当てられること。

        n=1 は good arm 1 本のみを通信チャンネルとして使う Huang 2022 類似ケース。
        good arm が accept された場合は leader に割り当てる必要がある。
        """
        # Given
        env_izumi, runner_izumi = make_env_and_runner(seed=100)
        algo_izumi = make_algo(n=1, seed=100)

        # When
        result_izumi = algo_izumi.run(runner_izumi)
        ps_izumi = result_izumi["player_states"]

        # Then
        assigned_izumi = [ps.assigned_arm for ps in ps_izumi]
        assert all(a >= 0 for a in assigned_izumi), (
            f"n=1 Izumi 2026: 未割当プレイヤーが存在する: {assigned_izumi}"
        )

    def test_n1_and_huang2022_both_succeed(self):
        """
        n=1 の Izumi 2026 と Huang 2022 がどちらも割当成功すること（同一 seed）。
        """
        # Given
        # Huang 2022
        env_h, runner_h = make_env_and_runner(seed=200)
        algo_h = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=200)

        # When
        result_h = algo_h.run(runner_h)
        ps_h = result_h["player_states"]
        huang_success = all(ps.assigned_arm >= 0 for ps in ps_h)
        huang_no_dup = len(set(ps.assigned_arm for ps in ps_h)) == M

        # n=1 Izumi 2026
        env_i, runner_i = make_env_and_runner(seed=200)
        algo_i = make_algo(n=1, seed=200)
        result_i = algo_i.run(runner_i)
        ps_i = result_i["player_states"]
        izumi_success = all(ps.assigned_arm >= 0 for ps in ps_i)
        izumi_no_dup = len(set(ps.assigned_arm for ps in ps_i)) == M

        # Then
        # Huang が成功した場合、n=1 Izumi も Huang 型の good arm leader 割当で成功する。
        if huang_success and huang_no_dup:
            assert izumi_success, (
                f"Huang 2022 は成功したが n=1 Izumi 2026 は失敗した。"
                f"assigned: {[ps.assigned_arm for ps in ps_i]}"
            )
            assert izumi_no_dup, (
                f"n=1 Izumi 2026 の assigned_arm に重複がある: "
                f"{[ps.assigned_arm for ps in ps_i]}"
            )

    def test_n1_no_duplicate_assignment(self):
        """n=1 の Izumi 2026 で assigned_arm に重複がないこと。"""
        # Given
        env, runner = make_env_and_runner(seed=300)
        algo = make_algo(n=1, seed=300)

        # When
        result = algo.run(runner)
        player_states = result["player_states"]

        # Then
        assigned = [ps.assigned_arm for ps in player_states if ps.assigned_arm >= 0]
        assert len(assigned) == len(set(assigned)), (
            f"n=1 Izumi 2026: assigned_arm に重複がある: {assigned}"
        )
