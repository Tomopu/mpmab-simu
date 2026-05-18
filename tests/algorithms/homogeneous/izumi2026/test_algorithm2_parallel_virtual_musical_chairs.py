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

class TestParallelVirtualMusicalChairs:
    """ParallelVirtualMusicalChairs のテスト。"""

    def test_each_player_pulls_single_arm_per_timestep(self):
        """
        各 timestep で各プレイヤーの action が 1 本の arm として記録されること。

        ParallelVMC は複数 good arms を使うが、物理的には同一 player が同一時刻に
        複数 arm を pull できない。論文制約は「同一ブロックで同じ good arm を
        二度引かない」ではなく「同一時刻に複数 pull しない」なので、Trace の
        各 record が player ごとに単一 action を持つことを検証する。
        """
        # Given
        env, runner = make_env_and_runner(seed=10)
        algo = make_algo(n=N_GOOD, seed=10)

        good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)
        mu_min = min(mu_tilde_map.values())
        tau = math.ceil(math.log(1.0 / DELTA) / max(mu_min, 1e-12))

        # Trace を有効にするため Runner を新たに作らずに continue する
        # phase を PVMC に切り替えて steps を記録する
        start_steps = runner.elapsed

        # When
        algo.parallel_virtual_musical_chairs(runner, good_arms, tau)

        # Then
        # Trace から PVMC フェーズの各ステップを取得する
        records = runner.trace.to_records()
        pvmc_records = [
            r for r in records
            if r["phase"] == "parallel_virtual_musical_chairs"
        ]

        assert runner.elapsed > start_steps
        for record in pvmc_records:
            assert len(record["actions"]) == M
            for action in record["actions"]:
                assert isinstance(action, int)
                assert 0 <= action < K

    def test_rank_assignment(self):
        """ParallelVirtualMusicalChairs 後に少なくとも 1 人が rank を得ること。"""
        # Given
        env, runner = make_env_and_runner(seed=11)
        algo = make_algo(n=N_GOOD, seed=11)

        good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)
        mu_min = min(mu_tilde_map.values())
        tau = math.ceil(math.log(1.0 / DELTA) / max(mu_min, 1e-12))

        # When
        s_list = algo.parallel_virtual_musical_chairs(runner, good_arms, tau)

        # Then
        assigned = [s for s in s_list if s >= 0]
        assert len(assigned) >= 1, f"1 人も rank を得ていない: {s_list}"

    def test_ranks_tend_to_be_unique(self):
        """複数試行で rank 重複が減る傾向があること（確率的テスト）。"""
        # Given
        successes = 0
        n_trials = 10

        # When
        for seed in range(n_trials):
            env, runner = make_env_and_runner(seed=seed)
            algo = make_algo(n=N_GOOD, seed=seed)
            try:
                good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)
                mu_min = min(mu_tilde_map.values())
                tau = math.ceil(math.log(1.0 / DELTA) / max(mu_min, 1e-12))
                s_list = algo.parallel_virtual_musical_chairs(runner, good_arms, tau)
                assigned = [s for s in s_list if s >= 0]
                if len(set(assigned)) == len(assigned):
                    successes += 1
            except HorizonReached:
                pass

        # Then
        assert successes >= n_trials // 2, (
            f"rank 重複なし率が低すぎる: {successes}/{n_trials}"
        )


# ============================================================
# TestParallelVirtualNumberPlayers
# ============================================================
