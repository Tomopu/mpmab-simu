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

class TestFindMultipleGoodArms:
    """FindMultipleGoodArms のテスト。"""

    def test_returns_n_arms(self):
        """FindMultipleGoodArms が n 本の arm を返すこと。"""
        # Given
        env, runner = make_env_and_runner(seed=0)
        algo = make_algo(n=N_GOOD, seed=0)

        # When
        good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)

        # Then
        assert len(good_arms) == N_GOOD, (
            f"good_arms の本数が {N_GOOD} でない: {good_arms}"
        )

    def test_returns_positive_mean_arms(self):
        """FindMultipleGoodArms が返す arm の真の平均報酬が正であること。"""
        # Given
        env, runner = make_env_and_runner(seed=1)
        algo = make_algo(n=N_GOOD, seed=1)

        # When
        good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)

        # Then
        for k in good_arms:
            assert 0 <= k < K, f"arm index {k} が範囲外"
            assert MEANS[k] > 0, f"arm {k} の真の平均報酬が 0: means={MEANS[k]}"

    def test_mu_tilde_lower_bounds(self):
        """FindMultipleGoodArms の mu_tilde が実際の arm 平均報酬以下（下界）であること。"""
        # Given
        env, runner = make_env_and_runner(seed=2)
        algo = make_algo(n=N_GOOD, seed=2)

        # When
        good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)

        # Then
        for k in good_arms:
            assert k in mu_tilde_map, f"arm {k} の mu_tilde が記録されていない"
            assert mu_tilde_map[k] <= MEANS[k] + 1e-9, (
                f"arm {k}: mu_tilde={mu_tilde_map[k]} が真値 {MEANS[k]} より大きい"
            )

    def test_arms_are_unique(self):
        """FindMultipleGoodArms が重複のない arm set を返すこと。"""
        # Given
        env, runner = make_env_and_runner(seed=3)
        algo = make_algo(n=N_GOOD, seed=3)

        # When
        good_arms, _ = algo.find_multiple_good_arms(runner)

        # Then
        assert len(good_arms) == len(set(good_arms)), (
            f"good_arms に重複がある: {good_arms}"
        )


# ============================================================
# TestParallelVirtualMusicalChairs
# ============================================================
