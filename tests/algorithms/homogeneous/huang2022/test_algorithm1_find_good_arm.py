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

class TestFindGoodArm:
    """FindGoodArm のテスト。"""

    def test_returns_positive_mean_arm(self):
        """FindGoodArm が正の平均報酬を持つ arm を返すこと。"""
        # Given
        env, runner = make_env_and_runner(seed=0)
        algo = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=0)

        # When
        k_tilde, mu_tilde = algo.find_good_arm(runner)

        # Then
        # k_tilde が有効な arm index であること
        assert 0 <= k_tilde < K, f"k_tilde={k_tilde} が範囲外"
        # mu_tilde が正であること
        assert mu_tilde > 0, f"mu_tilde={mu_tilde} が非正"
        # 真の平均報酬が正であること
        assert MEANS[k_tilde] > 0, f"k_tilde={k_tilde} の真の報酬が 0"

    def test_mu_tilde_is_lower_bound(self):
        """mu_tilde が実際の arm 平均報酬以下（下界）であること。"""
        # Given
        env, runner = make_env_and_runner(seed=1)
        algo = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=1)

        # When
        k_tilde, mu_tilde = algo.find_good_arm(runner)

        # Then
        # mu_tilde は 2^{-p} 形式なので真値以下になるはず
        assert mu_tilde <= MEANS[k_tilde] + 1e-9, (
            f"mu_tilde={mu_tilde} が真値 {MEANS[k_tilde]} より大きい"
        )
