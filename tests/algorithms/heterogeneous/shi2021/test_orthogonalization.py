"""
Wang 2020 Orthogonalization と Rank Assignment のテスト。

確認ポイント:
- 各プレイヤーが一意な state を得る
- M_hat が真の M と一致する
- rank が重複しない
"""

import pytest

from simulator.envs.heterogeneous_mpmab import HeterogeneousMPMABEnv
from simulator.algorithms.heterogeneous.shi2021.algorithm import HeterogeneousShiBeacon2021
from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner


def make_env_and_runner(K: int, M: int, seed: int = 0, horizon: int = 50000):
    """テスト用の HeterogeneousMPMABEnv と HeterogeneousRunner を作成する。"""
    # 単純な means 行列: means[m][k] = 0.5 (すべて等しい)
    # orthogonalization のテストには報酬値は関係しないが環境が必要
    means = [[0.5] * K for _ in range(M)]
    env = HeterogeneousMPMABEnv(means, collision_sensing=True, seed=seed)
    runner = HeterogeneousRunner(env, horizon=horizon)
    return env, runner


class TestOrthogonalization:
    """orthogonalization フェーズのテスト。"""

    @pytest.mark.parametrize("K,M", [(4, 2), (5, 3), (6, 4)])
    def test_states_are_unique(self, K: int, M: int) -> None:
        """各プレイヤーの state が一意であること（orthogonalization の保証）。"""
        _, runner = make_env_and_runner(K, M, seed=42)
        algo = HeterogeneousShiBeacon2021(K=K, M=M, seed=42)
        state_list = algo.orthogonalization(runner)

        # 未確定 state なし
        assert all(s != -1 for s in state_list), f"未確定 state あり: {state_list}"
        # state の重複なし
        assert len(set(state_list)) == M, f"state が重複している: {state_list}"
        # state は {0,...,K-2} の範囲内
        for s in state_list:
            assert 0 <= s <= K - 2, f"state {s} が範囲外"

    def test_states_unique_small(self) -> None:
        """K=3, M=2 の最小ケース。"""
        K, M = 3, 2
        _, runner = make_env_and_runner(K, M, seed=0)
        algo = HeterogeneousShiBeacon2021(K=K, M=M, seed=0)
        state_list = algo.orthogonalization(runner)
        assert len(set(state_list)) == M
        assert all(0 <= s <= K - 2 for s in state_list)


class TestRankAssignment:
    """rank_assignment フェーズのテスト。"""

    @pytest.mark.parametrize("K,M", [(4, 2), (5, 3), (6, 4)])
    def test_rank_is_unique_and_mhat_equals_m(self, K: int, M: int) -> None:
        """各プレイヤーの rank が一意、かつ M_hat が真の M と一致すること。"""
        _, runner = make_env_and_runner(K, M, seed=42, horizon=200000)
        algo = HeterogeneousShiBeacon2021(K=K, M=M, seed=42)

        state_list = algo.orthogonalization(runner)
        M_hat_list, rank_list = algo.rank_assignment(runner, state_list)

        # rank が重複しない（1-based）
        assert len(set(rank_list)) == M, f"rank が重複: {rank_list}"
        # rank は 1..M の範囲
        assert set(rank_list) == set(range(1, M + 1)), f"rank の値が不正: {rank_list}"
        # M_hat が全プレイヤーで M に一致する
        for m, mh in enumerate(M_hat_list):
            assert mh == M, f"player {m}: M_hat={mh} != M={M}"
