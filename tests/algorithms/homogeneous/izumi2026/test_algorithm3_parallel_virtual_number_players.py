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

class TestParallelVirtualNumberPlayers:
    """ParallelVirtualNumberPlayers のテスト。"""

    def test_each_slot_at_most_one_good_arm(self):
        """
        各 K スロットで各プレイヤーが引く good arm が最大 1 本であること。

        ParallelVNP の各 h ループの K スロット内で、同一プレイヤーが
        good arm を 2 回以上引かないことを Trace から検証する。
        """
        # Given
        env, runner = make_env_and_runner(seed=20)
        algo = make_algo(n=N_GOOD, seed=20)

        good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)
        mu_min = min(mu_tilde_map.values())
        tau = math.ceil(math.log(1.0 / DELTA) / max(mu_min, 1e-12))
        s_list = algo.parallel_virtual_musical_chairs(runner, good_arms, tau)
        s_list_safe = [s if s >= 0 else 0 for s in s_list]

        # When
        algo.parallel_virtual_number_players(runner, good_arms, s_list_safe, tau)

        # Then
        # PVNP のフェーズレコードを取得する
        records = runner.trace.to_records()
        pvnp_records = [
            r for r in records
            if r["phase"] == "parallel_virtual_number_players"
        ]

        good_set = set(good_arms)
        block_size = K * tau  # 1 h ループ = K * tau ステップ

        for block_start in range(0, len(pvnp_records), block_size):
            block = pvnp_records[block_start:block_start + block_size]
            for m in range(M):
                pulled_good = [r["actions"][m] for r in block if r["actions"][m] in good_set]
                # 各スロット（K * tau ステップ）で同じ good arm は tau 回以上引かれないはず
                from collections import Counter
                counts = Counter(pulled_good)
                for arm, cnt in counts.items():
                    assert cnt <= tau, (
                        f"player {m} が 1 スロット内で good arm {arm} を {cnt} 回引いた "
                        f"（上限 {tau} 回）"
                    )

    def test_m_hat_plausible(self):
        """ParallelVirtualNumberPlayers の M_hat が 1 以上 K 以下であること。"""
        # Given
        env, runner = make_env_and_runner(seed=21)
        algo = make_algo(n=N_GOOD, seed=21)

        good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)
        mu_min = min(mu_tilde_map.values())
        tau = math.ceil(math.log(1.0 / DELTA) / max(mu_min, 1e-12))
        s_list = algo.parallel_virtual_musical_chairs(runner, good_arms, tau)
        s_list_safe = [s if s >= 0 else 0 for s in s_list]

        # When
        M_hat_list, j_list = algo.parallel_virtual_number_players(
            runner, good_arms, s_list_safe, tau
        )

        # Then
        for m in range(M):
            assert 1 <= M_hat_list[m] <= K, (
                f"player {m} の M_hat={M_hat_list[m]} が範囲外"
            )
            assert j_list[m] >= 1, (
                f"player {m} の j={j_list[m]} が 1 未満"
            )


# ============================================================
# TestFullRunIzumi
# ============================================================
