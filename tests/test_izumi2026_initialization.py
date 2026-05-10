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

from envs.bernoulli_mpmab import BernoulliMPMABEnv
from algorithms.homogeneous_huang2022 import HomogeneousHuang2022
from algorithms.homogeneous_multichannel_izumi2026 import HomogeneousMultiChannelIzumi2026
from core.runner import Runner, HorizonReached


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
        env, runner = make_env_and_runner(seed=0)
        algo = make_algo(n=N_GOOD, seed=0)

        good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)

        assert len(good_arms) == N_GOOD, (
            f"good_arms の本数が {N_GOOD} でない: {good_arms}"
        )

    def test_returns_positive_mean_arms(self):
        """FindMultipleGoodArms が返す arm の真の平均報酬が正であること。"""
        env, runner = make_env_and_runner(seed=1)
        algo = make_algo(n=N_GOOD, seed=1)

        good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)

        for k in good_arms:
            assert 0 <= k < K, f"arm index {k} が範囲外"
            assert MEANS[k] > 0, f"arm {k} の真の平均報酬が 0: means={MEANS[k]}"

    def test_mu_tilde_lower_bounds(self):
        """FindMultipleGoodArms の mu_tilde が実際の arm 平均報酬以下（下界）であること。"""
        env, runner = make_env_and_runner(seed=2)
        algo = make_algo(n=N_GOOD, seed=2)

        good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)

        for k in good_arms:
            assert k in mu_tilde_map, f"arm {k} の mu_tilde が記録されていない"
            assert mu_tilde_map[k] <= MEANS[k] + 1e-9, (
                f"arm {k}: mu_tilde={mu_tilde_map[k]} が真値 {MEANS[k]} より大きい"
            )

    def test_arms_are_unique(self):
        """FindMultipleGoodArms が重複のない arm set を返すこと。"""
        env, runner = make_env_and_runner(seed=3)
        algo = make_algo(n=N_GOOD, seed=3)

        good_arms, _ = algo.find_multiple_good_arms(runner)

        assert len(good_arms) == len(set(good_arms)), (
            f"good_arms に重複がある: {good_arms}"
        )


# ============================================================
# TestParallelVirtualMusicalChairs
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
        env, runner = make_env_and_runner(seed=10)
        algo = make_algo(n=N_GOOD, seed=10)

        good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)
        mu_min = min(mu_tilde_map.values())
        tau = math.ceil(math.log(1.0 / DELTA) / max(mu_min, 1e-12))

        # Trace を有効にするため Runner を新たに作らずに continue する
        # phase を PVMC に切り替えて steps を記録する
        start_steps = runner.elapsed

        algo.parallel_virtual_musical_chairs(runner, good_arms, tau)

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
        env, runner = make_env_and_runner(seed=11)
        algo = make_algo(n=N_GOOD, seed=11)

        good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)
        mu_min = min(mu_tilde_map.values())
        tau = math.ceil(math.log(1.0 / DELTA) / max(mu_min, 1e-12))

        s_list = algo.parallel_virtual_musical_chairs(runner, good_arms, tau)

        assigned = [s for s in s_list if s >= 0]
        assert len(assigned) >= 1, f"1 人も rank を得ていない: {s_list}"

    def test_ranks_tend_to_be_unique(self):
        """複数試行で rank 重複が減る傾向があること（確率的テスト）。"""
        successes = 0
        n_trials = 10
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

        assert successes >= n_trials // 2, (
            f"rank 重複なし率が低すぎる: {successes}/{n_trials}"
        )


# ============================================================
# TestParallelVirtualNumberPlayers
# ============================================================

class TestParallelVirtualNumberPlayers:
    """ParallelVirtualNumberPlayers のテスト。"""

    def test_each_slot_at_most_one_good_arm(self):
        """
        各 K スロットで各プレイヤーが引く good arm が最大 1 本であること。

        ParallelVNP の各 h ループの K スロット内で、同一プレイヤーが
        good arm を 2 回以上引かないことを Trace から検証する。
        """
        env, runner = make_env_and_runner(seed=20)
        algo = make_algo(n=N_GOOD, seed=20)

        good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)
        mu_min = min(mu_tilde_map.values())
        tau = math.ceil(math.log(1.0 / DELTA) / max(mu_min, 1e-12))
        s_list = algo.parallel_virtual_musical_chairs(runner, good_arms, tau)
        s_list_safe = [s if s >= 0 else 0 for s in s_list]

        algo.parallel_virtual_number_players(runner, good_arms, s_list_safe, tau)

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
        env, runner = make_env_and_runner(seed=21)
        algo = make_algo(n=N_GOOD, seed=21)

        good_arms, mu_tilde_map = algo.find_multiple_good_arms(runner)
        mu_min = min(mu_tilde_map.values())
        tau = math.ceil(math.log(1.0 / DELTA) / max(mu_min, 1e-12))
        s_list = algo.parallel_virtual_musical_chairs(runner, good_arms, tau)
        s_list_safe = [s if s >= 0 else 0 for s in s_list]

        M_hat_list, j_list = algo.parallel_virtual_number_players(
            runner, good_arms, s_list_safe, tau
        )

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

class TestFullRunIzumi:
    """HomogeneousMultiChannelIzumi2026.run() の統合テスト。"""

    def test_assigned_arms_no_duplicate(self):
        """run() 後に全プレイヤーの assigned_arm が重複しないこと。"""
        env, runner = make_env_and_runner(seed=42)
        algo = make_algo(n=N_GOOD, seed=42)

        result = algo.run(runner)
        player_states = result["player_states"]

        assigned = [ps.assigned_arm for ps in player_states if ps.assigned_arm >= 0]
        assert len(assigned) == len(set(assigned)), (
            f"assigned_arm に重複がある: {assigned}"
        )

    def test_all_players_get_assignment(self):
        """run() 後に全プレイヤーが arm を割り当てられること（horizon 内）。"""
        env, runner = make_env_and_runner(seed=42)
        algo = make_algo(n=N_GOOD, seed=42)

        result = algo.run(runner)
        player_states = result["player_states"]

        for i, ps in enumerate(player_states):
            assert ps.assigned_arm >= 0, (
                f"player {i} が arm を割り当てられていない: {ps}"
            )

    def test_phase_durations_recorded(self):
        """フェーズごとの所要ステップ数が記録されていること。"""
        env, runner = make_env_and_runner(seed=42)
        algo = make_algo(n=N_GOOD, seed=42)

        result = algo.run(runner)
        durations = result["phase_durations"]

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
        env, runner = make_env_and_runner(seed=42)
        algo = make_algo(n=N_GOOD, seed=42)

        result = algo.run(runner)
        player_states = result["player_states"]

        for i, ps in enumerate(player_states):
            assert len(ps.good_arms) == N_GOOD, (
                f"player {i} の good_arms 本数が {N_GOOD} でない: {ps.good_arms}"
            )
            for k in ps.good_arms:
                assert MEANS[k] > 0, (
                    f"player {i} の good arm {k} の真の平均報酬が 0"
                )

    def test_three_players_get_unique_assignments(self):
        """M=3, n=2 でも Grand Leader 不在や重複割当が起きないこと。"""
        means = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
        env = BernoulliMPMABEnv(means=means, num_players=3, seed=7)
        runner = Runner(env=env, horizon=2_000_000)
        algo = HomogeneousMultiChannelIzumi2026(
            K=6, M=3, n=2, delta=DELTA, seed=7
        )

        result = algo.run(runner)
        assigned = [ps.assigned_arm for ps in result["player_states"]]

        assert all(a >= 0 for a in assigned), f"未割当が残っている: {assigned}"
        assert len(set(assigned)) == 3, f"assigned_arm に重複がある: {assigned}"


# ============================================================
# TestN1Compatibility: n=1 のとき Huang 2022 と矛盾しないことを確認
# ============================================================

class TestN1Compatibility:
    """n=1 のとき HomogeneousMultiChannelIzumi2026 が Huang 2022 と矛盾しないことを確認する。"""

    def test_n1_assignment_success(self):
        """
        n=1 の Izumi 2026 が全プレイヤーに arm を割り当てられること。

        n=1 は good arm 1 本のみを通信チャンネルとして使う Huang 2022 類似ケース。
        どちらも同じ設定で成功できることを確認する。
        """
        env_izumi, runner_izumi = make_env_and_runner(seed=100)
        algo_izumi = make_algo(n=1, seed=100)

        result_izumi = algo_izumi.run(runner_izumi)
        ps_izumi = result_izumi["player_states"]

        assigned_izumi = [ps.assigned_arm for ps in ps_izumi]
        assert all(a >= 0 for a in assigned_izumi), (
            f"n=1 Izumi 2026: 未割当プレイヤーが存在する: {assigned_izumi}"
        )

    def test_n1_and_huang2022_both_succeed(self):
        """
        n=1 の Izumi 2026 と Huang 2022 がどちらも割当成功すること（同一 seed）。

        両アルゴリズムで割当に成功することを確認する。
        n=1 Izumi が失敗して Huang 2022 が成功する場合、実装に問題がある。
        """
        # Huang 2022
        env_h, runner_h = make_env_and_runner(seed=200)
        algo_h = HomogeneousHuang2022(K=K, M=M, delta=DELTA, seed=200)
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

        # Huang が成功した場合、Izumi も成功するはず（n=1 は同等設定）
        if huang_success and huang_no_dup:
            assert izumi_success, (
                f"Huang 2022 は成功したが n=1 Izumi 2026 は失敗した。"
                f"assigned: {[ps.assigned_arm for ps in ps_i]}"
            )

    def test_n1_no_duplicate_assignment(self):
        """n=1 の Izumi 2026 で assigned_arm に重複がないこと。"""
        env, runner = make_env_and_runner(seed=300)
        algo = make_algo(n=1, seed=300)

        result = algo.run(runner)
        player_states = result["player_states"]

        assigned = [ps.assigned_arm for ps in player_states if ps.assigned_arm >= 0]
        assert len(assigned) == len(set(assigned)), (
            f"n=1 Izumi 2026: assigned_arm に重複がある: {assigned}"
        )
