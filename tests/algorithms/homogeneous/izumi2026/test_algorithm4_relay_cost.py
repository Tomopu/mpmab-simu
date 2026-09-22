"""
HierarchicalDistributedExploration の通信時間の課金規則のテスト。

確認内容:
- サブリーダー段（Sub-Leader → Grand Leader）は、サブリーダーが割当済みになった後の
  フェーズでも (n-1) 本分の時間を課金すること（論文: 割当済みのサブリーダーも
  全員の割当が確定するまで通信役を続ける）。
- n=1 ではサブリーダー段の課金が 0 であること。

2026-09-22 より前の実装は未割当のサブリーダーだけを数えていたため、
n>=2 で通信時間を過小に課金していた。このテストはその回帰を防ぐ。
"""

import pytest

from simulator.algorithms.homogeneous import HomogeneousMultiChannelIzumi2026
from simulator.core.runner import Runner
from simulator.envs.bernoulli_mpmab import BernoulliMPMABEnv

# 通信路（arm 0, 1）はすぐに受理されて担当のサブリーダーに割り当てられる一方、
# フォロワーが受け取る arm 2, 3 は arm 4 との差が小さく、受理まで数フェーズかかる。
# そのため「サブリーダーは割当済み、フォロワーは未割当」のフェーズが必ず生じる。
K = 8
M = 4
MEANS = [0.9, 0.85, 0.5, 0.30, 0.25, 0.05, 0.04, 0.03]
DELTA = 0.1
HORIZON = 5_000_000


def run(n: int, seed: int = 0) -> HomogeneousMultiChannelIzumi2026:
    env = BernoulliMPMABEnv(means=MEANS, num_players=M, seed=seed)
    runner = Runner(env=env, horizon=HORIZON)
    algo = HomogeneousMultiChannelIzumi2026(K=K, M=M, n=n, delta=DELTA, seed=seed)
    result = algo.run(runner)
    assigned = [ps.assigned_arm for ps in result["player_states"]]
    assert -1 not in assigned, "全員が割り当てられる設定のはず"
    return algo


class TestSubleaderRelayCost:
    """サブリーダー段の課金規則のテスト。"""

    def test_assigned_subleaders_are_still_charged(self):
        """割当済みのサブリーダーがいるフェーズでも (n-1) 本分を課金すること。"""
        n = 2
        algo = run(n=n)
        log = algo.hde_comm_log
        assert len(log) >= 2

        # 設定が意図どおりか: サブリーダーが割当済みで通信が続くフェーズがある
        phases_with_assigned_sub = [
            e for e in log if e["n_subleaders_unassigned"] < n - 1
        ]
        assert phases_with_assigned_sub, "割当済みサブリーダーのいるフェーズが生じる設定にする"

        for e in log:
            assert e["n_subleaders_charged"] == n - 1
            assert e["uplink_sub"] == (n - 1) * e["active_arms"] * e["Q"] * e["tau"]
            assert e["downlink_sub"] % ((n - 1) * e["Q0"] * e["tau"]) == 0

    def test_n1_has_no_subleader_stage(self):
        """n=1 ではサブリーダー段の課金が 0 であること（Huang 2022 と同じ）。"""
        algo = run(n=1)
        assert algo.hde_comm_log
        for e in algo.hde_comm_log:
            assert e["n_subleaders_charged"] == 0
            assert e["uplink_sub"] == 0
            assert e["downlink_sub"] == 0

    @pytest.mark.parametrize("n", [2, 3])
    def test_follower_stage_counts_only_unassigned_followers(self, n):
        """フォロワー段の課金は「未割当フォロワーの最大人数」であること。"""
        algo = run(n=n)
        for e in algo.hde_comm_log:
            assert e["uplink_follower"] == (
                e["max_followers_per_group"] * e["active_arms"] * e["Q"] * e["tau"]
            )
            assert 0 <= e["max_followers_per_group"] <= M - n
