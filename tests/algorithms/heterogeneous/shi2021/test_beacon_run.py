"""
BEACON エンドツーエンドテスト。

確認ポイント:
- run() が完了して結果を返す
- 最終割当が重複しない
- rank=1 の player が leader になる
- 高報酬設定で最終割当が近似最適に収束する傾向がある
"""

import pytest

from simulator.envs.heterogeneous_mpmab import HeterogeneousMPMABEnv
from simulator.algorithms.heterogeneous.shi2021.algorithm import HeterogeneousShiBeacon2021
from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner
from simulator.algorithms.heterogeneous.shi2021.states import BeaconPlayerState


def run_beacon(K: int, M: int, means, seed: int = 0, horizon: int = 100000):
    """BEACON を実行して result を返す。"""
    env = HeterogeneousMPMABEnv(means, collision_sensing=True, seed=seed)
    runner = HeterogeneousRunner(env, horizon=horizon)
    algo = HeterogeneousShiBeacon2021(K=K, M=M, seed=seed)
    result = algo.run(runner)
    return result, runner


class TestBeaconRun:
    """BEACON エンドツーエンドテスト。"""

    def test_result_has_player_states_and_phase_durations(self) -> None:
        """run() が player_states と phase_durations を含む結果を返す。"""
        K, M = 4, 2
        means = [[0.9, 0.7, 0.3, 0.1], [0.1, 0.3, 0.7, 0.9]]
        result, _ = run_beacon(K, M, means, seed=0, horizon=10000)

        assert "player_states" in result
        assert "phase_durations" in result
        assert len(result["player_states"]) == M

    def test_final_assignment_no_duplicates(self) -> None:
        """最終割当に重複がない（各プレイヤーが異なる arm を持つ）。"""
        K, M = 4, 2
        means = [[0.9, 0.7, 0.3, 0.1], [0.1, 0.3, 0.7, 0.9]]
        result, _ = run_beacon(K, M, means, seed=0, horizon=50000)

        states: list[BeaconPlayerState] = result["player_states"]
        arms = [s.assigned_arm for s in states]
        assert len(set(arms)) == M, f"割当 arm に重複あり: {arms}"

    def test_leader_has_rank_1(self) -> None:
        """is_leader=True のプレイヤーの rank が 1 であること。"""
        K, M = 4, 2
        means = [[0.9, 0.7, 0.3, 0.1], [0.1, 0.3, 0.7, 0.9]]
        result, _ = run_beacon(K, M, means, seed=0, horizon=50000)

        states: list[BeaconPlayerState] = result["player_states"]
        leaders = [s for s in states if s.is_leader]
        assert len(leaders) == 1, "leader が 1 人でない"
        assert leaders[0].rank == 1

    def test_seed_reproducibility(self) -> None:
        """同じ seed で同じ結果が得られる。"""
        K, M = 4, 2
        means = [[0.9, 0.7, 0.3, 0.1], [0.1, 0.3, 0.7, 0.9]]
        r1, _ = run_beacon(K, M, means, seed=7, horizon=30000)
        r2, _ = run_beacon(K, M, means, seed=7, horizon=30000)

        s1 = [ps.assigned_arm for ps in r1["player_states"]]
        s2 = [ps.assigned_arm for ps in r2["player_states"]]
        assert s1 == s2

    def test_horizon_reached_gracefully(self) -> None:
        """horizon が短すぎてもエラーにならず部分結果を返す。"""
        K, M = 4, 2
        means = [[0.9, 0.7, 0.3, 0.1], [0.1, 0.3, 0.7, 0.9]]
        result, runner = run_beacon(K, M, means, seed=0, horizon=50)
        assert "player_states" in result
        assert runner.elapsed <= 50

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_near_optimal_assignment_k4_m2(self, seed: int) -> None:
        """
        明確な差がある means 行列で、最終割当が近似最適に収束する傾向を確認。

        means = [[0.9, 0.1, 0.5, 0.3],
                 [0.1, 0.9, 0.3, 0.5]]
        最適: player0→arm0(0.9), player1→arm1(0.9), 期待報酬和 = 1.8
        """
        K, M = 4, 2
        means = [[0.9, 0.1, 0.5, 0.3], [0.1, 0.9, 0.3, 0.5]]
        result, runner = run_beacon(K, M, means, seed=seed, horizon=200000)

        states: list[BeaconPlayerState] = result["player_states"]
        arms = [s.assigned_arm for s in states]
        # 最適割当は {0→arm0, 1→arm1} または等価な割当
        optimal_arms_set = {0, 1}
        assert set(arms) == optimal_arms_set, (
            f"seed={seed}: 割当 {arms} が最適 {optimal_arms_set} に一致しない"
        )

    def test_phase_durations_recorded(self) -> None:
        """phase_durations に各フェーズが記録されている。"""
        K, M = 4, 2
        means = [[0.9, 0.7, 0.3, 0.1], [0.1, 0.3, 0.7, 0.9]]
        result, _ = run_beacon(K, M, means, seed=0, horizon=50000)

        phases = result["phase_durations"]
        assert "orthogonalization" in phases
        assert "rank_assignment" in phases
        assert "beacon_initial_sampling" in phases
