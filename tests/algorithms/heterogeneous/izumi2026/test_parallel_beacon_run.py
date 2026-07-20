"""
ParallelBEACON (Izumi 2026 Heterogeneous) エンドツーエンドテスト。

確認ポイント:
- run() が player_states と phase_durations を含む結果を返す
- 最終割当に重複がない
- grand leader が is_grand_leader=True を持つ
- 同じ seed で同じ結果が得られる
- horizon 短縮でもエラーにならない
- 高報酬設定で最終割当が近似最適に収束する傾向がある
- compute_hetero_metrics が正しい指標を計算する
"""

import pytest

from simulator.envs.heterogeneous_mpmab import HeterogeneousMPMABEnv
from simulator.algorithms.heterogeneous.izumi2026.algorithm import (
    HeterogeneousMultiChannelIzumi2026,
)
from simulator.algorithms.heterogeneous.izumi2026.states import ParallelBeaconPlayerState
from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner
from simulator.utils.metrics import compute_hetero_metrics


def run_parallel_beacon(
    K: int,
    M: int,
    n: int,
    means,
    seed: int = 0,
    horizon: int = 200000,
):
    """ParallelBEACON を実行して (result, runner) を返す。"""
    env = HeterogeneousMPMABEnv(means, collision_sensing=True, seed=seed)
    runner = HeterogeneousRunner(env, horizon=horizon)
    algo = HeterogeneousMultiChannelIzumi2026(K=K, M=M, n=n, delta=1e-2, seed=seed)
    result = algo.run(runner)
    return result, runner


class TestParallelBeaconRun:
    """ParallelBEACON エンドツーエンドテスト。"""

    def test_result_has_player_states_and_phase_durations(self) -> None:
        """run() が player_states と phase_durations を含む結果を返す。"""
        K, M, n = 5, 2, 2
        means = [[0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
        result, _ = run_parallel_beacon(K, M, n, means, seed=0, horizon=20000)

        assert "player_states" in result, "player_states キーがない"
        assert "phase_durations" in result, "phase_durations キーがない"
        assert len(result["player_states"]) == M

    def test_player_states_type(self) -> None:
        """player_states の各要素が ParallelBeaconPlayerState 型である。"""
        K, M, n = 5, 2, 2
        means = [[0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
        result, _ = run_parallel_beacon(K, M, n, means, seed=0, horizon=20000)

        for ps in result["player_states"]:
            assert isinstance(ps, ParallelBeaconPlayerState)

    def test_final_assignment_no_duplicates(self) -> None:
        """最終割当に重複がない（各プレイヤーが異なる arm を持つ）。"""
        K, M, n = 5, 2, 2
        means = [[0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
        result, _ = run_parallel_beacon(K, M, n, means, seed=0, horizon=100000)

        states: list[ParallelBeaconPlayerState] = result["player_states"]
        arms = [s.assigned_arm for s in states]
        assert len(set(arms)) == M, f"割当 arm に重複あり: {arms}"

    def test_grand_leader_flag(self) -> None:
        """is_grand_leader=True のプレイヤーがちょうど 1 人存在する。"""
        K, M, n = 5, 2, 2
        means = [[0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
        result, _ = run_parallel_beacon(K, M, n, means, seed=0, horizon=100000)

        states: list[ParallelBeaconPlayerState] = result["player_states"]
        grand_leaders = [s for s in states if s.is_grand_leader]
        assert len(grand_leaders) == 1, f"grand leader が 1 人でない: {len(grand_leaders)}"

    def test_grand_leader_internal_rank_is_1(self) -> None:
        """grand leader の internal_rank_j が 1 である。"""
        K, M, n = 5, 2, 2
        means = [[0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
        result, _ = run_parallel_beacon(K, M, n, means, seed=0, horizon=100000)

        states: list[ParallelBeaconPlayerState] = result["player_states"]
        grand_leaders = [s for s in states if s.is_grand_leader]
        assert grand_leaders[0].internal_rank_j == 1

    def test_seed_reproducibility(self) -> None:
        """同じ seed で同じ最終割当が得られる。"""
        K, M, n = 5, 2, 2
        means = [[0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
        r1, _ = run_parallel_beacon(K, M, n, means, seed=7, horizon=50000)
        r2, _ = run_parallel_beacon(K, M, n, means, seed=7, horizon=50000)

        a1 = [ps.assigned_arm for ps in r1["player_states"]]
        a2 = [ps.assigned_arm for ps in r2["player_states"]]
        assert a1 == a2, f"seed=7 で再現性なし: {a1} != {a2}"

    def test_horizon_reached_gracefully(self) -> None:
        """horizon が短すぎてもエラーにならず部分結果を返す。"""
        K, M, n = 5, 2, 2
        means = [[0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
        result, runner = run_parallel_beacon(K, M, n, means, seed=0, horizon=30)

        assert "player_states" in result
        assert runner.elapsed <= 30

    def test_invalid_n_raises(self) -> None:
        """n >= K - M のとき ValueError が上がる。"""
        with pytest.raises(ValueError):
            HeterogeneousMultiChannelIzumi2026(K=4, M=2, n=3, delta=1e-2)

    def test_phase_durations_has_init_phases(self) -> None:
        """phase_durations に初期化フェーズのキーが含まれる。"""
        K, M, n = 5, 2, 2
        means = [[0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
        result, _ = run_parallel_beacon(K, M, n, means, seed=0, horizon=50000)

        phases = result["phase_durations"]
        # 初期化フェーズキーのいずれかが存在する（FindMultipleGoodArms or ParallelVMC など）
        has_init = any(
            k.startswith("find_multiple_good_arms")
            or k.startswith("parallel_virtual")
            or k.startswith("parallel_beacon_initial_sampling")
            for k in phases
        )
        assert has_init, f"初期化フェーズキーが phase_durations にない: {list(phases.keys())}"

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_near_optimal_assignment_k5_m2(self, seed: int) -> None:
        """
        明確な差がある means 行列で、最終割当が近似最適に収束する傾向を確認。

        means = [[0.9, 0.1, 0.5, 0.3, 0.2],
                 [0.1, 0.9, 0.3, 0.5, 0.2]]
        最適: player0→arm0(0.9), player1→arm1(0.9), 期待報酬和 = 1.8
        """
        K, M, n = 5, 2, 2
        means = [[0.9, 0.1, 0.5, 0.3, 0.2], [0.1, 0.9, 0.3, 0.5, 0.2]]
        result, runner = run_parallel_beacon(K, M, n, means, seed=seed, horizon=500000)

        states: list[ParallelBeaconPlayerState] = result["player_states"]
        arms = [s.assigned_arm for s in states]
        optimal_arms_set = {0, 1}
        assert set(arms) == optimal_arms_set, (
            f"seed={seed}: 割当 {arms} が最適 {optimal_arms_set} に一致しない"
            f"(horizon={runner.elapsed})"
        )


class TestComputeHeteroMetrics:
    """compute_hetero_metrics のユニットテスト。"""

    def test_metrics_keys_exist(self) -> None:
        """主要な指標キーが metrics に存在する。"""
        K, M, n = 5, 2, 2
        means = [[0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
        result, runner = run_parallel_beacon(K, M, n, means, seed=0, horizon=30000)

        metrics = compute_hetero_metrics(
            runner.trace,
            player_states=result["player_states"],
            means_matrix=means,
            M=M,
        )

        expected_keys = [
            "cumulative_regret",
            "total_reward",
            "total_steps",
            "collision_count",
            "phase_durations",
            "init_duration",
            "ortho_duration",
            "rank_assignment_duration",
            "beacon_comm_duration",
            "beacon_explore_duration",
            "optimal_matching_reward",
            "rank_assignment_success",
            "assignment_duplicate",
            "final_assignment_success",
        ]
        for key in expected_keys:
            assert key in metrics, f"指標キー '{key}' がない"

    def test_cumulative_regret_nonnegative(self) -> None:
        """累積 regret が非負である。"""
        K, M, n = 5, 2, 2
        means = [[0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
        result, runner = run_parallel_beacon(K, M, n, means, seed=0, horizon=30000)

        metrics = compute_hetero_metrics(runner.trace, M=M)
        assert metrics["cumulative_regret"] >= 0

    def test_total_steps_matches_elapsed(self) -> None:
        """total_steps が runner.elapsed と一致する。"""
        K, M, n = 5, 2, 2
        means = [[0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
        result, runner = run_parallel_beacon(K, M, n, means, seed=0, horizon=30000)

        metrics = compute_hetero_metrics(runner.trace, M=M)
        assert metrics["total_steps"] == runner.elapsed

    def test_optimal_matching_reward_positive(self) -> None:
        """optimal_matching_reward が正の値である（有効な環境から取得）。"""
        K, M, n = 5, 2, 2
        means = [[0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
        result, runner = run_parallel_beacon(K, M, n, means, seed=0, horizon=5000)

        metrics = compute_hetero_metrics(
            runner.trace,
            player_states=result["player_states"],
            means_matrix=means,
            M=M,
        )
        opt = metrics["optimal_matching_reward"]
        if opt is not None:
            assert opt > 0, f"optimal_matching_reward が非正: {opt}"

    def test_assignment_duplicate_false_after_full_run(self) -> None:
        """十分な horizon で割当に重複がない（assignment_duplicate=False）。"""
        K, M, n = 5, 2, 2
        means = [[0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
        result, runner = run_parallel_beacon(K, M, n, means, seed=0, horizon=100000)

        metrics = compute_hetero_metrics(
            runner.trace,
            player_states=result["player_states"],
            means_matrix=means,
            M=M,
        )
        assert metrics["assignment_duplicate"] is False, (
            f"割当に重複あり: {[ps.assigned_arm for ps in result['player_states']]}"
        )

    def test_metrics_without_player_states(self) -> None:
        """player_states=None でも基本指標は計算できる。"""
        K, M, n = 5, 2, 2
        means = [[0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
        _, runner = run_parallel_beacon(K, M, n, means, seed=0, horizon=10000)

        metrics = compute_hetero_metrics(runner.trace)
        assert "cumulative_regret" in metrics
        assert metrics["rank_assignment_success"] is None
        assert metrics["assignment_duplicate"] is None
