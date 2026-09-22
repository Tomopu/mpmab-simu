"""
失敗した試行の評価が正しく数えられることのテスト（F24）。

確認内容:
- 重複割当は final_assignment_success にならないこと
- 割当が決まって実行を止めた後の残り時間の期待損失（expected_tail_regret）が
  cumulative_regret に加わること。正しい割当なら 0 で値が変わらないこと
- 曲線の延長が tail_loss_per_step の傾きで伸びること
- Good Arm の不一致が good_arm_agreement=False として記録され、成功と数えないこと
- 初期化の確率的な失敗（j=1 がいない）が例外にならず init_failure_reason に残ること
- 論文の仮定 (A1) の範囲外の n をコンストラクターが既定で拒み、フラグで許すこと
"""

from types import SimpleNamespace

import pytest

from simulator.algorithms.homogeneous import HomogeneousMultiChannelIzumi2026
from simulator.algorithms.homogeneous.izumi2026.results import build_result
from simulator.core.runner import Runner
from simulator.core.trace import Trace
from simulator.envs.bernoulli_mpmab import BernoulliMPMABEnv
from simulator.experiments.configs import ExperimentConfig
from simulator.experiments.io import build_curve_rows
from simulator.utils.metrics import compute_metrics

MEANS = [0.9, 0.8, 0.3, 0.2, 0.1]
TOP2 = 0.9 + 0.8


def _trace_with_steps(num_players: int, steps: int, regret_per_step: float = 1.0) -> Trace:
    """steps ステップ分の記録を持つ Trace を作る（内容は regret の値だけが意味を持つ）。"""
    trace = Trace(num_players)
    trace.set_phase("test")
    for t in range(1, steps + 1):
        trace.append_step(
            time=t,
            actions=[0] * num_players,
            rewards=[0.0] * num_players,
            collisions=[False] * num_players,
            total_reward=0.0,
            optimal_total_reward=regret_per_step,
            instant_regret=regret_per_step,
        )
    return trace


def _states(assigned, good_arms=None):
    good_arms = good_arms or [[0, 1] for _ in assigned]
    return [
        SimpleNamespace(external_rank_s=i, internal_rank_j=i + 1, M_hat=len(assigned),
                        good_arms=list(g), mu_tilde_min=0.5, assigned_arm=a)
        for i, (a, g) in enumerate(zip(assigned, good_arms))
    ]


class TestDuplicateAndTail:
    def test_duplicate_assignment_is_not_success(self):
        trace = _trace_with_steps(2, 10)
        metrics = compute_metrics(trace, _states([0, 0]), MEANS, M=2, horizon=100)
        assert metrics["assignment_duplicate"] is True
        assert metrics["final_assignment_success"] is False

    def test_correct_assignment_has_no_tail(self):
        trace = _trace_with_steps(2, 10)
        metrics = compute_metrics(trace, _states([0, 1]), MEANS, M=2, horizon=100)
        assert metrics["final_assignment_success"] is True
        assert metrics["expected_tail_regret"] == 0.0
        assert metrics["cumulative_regret"] == metrics["regret_at_stop"]

    def test_wrong_assignment_adds_expected_tail(self):
        trace = _trace_with_steps(2, 10)
        metrics = compute_metrics(trace, _states([2, 0]), MEANS, M=2, horizon=100)
        loss = TOP2 - (0.3 + 0.9)
        assert metrics["final_assignment_success"] is False
        assert metrics["tail_loss_per_step"] == pytest.approx(loss)
        assert metrics["expected_tail_regret"] == pytest.approx(90 * loss)
        assert metrics["cumulative_regret"] == pytest.approx(metrics["regret_at_stop"] + 90 * loss)

    def test_duplicate_arms_count_as_collision_in_tail(self):
        trace = _trace_with_steps(2, 10)
        metrics = compute_metrics(trace, _states([0, 0]), MEANS, M=2, horizon=100)
        # 2 人とも腕 0 なら衝突して報酬 0。未割当も 0 として数える。
        assert metrics["tail_loss_per_step"] == pytest.approx(TOP2)

    def test_no_tail_without_horizon(self):
        trace = _trace_with_steps(2, 10)
        metrics = compute_metrics(trace, _states([2, 0]), MEANS, M=2)
        assert metrics["expected_tail_regret"] == 0.0
        assert metrics["cumulative_regret"] == metrics["regret_at_stop"]


class TestCurveExtension:
    def test_curve_extends_with_tail_slope(self):
        trace = _trace_with_steps(2, 10)
        config = ExperimentConfig(name="t", K=5, M=2, T=100, means=MEANS, n_values=[1], trials=1)
        rows = build_curve_rows(trace.to_records(), config, "izumi2026", 1, 0, 0,
                                sample_points=11, stop_time=10, tail_loss_per_step=0.5)
        by_time = {r["time"]: r["cumulative_regret"] for r in rows}
        assert by_time[10] == pytest.approx(10.0)
        assert by_time[50] == pytest.approx(10.0 + 0.5 * 40)
        assert by_time[100] == pytest.approx(10.0 + 0.5 * 90)

    def test_curve_flat_without_tail(self):
        trace = _trace_with_steps(2, 10)
        config = ExperimentConfig(name="t", K=5, M=2, T=100, means=MEANS, n_values=[1], trials=1)
        rows = build_curve_rows(trace.to_records(), config, "izumi2026", 1, 0, 0,
                                sample_points=11, stop_time=10, tail_loss_per_step=0.0)
        assert all(r["cumulative_regret"] == pytest.approx(10.0) for r in rows if r["time"] >= 10)


class TestGoodArmAgreement:
    def test_disagreement_is_recorded_and_not_success(self):
        trace = _trace_with_steps(2, 10)
        states = _states([0, 1], good_arms=[[0, 2], [1, 0]])
        metrics = compute_metrics(trace, states, MEANS, M=2, horizon=100)
        assert metrics["good_arm_agreement"] is False
        assert metrics["final_assignment_success"] is False

    def test_agreement_uses_sets(self):
        trace = _trace_with_steps(2, 10)
        states = _states([0, 1], good_arms=[[0, 1], [1, 0]])
        metrics = compute_metrics(trace, states, MEANS, M=2, horizon=100)
        assert metrics["good_arm_agreement"] is True
        assert metrics["final_assignment_success"] is True

    def test_build_result_keeps_per_player_good_arms(self):
        env = BernoulliMPMABEnv(means=MEANS, num_players=2, seed=0)
        runner = Runner(env=env, horizon=10)
        result = build_result(2, [0, 2], {0: 0.125, 2: 0.125}, [0, 1], [1, 2], [2, 2], [0, 1], runner,
                              player_good_arms=[[0, 2], [1, 0]],
                              player_mu_tilde=[{0: 0.125, 2: 0.125}, {1: 0.25, 0: 0.125}])
        assert [ps.good_arms for ps in result["player_states"]] == [[0, 2], [1, 0]]
        assert result["init_failure_reason"] is None


class TestInitializationFailure:
    def test_missing_leader_is_recorded_not_raised(self, monkeypatch):
        env = BernoulliMPMABEnv(means=MEANS, num_players=2, seed=0)
        runner = Runner(env=env, horizon=200_000)
        algo = HomogeneousMultiChannelIzumi2026(K=5, M=2, n=2, delta=0.1, seed=0)
        # 内部ランクの誤推定で j=1 がいない状況を作る
        monkeypatch.setattr(algo, "parallel_virtual_number_players", lambda runner, G, s, tau: ([2, 2], [2, 2]))
        result = algo.run(runner)
        assert result["init_failure_reason"] is not None
        assert "Grand Leader" in result["init_failure_reason"]
        assert all(ps.assigned_arm == -1 for ps in result["player_states"])
        metrics = compute_metrics(runner.trace, result["player_states"], MEANS, M=2, horizon=200_000)
        assert metrics["final_assignment_success"] is False
        # 未割当のまま止まったので、残り時間は最大損失で数える
        assert metrics["tail_loss_per_step"] == pytest.approx(TOP2)
        assert metrics["expected_tail_regret"] > 0

    def test_normal_run_has_no_failure_reason(self):
        env = BernoulliMPMABEnv(means=MEANS, num_players=2, seed=0)
        runner = Runner(env=env, horizon=2_000_000)
        algo = HomogeneousMultiChannelIzumi2026(K=5, M=2, n=2, delta=0.1, seed=0)
        result = algo.run(runner)
        assert result["init_failure_reason"] is None
        assert [ps.good_arms for ps in result["player_states"]] == [algo.fmga_player_good_arms[m] for m in range(2)]


class TestRangeCheck:
    def test_out_of_range_n_is_rejected_by_default(self):
        with pytest.raises(ValueError):
            HomogeneousMultiChannelIzumi2026(K=7, M=2, n=4, delta=0.1, seed=0)
        with pytest.raises(ValueError):
            HomogeneousMultiChannelIzumi2026(K=10, M=4, n=5, delta=0.1, seed=0)

    def test_out_of_range_n_is_allowed_with_flag(self):
        algo = HomogeneousMultiChannelIzumi2026(K=10, M=4, n=5, delta=0.1, seed=0, allow_out_of_range_n=True)
        assert algo.n == 5

    def test_in_range_n_is_accepted(self):
        algo = HomogeneousMultiChannelIzumi2026(K=8, M=4, n=3, delta=0.1, seed=0)
        assert algo.n == 3
