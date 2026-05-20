"""Trial execution and tabular result builders for homogeneous experiments."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from simulator.algorithms import HomogeneousHuang2022, HomogeneousMultiChannelIzumi2026
from simulator.core.runner import Runner
from simulator.envs import BernoulliMPMABEnv
from simulator.experiments.configs import ExperimentConfig
from simulator.utils import compute_metrics


def run_huang_trial(
    config: ExperimentConfig, trial: int, seed: int, sample_points: int
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Huang 2022 を 1 trial 実行する。"""
    env = BernoulliMPMABEnv(
        means=config.means,
        num_players=config.M,
        seed=seed,
    )
    runner = Runner(env=env, horizon=config.T)
    algo = HomogeneousHuang2022(
        K=config.K,
        M=config.M,
        delta=config.delta,
        seed=seed,
    )
    result = algo.run(runner)
    metrics = compute_metrics(
        trace=runner.trace,
        player_states=result["player_states"],
        means=config.means,
        M=config.M,
    )
    summary = build_summary_row(
        config=config,
        algorithm="huang2022",
        n=1,
        trial=trial,
        seed=seed,
        metrics=metrics,
    )
    curves = build_curve_rows(
        trace_records=runner.trace.to_records(),
        config=config,
        algorithm="huang2022",
        n=1,
        trial=trial,
        seed=seed,
        sample_points=sample_points,
    )
    return summary, curves

def run_izumi_trial(
    config: ExperimentConfig,
    trial: int,
    seed: int,
    n: int,
    sample_points: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Izumi 2026 を 1 trial 実行する。"""
    env = BernoulliMPMABEnv(
        means=config.means,
        num_players=config.M,
        seed=seed,
    )
    runner = Runner(env=env, horizon=config.T)
    algo = HomogeneousMultiChannelIzumi2026(
        K=config.K,
        M=config.M,
        n=n,
        delta=config.delta,
        seed=seed,
    )
    result = algo.run(runner)
    metrics = compute_metrics(
        trace=runner.trace,
        player_states=result["player_states"],
        means=config.means,
        M=config.M,
    )
    summary = build_summary_row(
        config=config,
        algorithm="izumi2026",
        n=n,
        trial=trial,
        seed=seed,
        metrics=metrics,
    )
    curves = build_curve_rows(
        trace_records=runner.trace.to_records(),
        config=config,
        algorithm="izumi2026",
        n=n,
        trial=trial,
        seed=seed,
        sample_points=sample_points,
    )
    return summary, curves

def run_with_optional_retry(
    config: ExperimentConfig,
    algorithm: str,
    trial: int,
    seed: int,
    n: int,
    sample_points: int,
    retry_on_failure: bool,
    max_attempts: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    1 trial を実行し、必要なら割当失敗時に seed を変えて再試行する。

    再試行で破棄した attempt は summary/curve には混ぜない。全 attempt が
    失敗した場合だけ、最後の失敗 attempt を retry_exhausted=1 として保存する。
    """
    attempts = max(1, max_attempts if retry_on_failure else 1)
    last_summary: Dict[str, object] = {}
    last_curves: List[Dict[str, object]] = []

    for attempt in range(attempts):
        attempt_seed = seed + attempt * 1_000_000
        if algorithm == "huang2022":
            summary, curves = run_huang_trial(
                config, trial, attempt_seed, sample_points
            )
        elif algorithm == "izumi2026":
            summary, curves = run_izumi_trial(
                config, trial, attempt_seed, n, sample_points
            )
        else:
            raise ValueError(f"unknown algorithm: {algorithm}")

        summary["attempt"] = attempt
        summary["attempts_used"] = attempt + 1
        summary["retry_exhausted"] = 0
        for row in curves:
            row["attempt"] = attempt
            row["attempts_used"] = attempt + 1

        last_summary = summary
        last_curves = curves

        if _is_success(summary):
            return summary, curves

        if not retry_on_failure:
            return summary, curves

        print(
            f"retry: algorithm={algorithm} n={n} trial={trial} "
            f"attempt={attempt + 1}/{attempts} seed={attempt_seed} failed"
        )

    last_summary["retry_exhausted"] = 1
    return last_summary, last_curves

def _is_success(summary: Dict[str, object]) -> bool:
    """実験再試行で成功扱いにする条件。"""
    return bool(summary.get("final_assignment_success"))

def build_summary_row(
    config: ExperimentConfig,
    algorithm: str,
    n: int,
    trial: int,
    seed: int,
    metrics: Dict[str, object],
) -> Dict[str, object]:
    """CSV summary 1 行分を作る。"""
    row: Dict[str, object] = {
        "algorithm": algorithm,
        "K": config.K,
        "M": config.M,
        "T": config.T,
        "delta": config.delta,
        "n": n,
        "trial": trial,
        "seed": seed,
    }

    for key in [
        "cumulative_regret",
        "total_reward",
        "total_steps",
        "init_duration",
        "find_good_duration",
        "rank_duration",
        "number_players_duration",
        "exploration_duration",
        "collision_count",
        "final_assignment_success",
        "player_count_success",
        "rank_assignment_success",
        "assignment_duplicate",
    ]:
        value = metrics.get(key)
        if isinstance(value, bool):
            value = int(value)
        row[key] = value

    return row

def build_curve_rows(
    trace_records: List[Dict[str, object]],
    config: ExperimentConfig,
    algorithm: str,
    n: int,
    trial: int,
    seed: int,
    sample_points: int,
) -> List[Dict[str, object]]:
    """Trace から平均 regret curve 用のサンプル行を作る。"""
    sample_times = list(_sample_times(config.T, sample_points))
    rows: List[Dict[str, object]] = []
    idx = 0
    last_regret = 0.0
    last_phase = "not_started"

    for t in sample_times:
        while idx < len(trace_records) and int(trace_records[idx]["time"]) <= t:
            last_regret = float(trace_records[idx]["cumulative_regret"])
            last_phase = str(trace_records[idx]["phase"])
            idx += 1
        rows.append(
            {
                "algorithm": algorithm,
                "K": config.K,
                "M": config.M,
                "T": config.T,
                "delta": config.delta,
                "n": n,
                "trial": trial,
                "seed": seed,
                "time": t,
                "cumulative_regret": last_regret,
                "phase": last_phase,
            }
        )

    return rows

def _sample_times(T: int, sample_points: int) -> Iterable[int]:
    """0 と T を含む等間隔 time grid を返す。"""
    sample_points = max(2, sample_points)
    seen = set()
    for i in range(sample_points):
        t = round(i * T / (sample_points - 1))
        if t not in seen:
            seen.add(t)
            yield t
