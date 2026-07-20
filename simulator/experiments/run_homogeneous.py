"""Trial execution and tabular result builders for homogeneous experiments."""

from __future__ import annotations

from typing import Dict, List, Tuple

from simulator.algorithms import HomogeneousHuang2022, HomogeneousMultiChannelIzumi2026
from simulator.core.runner import Runner
from simulator.envs import BernoulliMPMABEnv
from simulator.experiments.configs import ExperimentConfig
from simulator.experiments.io import _run_with_retry, build_curve_rows
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
    if algorithm == "huang2022":
        run_fn = lambda s: run_huang_trial(config, trial, s, sample_points)
    elif algorithm == "izumi2026":
        run_fn = lambda s: run_izumi_trial(config, trial, s, n, sample_points)
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    return _run_with_retry(run_fn, seed, retry_on_failure, max_attempts, algorithm, n, trial)


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
