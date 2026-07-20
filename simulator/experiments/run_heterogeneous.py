"""Trial execution and tabular result builders for heterogeneous experiments."""

from __future__ import annotations

from typing import Dict, List, Tuple

from simulator.experiments.configs import HeteroExperimentConfig
from simulator.experiments.io import _run_with_retry, build_curve_rows


# ---- Trial runner ----

def run_beacon_trial(
    config: HeteroExperimentConfig,
    trial: int,
    seed: int,
    sample_points: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Shi 2021 BEACON を 1 trial 実行する。

    Args:
        config: 実験設定
        trial: trial 番号（0-based）
        seed: 乱数シード
        sample_points: regret curve のサンプル点数

    Returns:
        (summary_row, curve_rows)
    """
    from simulator.envs.heterogeneous_mpmab import HeterogeneousMPMABEnv
    from simulator.algorithms.heterogeneous.shi2021.algorithm import HeterogeneousShiBeacon2021
    from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner
    from simulator.utils.metrics import compute_hetero_metrics

    env = HeterogeneousMPMABEnv(config.means_matrix, collision_sensing=True, seed=seed)
    runner = HeterogeneousRunner(env, horizon=config.T)
    algo = HeterogeneousShiBeacon2021(K=config.K, M=config.M, seed=seed)
    result = algo.run(runner)

    metrics = compute_hetero_metrics(
        trace=runner.trace,
        player_states=result["player_states"],
        means_matrix=config.means_matrix,
        M=config.M,
    )
    summary = build_hetero_summary_row(
        config=config,
        algorithm="shi2021_beacon",
        n=1,
        trial=trial,
        seed=seed,
        metrics=metrics,
    )
    curves = build_curve_rows(
        trace_records=runner.trace.to_records(),
        config=config,
        algorithm="shi2021_beacon",
        n=1,
        trial=trial,
        seed=seed,
        sample_points=sample_points,
    )
    return summary, curves


def run_parallel_beacon_trial(
    config: HeteroExperimentConfig,
    trial: int,
    seed: int,
    n: int,
    sample_points: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Izumi 2026 ParallelBEACON を 1 trial 実行する。

    Args:
        config: 実験設定
        trial: trial 番号（0-based）
        seed: 乱数シード
        n: good arm の数（通信チャネル数）
        sample_points: regret curve のサンプル点数

    Returns:
        (summary_row, curve_rows)
    """
    from simulator.envs.heterogeneous_mpmab import HeterogeneousMPMABEnv
    from simulator.algorithms.heterogeneous.izumi2026.algorithm import (
        HeterogeneousMultiChannelIzumi2026,
    )
    from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner
    from simulator.utils.metrics import compute_hetero_metrics

    env = HeterogeneousMPMABEnv(config.means_matrix, collision_sensing=True, seed=seed)
    runner = HeterogeneousRunner(env, horizon=config.T)
    algo = HeterogeneousMultiChannelIzumi2026(
        K=config.K,
        M=config.M,
        n=n,
        delta=config.delta,
        seed=seed,
    )
    result = algo.run(runner)

    metrics = compute_hetero_metrics(
        trace=runner.trace,
        player_states=result["player_states"],
        means_matrix=config.means_matrix,
        M=config.M,
    )
    summary = build_hetero_summary_row(
        config=config,
        algorithm="izumi2026_parallel_beacon",
        n=n,
        trial=trial,
        seed=seed,
        metrics=metrics,
    )
    curves = build_curve_rows(
        trace_records=runner.trace.to_records(),
        config=config,
        algorithm="izumi2026_parallel_beacon",
        n=n,
        trial=trial,
        seed=seed,
        sample_points=sample_points,
    )
    return summary, curves


def run_hetero_with_optional_retry(
    config: HeteroExperimentConfig,
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

    Args:
        config: 実験設定
        algorithm: "shi2021_beacon" or "izumi2026_parallel_beacon"
        trial, seed, n, sample_points: run_*_trial と同じ
        retry_on_failure: True なら割当失敗 trial を再試行する
        max_attempts: 最大 attempt 数

    Returns:
        (summary_row, curve_rows)
    """
    if algorithm == "shi2021_beacon":
        run_fn = lambda s: run_beacon_trial(config, trial, s, sample_points)
    elif algorithm == "izumi2026_parallel_beacon":
        run_fn = lambda s: run_parallel_beacon_trial(config, trial, s, n, sample_points)
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    return _run_with_retry(run_fn, seed, retry_on_failure, max_attempts, algorithm, n, trial)


def build_hetero_summary_row(
    config: HeteroExperimentConfig,
    algorithm: str,
    n: int,
    trial: int,
    seed: int,
    metrics: Dict[str, object],
) -> Dict[str, object]:
    """
    CSV summary 1 行分を作る。

    Args:
        config: 実験設定
        algorithm: アルゴリズム識別名
        n: good arm の数（shi2021_beacon では 1 で固定）
        trial: trial 番号
        seed: 乱数シード
        metrics: compute_hetero_metrics の出力

    Returns:
        CSV 行辞書
    """
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
        "ortho_duration",
        "rank_assignment_duration",
        "init_sample_duration",
        "beacon_comm_duration",
        "beacon_explore_duration",
        "collision_count",
        "optimal_matching_reward",
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
