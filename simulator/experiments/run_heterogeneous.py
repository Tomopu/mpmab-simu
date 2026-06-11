"""Trial execution and tabular result builders for heterogeneous experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class HeteroExperimentConfig:
    """
    Heterogeneous 比較実験の設定。

    Attributes:
        name: 実験識別名
        K: arm 数
        M: プレイヤー数
        T: horizon
        means_matrix: 報酬行列 means_matrix[m][k]（プレイヤー × arm）
        n_values: Izumi 2026 の n（good arm 数 / チャネル数）候補リスト
        trials: trial 数
        seed_base: 1 trial 目の seed
    """

    name: str
    K: int
    M: int
    T: int
    means_matrix: List[List[float]]
    n_values: List[int]
    trials: int
    seed_base: int = 20260510

    @property
    def delta(self) -> float:
        return 1.0 / (self.T * math.log(self.T))


# ---- プリセット設定 ----

HETERO_CONFIGS: Dict[str, HeteroExperimentConfig] = {
    "small": HeteroExperimentConfig(
        name="hetero_small_sanity",
        K=5,
        M=2,
        T=500_000,
        means_matrix=[
            [0.9, 0.7, 0.5, 0.3, 0.1],  # player 0
            [0.1, 0.3, 0.5, 0.7, 0.9],  # player 1
        ],
        n_values=[2],
        trials=10,
    ),
    "asym": HeteroExperimentConfig(
        name="hetero_asymmetric",
        K=6,
        M=3,
        T=1_000_000,
        means_matrix=[
            [0.9, 0.6, 0.4, 0.3, 0.2, 0.1],  # player 0: prefers arm 0
            [0.2, 0.8, 0.5, 0.3, 0.1, 0.1],  # player 1: prefers arm 1
            [0.1, 0.2, 0.3, 0.7, 0.4, 0.1],  # player 2: prefers arm 3
        ],
        n_values=[2, 3],
        trials=20,
    ),
}


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
    attempts = max(1, max_attempts if retry_on_failure else 1)
    last_summary: Dict[str, object] = {}
    last_curves: List[Dict[str, object]] = []

    for attempt in range(attempts):
        attempt_seed = seed + attempt * 1_000_000
        if algorithm == "shi2021_beacon":
            summary, curves = run_beacon_trial(config, trial, attempt_seed, sample_points)
        elif algorithm == "izumi2026_parallel_beacon":
            summary, curves = run_parallel_beacon_trial(
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

        if bool(summary.get("final_assignment_success")):
            return summary, curves

        if not retry_on_failure:
            return summary, curves

        print(
            f"retry: algorithm={algorithm} n={n} trial={trial} "
            f"attempt={attempt + 1}/{attempts} seed={attempt_seed} failed"
        )

    last_summary["retry_exhausted"] = 1
    return last_summary, last_curves


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


def build_curve_rows(
    trace_records: List[Dict[str, object]],
    config: HeteroExperimentConfig,
    algorithm: str,
    n: int,
    trial: int,
    seed: int,
    sample_points: int,
) -> List[Dict[str, object]]:
    """
    Trace から cumulative regret curve 用のサンプル行を作る。

    Args:
        trace_records: Trace.to_records() の出力
        config: 実験設定
        algorithm, n, trial, seed: summary と同じ識別子
        sample_points: サンプル点数

    Returns:
        時刻ごとの regret curve 行リスト
    """
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
