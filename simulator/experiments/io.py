"""Filesystem and reporting helpers for experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from simulator.experiments.configs import ExperimentConfig, HeteroExperimentConfig

# どちらの設定型でも受け付けられる型エイリアス
AnyConfig = Union[ExperimentConfig, HeteroExperimentConfig]

# print_summary に渡すデフォルト列セット
HOMO_SUMMARY_COLS: List[str] = [
    "cumulative_regret",
    "init_duration",
    "rank_duration",
    "number_players_duration",
    "collision_count",
    "final_assignment_success",
]
HETERO_SUMMARY_COLS: List[str] = [
    "cumulative_regret",
    "init_duration",
    "beacon_comm_duration",
    "beacon_explore_duration",
    "collision_count",
    "final_assignment_success",
]


def create_run_dir(output_root: Path, config_name: str) -> Path:
    """日付時刻と実験名から run ディレクトリを作る。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in config_name)
    base = output_root / f"{timestamp}_{safe_name}"
    run_dir = base
    suffix = 2
    while run_dir.exists():
        run_dir = Path(f"{base}_{suffix:02d}")
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def save_run_config(
    run_dir: Path, config: AnyConfig, args: argparse.Namespace
) -> None:
    """再実行に必要な設定を JSON として保存する。HomogeneousとHeterogeneous両対応。"""
    if isinstance(config, HeteroExperimentConfig):
        config_dict: Dict[str, object] = {
            "name": config.name,
            "K": config.K,
            "M": config.M,
            "T": config.T,
            "delta": config.delta,
            "means_matrix": config.means_matrix,
            "n_values": config.n_values,
            "trials": config.trials,
            "seed_base": config.seed_base,
        }
    else:
        config_dict = {
            "name": config.name,
            "K": config.K,
            "M": config.M,
            "T": config.T,
            "delta": config.delta,
            "means": config.means,
            "n_values": config.n_values,
            "trials": config.trials,
            "seed_base": config.seed_base,
        }
    payload = {"config": config_dict, "args": vars(args)}
    (run_dir / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_summary(
    summary_df: pd.DataFrame, columns: Optional[List[str]] = None
) -> None:
    """
    CLI 用の短い集計を表示する。

    Args:
        summary_df: summary CSV の DataFrame
        columns: 集計・表示する列名リスト。None なら HOMO_SUMMARY_COLS を使う。
                 存在しない列は自動スキップ。
    """
    if columns is None:
        columns = HOMO_SUMMARY_COLS
    available = [c for c in columns if c in summary_df.columns]
    grouped = summary_df.groupby(["algorithm", "n"], as_index=False)[available].mean()
    print(grouped.to_string(index=False))


def build_curve_rows(
    trace_records: List[Dict[str, object]],
    config: AnyConfig,
    algorithm: str,
    n: int,
    trial: int,
    seed: int,
    sample_points: int,
) -> List[Dict[str, object]]:
    """
    Trace から cumulative regret curve 用のサンプル行を作る。

    HomogeneousとHeterogeneous両方の設定型で動作する（K, M, T, delta を共通フィールドとして参照）。

    Args:
        trace_records: Trace.to_records() の出力
        config: 実験設定（ExperimentConfig または HeteroExperimentConfig）
        algorithm, n, trial, seed: summary と同じ識別子
        sample_points: サンプル点数

    Returns:
        時刻ごとの regret curve 行リスト
    """
    times = list(_sample_times(config.T, sample_points))
    rows: List[Dict[str, object]] = []
    idx = 0
    last_regret = 0.0
    last_phase = "not_started"

    for t in times:
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
    seen: set = set()
    for i in range(sample_points):
        t = round(i * T / (sample_points - 1))
        if t not in seen:
            seen.add(t)
            yield t


def _run_with_retry(
    run_fn: Callable[[int], Tuple[Dict[str, object], List[Dict[str, object]]]],
    seed: int,
    retry_on_failure: bool,
    max_attempts: int,
    algorithm: str,
    n: int,
    trial: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    trial 実行のリトライループ コア。

    Args:
        run_fn: seed を受け取り (summary_row, curve_rows) を返す callable
        seed: 1 attempt 目の seed
        retry_on_failure: True なら final_assignment_success=False 時に再試行
        max_attempts: 最大 attempt 数（retry_on_failure=False なら 1 で上書き）
        algorithm, n, trial: ログ用識別子

    Returns:
        (summary_row, curve_rows)
    """
    attempts = max(1, max_attempts if retry_on_failure else 1)
    last_summary: Dict[str, object] = {}
    last_curves: List[Dict[str, object]] = []

    for attempt in range(attempts):
        attempt_seed = seed + attempt * 1_000_000
        summary, curves = run_fn(attempt_seed)

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
