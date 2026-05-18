"""Filesystem and reporting helpers for experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from simulator.experiments.configs import ExperimentConfig


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
    run_dir: Path, config: ExperimentConfig, args: argparse.Namespace
) -> None:
    """再実行に必要な設定を JSON として保存する。"""
    payload = {
        "config": {
            "name": config.name,
            "K": config.K,
            "M": config.M,
            "T": config.T,
            "delta": config.delta,
            "means": config.means,
            "n_values": config.n_values,
            "trials": config.trials,
            "seed_base": config.seed_base,
        },
        "args": vars(args),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

def print_summary(summary_df: pd.DataFrame) -> None:
    """CLI 用の短い集計を表示する。"""
    cols = [
        "cumulative_regret",
        "init_duration",
        "rank_duration",
        "number_players_duration",
        "collision_count",
        "final_assignment_success",
    ]
    grouped = summary_df.groupby(["algorithm", "n"], as_index=False)[cols].mean()
    print(grouped.to_string(index=False))
