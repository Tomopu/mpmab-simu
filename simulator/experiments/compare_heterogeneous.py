"""Shi 2021 BEACON と Izumi 2026 ParallelBEACON の比較実験 CLI。

実行例:
    python -m simulator.experiments.compare_heterogeneous --experiment small --trials 10
    python -m simulator.experiments.compare_heterogeneous --experiment asym --trials 20 --no-plots

出力:
    outputs/runs/<YYYYMMDD_HHMMSS>_<experiment>/summary.csv
    outputs/runs/<YYYYMMDD_HHMMSS>_<experiment>/curves.csv
    outputs/runs/<YYYYMMDD_HHMMSS>_<experiment>/*.png
    outputs/runs/<YYYYMMDD_HHMMSS>_<experiment>/run_config.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from simulator.experiments.configs import HETERO_CONFIGS, build_hetero_config
from simulator.experiments.io import (
    HETERO_SUMMARY_COLS,
    create_run_dir,
    print_summary,
    save_run_config,
)
from simulator.experiments.run_heterogeneous import run_hetero_with_optional_retry
from simulator.utils import save_metric_bar, save_regret_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """比較実験 CLI の parser を作る。"""
    parser = argparse.ArgumentParser(
        description="Shi 2021 BEACON と Izumi 2026 ParallelBEACON の比較実験"
    )
    parser.add_argument(
        "--experiment",
        choices=sorted(HETERO_CONFIGS),
        default="small",
        help="実験設定名",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="trial 数。未指定なら設定ファイルの値を使う。",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="horizon T。未指定なら設定ファイルの値を使う。",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=300,
        help="regret curve CSV/PNG 用のサンプル点数。",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=None,
        help="腕数 K を上書き（--means-matrix も必須）。",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=None,
        help="プレイヤー数 M を上書き（--means-matrix も必須）。",
    )
    parser.add_argument(
        "--n-values",
        type=str,
        default=None,
        help="Izumi 2026 の n 値。例: 1,2",
    )
    parser.add_argument(
        "--means-matrix",
        type=str,
        default=None,
        help=(
            "報酬行列を JSON 形式で指定。例: '[[0.9,0.1],[0.1,0.9]]'"
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="CSV のみ出力し、PNG を生成しない。",
    )
    parser.add_argument(
        "--no-phase-lines",
        action="store_true",
        help="regret curve に phase 終了時刻の縦線を描かない。",
    )
    parser.add_argument(
        "--log-xscale",
        action="store_true",
        help="regret curve の x 軸を対数スケールにする。",
    )
    parser.add_argument(
        "--retry-on-failure",
        action="store_true",
        help="final_assignment_success が false の trial を seed を変えて再試行する。",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="--retry-on-failure 時の最大 attempt 数。",
    )
    parser.add_argument(
        "--name-suffix",
        type=str,
        default=None,
        help="出力ファイル名に付ける suffix。",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/runs",
        help="実験 run ディレクトリを作成する親ディレクトリ。",
    )
    parser.add_argument(
        "--no-beacon",
        action="store_true",
        help="Shi 2021 BEACON を省略し、ParallelBEACON のみ実行する。",
    )
    parser.add_argument(
        "--show-epoch-phases",
        action="store_true",
        help=(
            "regret curve に BEACON エポックの comm/explore 切り替えを背景シェーディングで表示する。"
            " trial=0 の最初の ParallelBEACON（または BEACON）の phase 列を使用する。"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts は 1 以上にしてください。")

    config = build_hetero_config(args)
    run_dir = create_run_dir(PROJECT_ROOT / args.output_root, config.name)

    summary_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []

    for trial in range(config.trials):
        seed = config.seed_base + trial

        # Shi 2021 BEACON
        if not args.no_beacon:
            summary, curves = run_hetero_with_optional_retry(
                config=config,
                algorithm="shi2021_beacon",
                trial=trial,
                seed=seed,
                n=1,
                sample_points=args.sample_points,
                retry_on_failure=args.retry_on_failure,
                max_attempts=args.max_attempts,
            )
            summary_rows.append(summary)
            curve_rows.extend(curves)

        # Izumi 2026 ParallelBEACON（n_values ごとに実行）
        for n in config.n_values:
            summary, curves = run_hetero_with_optional_retry(
                config=config,
                algorithm="izumi2026_parallel_beacon",
                trial=trial,
                seed=seed,
                n=n,
                sample_points=args.sample_points,
                retry_on_failure=args.retry_on_failure,
                max_attempts=args.max_attempts,
            )
            summary_rows.append(summary)
            curve_rows.extend(curves)

        print(
            f"trial {trial + 1}/{config.trials} done  "
            f"(seed={seed})"
        )

    summary_df = pd.DataFrame(summary_rows)
    curve_df = pd.DataFrame(curve_rows)

    summary_path = run_dir / "summary.csv"
    curves_path = run_dir / "curves.csv"
    summary_df.to_csv(summary_path, index=False)
    curve_df.to_csv(curves_path, index=False)
    save_run_config(run_dir, config, args)

    if not args.no_plots:
        save_regret_curve(
            curve_df,
            run_dir / "regret_beacon_vs_parallel_beacon.png",
            confidence=0.95,
            phase_summary=summary_df,
            show_phase_boundaries=not args.no_phase_lines,
            log_xscale=args.log_xscale,
            show_epoch_phases=args.show_epoch_phases,
        )
        save_metric_bar(
            summary_df,
            "init_duration",
            run_dir / "init_duration_by_algo.png",
            ylabel="average init duration",
            title="Initialization Duration (Heterogeneous)",
        )
        save_metric_bar(
            summary_df,
            "collision_count",
            run_dir / "collision_count_by_algo.png",
            ylabel="average collision count",
            title="Collision Count (Heterogeneous)",
        )
        save_metric_bar(
            summary_df,
            "final_assignment_success",
            run_dir / "success_rate_by_algo.png",
            ylabel="success rate",
            title="Final Assignment Success Rate (Heterogeneous)",
        )
        save_metric_bar(
            summary_df,
            "beacon_comm_duration",
            run_dir / "comm_duration_by_algo.png",
            ylabel="average communication steps",
            title="Communication Duration (Heterogeneous)",
        )

    print(f"run_dir: {run_dir}")
    print(f"summary: {summary_path}")
    print(f"curves:  {curves_path}")
    if not args.no_plots:
        print(f"figures: {run_dir}")
    print_summary(summary_df, HETERO_SUMMARY_COLS)


if __name__ == "__main__":
    main()
