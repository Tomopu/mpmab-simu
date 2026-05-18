"""Huang 2022 と Izumi 2026 homogeneous 設定の比較実験 CLI。

実行例:
    python -m simulator.experiments.compare_homogeneous --experiment small --trials 20
    python -m simulator.experiments.compare_homogeneous --experiment speedup --trials 50

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

from simulator.experiments.configs import CONFIGS, build_config
from simulator.experiments.io import create_run_dir, print_summary, save_run_config
from simulator.experiments.run_homogeneous import run_with_optional_retry
from simulator.utils import save_metric_bar, save_regret_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """比較実験 CLI の parser を作る。"""
    parser = argparse.ArgumentParser(
        description="Huang 2022 と Izumi 2026 の homogeneous 比較実験"
    )
    parser.add_argument(
        "--experiment",
        choices=sorted(CONFIGS),
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
    parser.add_argument("--K", type=int, default=None, help="腕数 K を上書き。")
    parser.add_argument("--M", type=int, default=None, help="プレイヤー数 M を上書き。")
    parser.add_argument(
        "--n-values",
        type=str,
        default=None,
        help="Izumi 2026 の n 値。例: 1,2,3",
    )
    parser.add_argument(
        "--means",
        type=str,
        default=None,
        help="arm 平均報酬をカンマ区切りで指定。例: 0.9,0.8,0.5",
    )
    parser.add_argument(
        "--mean-high",
        type=float,
        default=0.9,
        help="--means 未指定時に自動生成する最大平均報酬。",
    )
    parser.add_argument(
        "--mean-low",
        type=float,
        default=0.1,
        help="--means 未指定時に自動生成する最小平均報酬。",
    )
    parser.add_argument(
        "--name-suffix",
        type=str,
        default=None,
        help="出力ファイル名に付ける suffix。K/M を変える実験の区別に使う。",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="regret curve に描く信頼区間。現在は 0.95 を想定。",
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
        "--output-root",
        type=str,
        default="outputs/runs",
        help="実験 run ディレクトリを作成する親ディレクトリ。",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts は 1 以上にしてください。")

    config = build_config(args)
    run_dir = create_run_dir(PROJECT_ROOT / args.output_root, config.name)

    summary_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []

    for trial in range(config.trials):
        seed = config.seed_base + trial
        summary, curves = run_with_optional_retry(
            config=config,
            algorithm="huang2022",
            trial=trial,
            seed=seed,
            n=1,
            sample_points=args.sample_points,
            retry_on_failure=args.retry_on_failure,
            max_attempts=args.max_attempts,
        )
        summary_rows.append(summary)
        curve_rows.extend(curves)

        for n in config.n_values:
            summary, curves = run_with_optional_retry(
                config=config,
                algorithm="izumi2026",
                trial=trial,
                seed=seed,
                n=n,
                sample_points=args.sample_points,
                retry_on_failure=args.retry_on_failure,
                max_attempts=args.max_attempts,
            )
            summary_rows.append(summary)
            curve_rows.extend(curves)

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
            run_dir / "regret_huang_vs_izumi.png",
            confidence=args.ci,
            phase_summary=summary_df,
            show_phase_boundaries=not args.no_phase_lines,
        )
        save_metric_bar(
            summary_df,
            "init_duration",
            run_dir / "init_duration_by_n.png",
            ylabel="average init duration",
            title="Initialization Duration",
        )
        save_metric_bar(
            summary_df,
            "collision_count",
            run_dir / "collision_count_by_n.png",
            ylabel="average collision count",
            title="Collision Count",
        )
        save_metric_bar(
            summary_df,
            "final_assignment_success",
            run_dir / "success_rate_by_n.png",
            ylabel="success rate",
            title="Final Top-M Assignment Success Rate",
        )

    print(f"run_dir: {run_dir}")
    print(f"summary: {summary_path}")
    print(f"curves:  {curves_path}")
    if not args.no_plots:
        print(f"figures: {run_dir}")
    print_summary(summary_df)


if __name__ == "__main__":
    main()
