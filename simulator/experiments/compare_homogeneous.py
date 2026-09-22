"""Huang 2022・Izumi 2026・Randomized Selfish KL-UCB の homogeneous 設定の比較実験 CLI。

実行例:
    python -m simulator.experiments.compare_homogeneous --experiment small --trials 20
    python -m simulator.experiments.compare_homogeneous --experiment speedup --trials 50
    # arm の並びを trial ごとにランダムにし、Randomized Selfish KL-UCB も比較する
    python -m simulator.experiments.compare_homogeneous --experiment tradeoff --shuffle-arms \
        --algorithms huang2022,izumi2026,trinh2021_rskl --workers 8

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
from simulator.experiments.run_homogeneous import (
    RSKL_ALGORITHMS,
    run_rskl_trials,
    run_with_optional_retry,
    trial_means,
)
from simulator.utils import save_metric_bar, save_regret_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --algorithms で指定できるアルゴリズム名
ALGORITHM_CHOICES = ("huang2022", "izumi2026", *RSKL_ALGORITHMS)


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
        "--log-xscale",
        action="store_true",
        help="regret curve の x 軸を対数スケールにする（初期フェーズの重なり解消に有効）。",
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
        "--algorithms",
        type=str,
        default="huang2022,izumi2026",
        help=f"実行するアルゴリズム名をカンマ区切りで指定。選択肢: {','.join(ALGORITHM_CHOICES)}",
    )
    parser.add_argument(
        "--shuffle-arms",
        action="store_true",
        help="trial ごとに arm の並び（index と期待値の対応）をランダムに入れ替える。",
    )
    parser.add_argument(
        "--allow-out-of-range-n",
        action="store_true",
        help="論文の仮定 (A1) の範囲外の n（n > M または 2n > K）も実行する。summary の n_within_assumptions=0 で区別される。",
    )
    parser.add_argument(
        "--rskl-batch-size",
        type=int,
        default=25,
        help="Randomized Selfish KL-UCB を 1 プロセスでまとめて計算する trial 数。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Randomized Selfish KL-UCB のバッチを並列に動かすプロセス数。",
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

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    unknown = [a for a in algorithms if a not in ALGORITHM_CHOICES]
    if unknown:
        raise ValueError(f"未知のアルゴリズム名: {unknown}。選択肢: {ALGORITHM_CHOICES}")

    config = build_config(args)
    run_dir = create_run_dir(PROJECT_ROOT / args.output_root, config.name)

    summary_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []

    for trial in range(config.trials):
        seed = config.seed_base + trial
        # 同じ trial の全アルゴリズムで同じ arm の並びを使う（--shuffle-arms なしなら元の並び）
        means, arm_order = trial_means(config, seed)

        if "huang2022" in algorithms:
            summary, curves = run_with_optional_retry(
                config=config,
                algorithm="huang2022",
                trial=trial,
                seed=seed,
                n=1,
                sample_points=args.sample_points,
                retry_on_failure=args.retry_on_failure,
                max_attempts=args.max_attempts,
                means=means,
                arm_order=arm_order,
            )
            summary_rows.append(summary)
            curve_rows.extend(curves)

        if "izumi2026" in algorithms:
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
                    means=means,
                    arm_order=arm_order,
                )
                summary_rows.append(summary)
                curve_rows.extend(curves)

    for rskl_algorithm in RSKL_ALGORITHMS:
        if rskl_algorithm not in algorithms:
            continue
        # Randomized Selfish KL-UCB は horizon まで毎ステップ動くため、trial をまとめて numpy で計算する
        summary, curves = run_rskl_trials(
            config,
            sample_points=args.sample_points,
            batch_size=args.rskl_batch_size,
            workers=args.workers,
            algorithm=rskl_algorithm,
        )
        summary_rows.extend(summary)
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
            log_xscale=args.log_xscale,
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
