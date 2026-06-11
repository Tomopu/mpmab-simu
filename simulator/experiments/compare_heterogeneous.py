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
import json
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd

from simulator.experiments.io import create_run_dir
from simulator.experiments.run_heterogeneous import (
    HETERO_CONFIGS,
    HeteroExperimentConfig,
    build_curve_rows,
    run_hetero_with_optional_retry,
)
from simulator.utils import save_metric_bar, save_regret_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _print_hetero_summary(summary_df: pd.DataFrame) -> None:
    """CLI 用の短い集計を表示する（Heterogeneous 版）。"""
    cols = [
        "cumulative_regret",
        "init_duration",
        "beacon_comm_duration",
        "beacon_explore_duration",
        "collision_count",
        "final_assignment_success",
    ]
    # 存在する列だけ選択する
    available = [c for c in cols if c in summary_df.columns]
    grouped = summary_df.groupby(["algorithm", "n"], as_index=False)[available].mean()
    print(grouped.to_string(index=False))


def _save_hetero_run_config(
    run_dir: Path,
    config: HeteroExperimentConfig,
    args: argparse.Namespace,
) -> None:
    """Heterogeneous 実験の再実行設定を JSON として保存する。"""
    payload = {
        "config": {
            "name": config.name,
            "K": config.K,
            "M": config.M,
            "T": config.T,
            "delta": config.delta,
            "means_matrix": config.means_matrix,
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
        help="腕数 K を上書き（means_matrix も --means-matrix で合わせること）。",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=None,
        help="プレイヤー数 M を上書き（means_matrix も --means-matrix で合わせること）。",
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
    return parser


def build_hetero_config(args: argparse.Namespace) -> HeteroExperimentConfig:
    """CLI 引数で preset を上書きした HeteroExperimentConfig を作る。"""
    base = HETERO_CONFIGS[args.experiment]

    K = args.K or base.K
    M = args.M or base.M
    T = args.horizon or base.T

    # means_matrix 上書き
    if args.means_matrix:
        means_matrix = json.loads(args.means_matrix)
        if len(means_matrix) != M:
            raise ValueError(
                f"means_matrix の行数 ({len(means_matrix)}) が M ({M}) と一致しない。"
            )
        for row in means_matrix:
            if len(row) != K:
                raise ValueError(
                    f"means_matrix の各行の長さ ({len(row)}) が K ({K}) と一致しない。"
                )
    else:
        means_matrix = base.means_matrix

    if M >= K:
        raise ValueError(f"M < K が必要。M={M}, K={K}")

    # n_values 検証: BEACON は K >= M+1 を必要とする
    if K < M + 1:
        raise ValueError(
            f"BEACON は K >= M+1 を必要とする。K={K}, M={M}"
        )

    if args.n_values:
        n_values = [int(x) for x in args.n_values.split(",") if x.strip()]
    else:
        n_values = list(base.n_values)

    # Izumi 2026 の制約: 1 <= n < K - M
    max_n = K - M - 1
    n_values = [n for n in n_values if 1 <= n <= max_n]
    if not n_values:
        raise ValueError(
            f"有効な n がない。ParallelBEACON では 1 <= n < K-M が必要。K-M={K - M}"
        )

    suffix = args.name_suffix or f"K{K}_M{M}_T{T}"
    name = f"{base.name}_{suffix}" if suffix else base.name

    return HeteroExperimentConfig(
        name=name,
        K=K,
        M=M,
        T=T,
        means_matrix=means_matrix,
        n_values=n_values,
        trials=args.trials or base.trials,
        seed_base=base.seed_base,
    )


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
    _save_hetero_run_config(run_dir, config, args)

    if not args.no_plots:
        save_regret_curve(
            curve_df,
            run_dir / "regret_beacon_vs_parallel_beacon.png",
            confidence=0.95,
            phase_summary=summary_df,
            show_phase_boundaries=not args.no_phase_lines,
            log_xscale=args.log_xscale,
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
    _print_hetero_summary(summary_df)


if __name__ == "__main__":
    main()
