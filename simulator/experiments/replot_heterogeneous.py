"""既存の heterogeneous 実験結果 CSV からグラフを再生成する CLI。

実験を再実行せずに、curves.csv / summary.csv からプロット PNG を生成し直す。
--show-epoch-phases などのオプションを追加して新しいグラフを作りたいときに使う。

実行例:
    python -m simulator.experiments.replot_heterogeneous \\
        outputs/runs/20260626_051217_hetero_large_converge_K20_M8_K20_M8_T100000

    # epoch comm/explore 切り替えシェーディングあり
    python -m simulator.experiments.replot_heterogeneous \\
        outputs/runs/20260626_051217_hetero_large_converge_K20_M8_K20_M8_T100000 \\
        --show-epoch-phases

    # 複数ディレクトリを一括処理
    python -m simulator.experiments.replot_heterogeneous \\
        outputs/runs/20260626_* --show-epoch-phases

出力:
    <run_dir>/regret_beacon_vs_parallel_beacon.png を上書き保存する。
    --suffix を指定した場合は regret_beacon_vs_parallel_beacon_<suffix>.png を別名で保存する。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from simulator.utils import save_metric_bar, save_regret_curve


def build_parser() -> argparse.ArgumentParser:
    """replot CLI の parser を作る。"""
    parser = argparse.ArgumentParser(
        description="既存の heterogeneous 実験結果 CSV からグラフを再生成する"
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        metavar="RUN_DIR",
        help="実験 run ディレクトリのパス（curves.csv / summary.csv を含むディレクトリ）",
    )
    parser.add_argument(
        "--show-epoch-phases",
        action="store_true",
        help=(
            "regret curve に BEACON エポックの comm/explore 切り替えを"
            " 背景シェーディングで表示する。trial=0 の phase 列を使用する。"
        ),
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
        "--suffix",
        type=str,
        default=None,
        help=(
            "出力ファイル名に付ける suffix。"
            " 指定すると regret_beacon_vs_parallel_beacon_<suffix>.png として保存し、"
            " 既存ファイルを上書きしない。"
        ),
    )
    parser.add_argument(
        "--no-bar-plots",
        action="store_true",
        help="棒グラフ（init_duration, collision_count など）を再生成しない。",
    )
    return parser


def replot_run_dir(run_dir: Path, args: argparse.Namespace) -> None:
    """
    1 つの run_dir に対してグラフを再生成する。

    Args:
        run_dir: curves.csv / summary.csv を含む実験ディレクトリ
        args: CLI 引数
    """
    curves_path = run_dir / "curves.csv"
    summary_path = run_dir / "summary.csv"

    if not curves_path.exists():
        print(f"[skip] curves.csv が見つかりません: {curves_path}")
        return
    if not summary_path.exists():
        print(f"[skip] summary.csv が見つかりません: {summary_path}")
        return

    curves = pd.read_csv(curves_path)
    summary = pd.read_csv(summary_path)

    # 出力ファイル名を決定する
    base_name = "regret_beacon_vs_parallel_beacon"
    if args.suffix:
        out_name = f"{base_name}_{args.suffix}.png"
    else:
        out_name = f"{base_name}.png"

    regret_path = run_dir / out_name

    save_regret_curve(
        curves,
        regret_path,
        confidence=0.95,
        phase_summary=summary,
        show_phase_boundaries=not args.no_phase_lines,
        log_xscale=args.log_xscale,
        show_epoch_phases=args.show_epoch_phases,
    )
    print(f"saved: {regret_path}")

    if not args.no_bar_plots:
        # 棒グラフは常に上書き（suffix なし）
        _save_bar_plots(summary, run_dir)


def _save_bar_plots(summary: pd.DataFrame, run_dir: Path) -> None:
    """summary から棒グラフを再生成する。"""
    bar_specs = [
        ("init_duration",       "average init duration",          "Initialization Duration (Heterogeneous)"),
        ("collision_count",     "average collision count",        "Collision Count (Heterogeneous)"),
        ("final_assignment_success", "success rate",             "Final Assignment Success Rate (Heterogeneous)"),
        ("beacon_comm_duration","average communication steps",    "Communication Duration (Heterogeneous)"),
    ]
    for col, ylabel, title in bar_specs:
        if col not in summary.columns:
            continue
        path = run_dir / f"{col}_by_algo.png"
        save_metric_bar(summary, col, path, ylabel=ylabel, title=title)
        print(f"saved: {path}")


def main() -> None:
    args = build_parser().parse_args()

    for run_dir in args.run_dirs:
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            print(f"[skip] ディレクトリではありません: {run_dir}")
            continue
        print(f"replotting: {run_dir}")
        replot_run_dir(run_dir, args)


if __name__ == "__main__":
    main()
