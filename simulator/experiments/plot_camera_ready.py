"""複数 run ディレクトリをまとめて 1 つの設定として作図する CLI。

camera ready 用の実験では、Huang 2022 と Izumi 2026 の run と Randomized Selfish
KL-UCB の run を別プロセスで走らせている。両者は同じ設定（K, M, T, means, seed_base）
なので、この CLI で summary.csv と curves.csv を連結して同じ図に描く。

run ディレクトリ名の末尾 `_rskl` `_rskl_c3` を取り除いた名前を設定名として束ねる。

実行例:
    python -m simulator.experiments.plot_camera_ready \
        --root outputs/camera_ready_20260917

出力:
    <root>/figures/<設定名>/regret_curve.png
    <root>/figures/<設定名>/regret_curve_log.png
    <root>/figures/<設定名>/regret_curve_loglog.png
    <root>/figures/<設定名>/final_regret_by_n.png
    <root>/figures/<設定名>/collision_count_by_n.png
    <root>/figures/<設定名>/success_rate_by_n.png
    <root>/figures/<設定名>/summary_table.csv
    <root>/figures/overview_final_regret.png
    <root>/figures/summary_all.csv
    <root>/figures/summary_all.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from simulator.utils import save_final_regret_by_n, save_metric_bar, save_regret_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 設定名を作るときに落とす run ディレクトリ名の末尾
_RUN_SUFFIX_PATTERN = re.compile(r"_rskl(_c\d+)?$")

# phase 境界線を描かない（フェーズを持たない）アルゴリズム
_PHASELESS_ALGORITHMS = ("trinh2021_rskl", "trinh2021_rskl_c3")


def build_parser() -> argparse.ArgumentParser:
    """作図 CLI の parser を作る。"""
    parser = argparse.ArgumentParser(
        description="複数 run ディレクトリを設定ごとにまとめて作図する"
    )
    parser.add_argument(
        "--root",
        default="outputs/camera_ready_20260917",
        help="run ディレクトリを含む親ディレクトリ。",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="図の出力先。省略すると <root>/figures を使う。",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="信頼区間。",
    )
    parser.add_argument(
        "--no-phase-lines",
        action="store_true",
        help="phase 境界の縦線を描かない。",
    )
    parser.add_argument(
        "--settings",
        default=None,
        help="作図する設定名をカンマ区切りで指定する。省略すると全設定を作図する。",
    )
    return parser


def setting_name(run_name: str) -> str:
    """run ディレクトリの config 名から設定名を作る。"""
    return _RUN_SUFFIX_PATTERN.sub("", run_name)


def collect_runs(root: Path) -> Dict[str, List[Path]]:
    """設定名ごとに run ディレクトリを集める。"""
    groups: Dict[str, List[Path]] = {}
    for config_path in sorted(root.glob("*/run_config.json")):
        config = json.loads(config_path.read_text())["config"]
        groups.setdefault(setting_name(config["name"]), []).append(config_path.parent)
    return groups


def load_group(run_dirs: List[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """run ディレクトリ群の summary と curves を連結する。"""
    summaries = [pd.read_csv(d / "summary.csv") for d in run_dirs]
    curves = [pd.read_csv(d / "curves.csv") for d in run_dirs]
    return pd.concat(summaries, ignore_index=True), pd.concat(curves, ignore_index=True)


def check_same_setting(run_dirs: List[Path]) -> None:
    """同じ設定の run だけを束ねているか確かめる。"""
    keys = []
    for run_dir in run_dirs:
        config = json.loads((run_dir / "run_config.json").read_text())["config"]
        keys.append(
            (
                config["K"],
                config["M"],
                config["T"],
                tuple(config["means"]),
                config["seed_base"],
                config["shuffle_arms"],
            )
        )
    if len(set(keys)) > 1:
        names = [d.name for d in run_dirs]
        raise ValueError(f"設定が一致しない run をまとめようとしています: {names}")


def summarize(summary: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """algorithm/n ごとの平均値の表を作る。"""
    z_value = 1.96 if confidence == 0.95 else 1.96
    grouped = (
        summary.groupby(["algorithm", "n"])
        .agg(
            trials=("cumulative_regret", "count"),
            regret_mean=("cumulative_regret", "mean"),
            regret_std=("cumulative_regret", "std"),
            collisions=("collision_count", "mean"),
            init_duration=("init_duration", "mean"),
            success_rate=("final_assignment_success", "mean"),
        )
        .reset_index()
    )
    grouped["regret_ci"] = (
        z_value * grouped["regret_std"].fillna(0.0) / grouped["trials"].pow(0.5)
    )
    return grouped.drop(columns=["regret_std"])


def to_markdown(table: pd.DataFrame) -> str:
    """集計表を markdown のテーブルにする。"""
    header = "| " + " | ".join(table.columns) + " |"
    sep = "| " + " | ".join("---" for _ in table.columns) + " |"
    lines = [header, sep]
    for row in table.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:,.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_overview(summaries: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """設定ごとの「n と最終 regret の関係」を 1 枚に並べた図を保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(summaries)
    cols = min(3, len(names))
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.0 * rows), squeeze=False)

    for idx, name in enumerate(names):
        ax = axes[idx // cols][idx % cols]
        summary = summaries[name]
        grouped = summary.groupby(["algorithm", "n"])["cumulative_regret"].mean()

        izumi = grouped.loc["izumi2026"].sort_index()
        ax.plot(izumi.index, izumi.values, marker="o", color="#1f77b4", label="Izumi 2026")
        ax.set_xticks([int(n) for n in izumi.index])

        baselines = [
            ("huang2022", "#111111", "--", "Huang 2022"),
            ("trinh2021_rskl", "#6b6b6b", ":", "R-SKL (c=0)"),
            ("trinh2021_rskl_c3", "#9a6b2f", ":", "R-SKL (c=3)"),
        ]
        for algorithm, color, linestyle, label in baselines:
            if algorithm not in grouped.index.get_level_values(0):
                continue
            value = float(grouped.loc[algorithm].mean())
            ax.axhline(value, color=color, linestyle=linestyle, linewidth=2.0, label=label)

        ax.set_yscale("log")
        ax.set_title(name.replace("good_arm_tradeoff_", ""), fontsize=10)
        ax.set_xlabel("n")
        ax.set_ylabel("average final regret")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    for idx in range(len(names), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    output_root = Path(args.output_root) if args.output_root else root / "figures"
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    groups = collect_runs(root)
    if not groups:
        raise SystemExit(f"run ディレクトリが見つかりません: {root}")
    if args.settings:
        wanted = {s.strip() for s in args.settings.split(",") if s.strip()}
        groups = {k: v for k, v in groups.items() if k in wanted}

    all_tables = []
    summaries: Dict[str, pd.DataFrame] = {}
    for name, run_dirs in sorted(groups.items()):
        check_same_setting(run_dirs)
        summary, curves = load_group(run_dirs)
        out_dir = output_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        phase_summary = summary[~summary["algorithm"].isin(_PHASELESS_ALGORITHMS)]
        save_regret_curve(
            curves,
            out_dir / "regret_curve.png",
            confidence=args.ci,
            phase_summary=phase_summary,
            show_phase_boundaries=not args.no_phase_lines,
        )
        save_regret_curve(
            curves,
            out_dir / "regret_curve_log.png",
            confidence=args.ci,
            phase_summary=phase_summary,
            show_phase_boundaries=not args.no_phase_lines,
            log_xscale=True,
        )
        # 桁が 3 つ違う手法を並べるので、両対数でフェーズ線なしの図も作る
        save_regret_curve(
            curves,
            out_dir / "regret_curve_loglog.png",
            confidence=args.ci,
            show_phase_boundaries=False,
            log_xscale=True,
            log_yscale=True,
        )
        save_final_regret_by_n(summary, out_dir / "final_regret_by_n.png", confidence=args.ci)
        save_metric_bar(
            summary,
            "collision_count",
            out_dir / "collision_count_by_n.png",
            ylabel="average collision count",
            title="Collision Count",
        )
        save_metric_bar(
            summary,
            "final_assignment_success",
            out_dir / "success_rate_by_n.png",
            ylabel="success rate",
            title="Final Top-M Assignment Success Rate",
        )

        table = summarize(summary, confidence=args.ci)
        table.to_csv(out_dir / "summary_table.csv", index=False)
        (out_dir / "summary_table.md").write_text(to_markdown(table) + "\n")

        summaries[name] = summary
        table.insert(0, "setting", name)
        all_tables.append(table)
        print(f"{name}: {len(run_dirs)} run, {len(summary)} trial rows -> {out_dir}")

    save_overview(summaries, output_root / "overview_final_regret.png")

    merged = pd.concat(all_tables, ignore_index=True)
    merged.to_csv(output_root / "summary_all.csv", index=False)
    (output_root / "summary_all.md").write_text(to_markdown(merged) + "\n")
    print(f"summary: {output_root / 'summary_all.csv'}")


if __name__ == "__main__":
    main()
