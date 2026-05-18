"""
plotter.py: 比較実験の CSV からグラフを生成するユーティリティ。

実験スクリプト側では pandas DataFrame を作り、このモジュールに渡して PNG を保存する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def save_regret_curve(
    curves: pd.DataFrame,
    output_path: str | Path,
    confidence: float = 0.95,
    phase_summary: pd.DataFrame | None = None,
    show_phase_boundaries: bool = True,
) -> None:
    """
    平均 cumulative regret の折れ線グラフを保存する。

    Args:
        curves: columns = algorithm, n, time, cumulative_regret を含む DataFrame。
        output_path: 保存先 PNG path。
        confidence: 信頼区間。現在は正規近似の 95% CI を想定する。
        phase_summary: trial summary DataFrame。指定すると phase 終了時刻を縦線で描く。
        show_phase_boundaries: True の場合、phase 終了時刻の平均を描く。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = (
        curves.groupby(["algorithm", "n", "time"])["cumulative_regret"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    z_value = 1.96 if confidence == 0.95 else 1.96
    grouped["sem"] = grouped["std"].fillna(0.0) / grouped["count"].pow(0.5)
    grouped["ci"] = z_value * grouped["sem"]

    for (algorithm, n), sub in grouped.groupby(["algorithm", "n"]):
        sub = sub.sort_values("time")
        label = algorithm if algorithm == "huang2022" else f"{algorithm} (n={n})"
        (line,) = ax.plot(sub["time"], sub["mean"], label=label, linewidth=2)
        lower = sub["mean"] - sub["ci"]
        upper = sub["mean"] + sub["ci"]
        ax.fill_between(
            sub["time"],
            lower,
            upper,
            color=line.get_color(),
            alpha=0.18,
            linewidth=0,
        )

    if show_phase_boundaries and phase_summary is not None and not phase_summary.empty:
        _draw_phase_boundaries(ax, phase_summary)

    ax.set_xlabel("time")
    ax.set_ylabel("average cumulative regret")
    condition = _format_experiment_condition(curves)
    title = f"Cumulative Regret with {int(confidence * 100)}% CI"
    if condition:
        title = f"{title} {condition}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _add_combined_legend(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _draw_phase_boundaries(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """algorithm/n ごとの平均 phase 終了時刻を縦線で描く。"""
    required = [
        "algorithm",
        "n",
        "find_good_duration",
        "rank_duration",
        "number_players_duration",
        "exploration_duration",
    ]
    if any(col not in summary.columns for col in required):
        return

    phase_specs = [
        ("find_good_duration", "find", (0, (1.5, 5.0))),
        ("rank_duration", "rank", (0, (8.0, 5.0))),
        ("number_players_duration", "count", (0, (8.0, 4.0, 2.0, 4.0))),
        ("exploration_duration", "explore", (0, (14.0, 4.0, 2.0, 4.0, 2.0, 4.0))),
    ]
    grouped = summary.groupby(["algorithm", "n"], as_index=False)[
        [name for name, _, _ in phase_specs]
    ].mean()

    curve_colors = {
        line.get_label(): line.get_color()
        for line in ax.get_lines()
        if not line.get_label().startswith("_")
    }
    for row in grouped.itertuples(index=False):
        algorithm = str(row.algorithm)
        n = int(row.n)
        series_label = algorithm if algorithm == "huang2022" else f"{algorithm} (n={n})"
        color = curve_colors.get(series_label, "0.45")

        elapsed = 0.0
        for duration_col, phase_label, linestyle in phase_specs:
            elapsed += float(getattr(row, duration_col))
            if elapsed <= 0:
                continue
            ax.axvline(
                elapsed,
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
                alpha=0.55,
            )


def _add_combined_legend(ax: plt.Axes) -> None:
    """Curve color and phase-line style legends without overlapping plot labels."""
    curve_handles, curve_labels = ax.get_legend_handles_labels()
    phase_handles = [
        Line2D([0], [0], color="0.35", linestyle=(0, (1.5, 5.0)), alpha=0.55, linewidth=1.6, label="find end"),
        Line2D([0], [0], color="0.35", linestyle=(0, (8.0, 5.0)), alpha=0.55, linewidth=1.6, label="rank end"),
        Line2D([0], [0], color="0.35", linestyle=(0, (8.0, 4.0, 2.0, 4.0)), alpha=0.55, linewidth=1.6, label="count end"),
        Line2D([0], [0], color="0.35", linestyle=(0, (14.0, 4.0, 2.0, 4.0, 2.0, 4.0)), alpha=0.55, linewidth=1.6, label="explore end"),
    ]
    handles = curve_handles + phase_handles
    labels = curve_labels + [handle.get_label() for handle in phase_handles]
    ax.legend(handles, labels, fontsize=8, ncols=2)


def _format_experiment_condition(curves: pd.DataFrame) -> str:
    """Return `(K = x, M = y)` when the plotted data has a single K/M setting."""
    if "K" not in curves.columns or "M" not in curves.columns or curves.empty:
        return ""
    k_values = sorted(curves["K"].dropna().unique())
    m_values = sorted(curves["M"].dropna().unique())
    if len(k_values) != 1 or len(m_values) != 1:
        return ""
    return f"(K = {int(k_values[0])}, M = {int(m_values[0])})"


def save_metric_bar(
    summary: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    algorithm/n ごとの平均 metric を棒グラフで保存する。

    Args:
        summary: trial summary DataFrame。
        metric: 集計対象の列名。
        output_path: 保存先 PNG path。
        ylabel: y 軸ラベル。None なら metric 名を使う。
        title: タイトル。None なら metric 名を使う。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = summary.groupby(["algorithm", "n"], as_index=False)[metric].mean()
    labels = [
        row.algorithm if row.algorithm == "huang2022" else f"{row.algorithm}\nn={row.n}"
        for row in grouped.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, grouped[metric])
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or metric)
    if "success" in metric or metric.endswith("_rate"):
        ax.set_ylim(0.0, 1.05)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
