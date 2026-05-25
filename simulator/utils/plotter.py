"""
plotter.py: 比較実験の CSV からグラフを生成するユーティリティ。

実験スクリプト側では pandas DataFrame を作り、このモジュールに渡して PNG を保存する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ============================================================
# フェーズ境界線のスタイル定義
#   linestyle は単純なパターンにして小サイズの図でも視認できるようにする
# ============================================================
_PHASE_SPECS = [
    ("find_good_duration",      "find",    (0, (2, 2))),
    ("rank_duration",           "rank",    (0, (6, 2))),
    ("number_players_duration", "count",   "--"),
    ("exploration_duration",    "explore", "-."),
]

# Huang2022 固定スタイル（黒太破線にして izumi2026 の曲線と明確に区別する）
_HUANG_STYLE: Dict = dict(color="#111111", linestyle="--", linewidth=2.5, zorder=5)


def _assign_curve_styles(all_groups) -> Dict[Tuple, Dict]:
    """
    (algorithm, n) ペアにプロットスタイルを割り当てる。

    - huang2022 は黒太破線で固定する。
    - izumi2026 は n の昇順に tab10 カラーマップの色を割り当てる。
    """
    izumi_ns = sorted(set(n for (algo, n) in all_groups if algo != "huang2022"))
    cmap = plt.cm.get_cmap("tab10")

    def izumi_color(n: int):
        idx = izumi_ns.index(n) if n in izumi_ns else 0
        # n 数が多い場合に色が近づかないよう間引く
        return cmap(idx / max(len(izumi_ns), 1))

    styles: Dict[Tuple, Dict] = {}
    for (algo, n) in all_groups:
        if algo == "huang2022":
            styles[(algo, n)] = dict(_HUANG_STYLE)
        else:
            styles[(algo, n)] = dict(
                color=izumi_color(n),
                linestyle="-",
                linewidth=2.0,
                zorder=3,
            )
    return styles


def save_regret_curve(
    curves: pd.DataFrame,
    output_path: str | Path,
    confidence: float = 0.95,
    phase_summary: pd.DataFrame | None = None,
    show_phase_boundaries: bool = True,
    log_xscale: bool = False,
) -> None:
    """
    平均 cumulative regret の折れ線グラフを保存する。

    Args:
        curves: columns = algorithm, n, time, cumulative_regret を含む DataFrame。
        output_path: 保存先 PNG path。
        confidence: 信頼区間（正規近似の 95% CI を想定）。
        phase_summary: trial summary DataFrame。指定すると phase 終了時刻を縦線で描く。
        show_phase_boundaries: True の場合、phase 終了時刻の平均を描く。
        log_xscale: True にすると x 軸を対数スケールにする（初期フェーズの重なり解消に有効）。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped = (
        curves.groupby(["algorithm", "n", "time"])["cumulative_regret"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    z_value = 1.96 if confidence == 0.95 else 1.96
    grouped["sem"] = grouped["std"].fillna(0.0) / grouped["count"].pow(0.5)
    grouped["ci"] = z_value * grouped["sem"]

    all_groups = list(grouped.groupby(["algorithm", "n"]).groups.keys())
    styles = _assign_curve_styles(all_groups)

    # 曲線とCI帯を描く
    for (algorithm, n), sub in grouped.groupby(["algorithm", "n"]):
        sub = sub.sort_values("time")
        label = algorithm if algorithm == "huang2022" else f"izumi2026 (n={n})"
        style = styles.get((algorithm, n), {})
        (line,) = ax.plot(
            sub["time"],
            sub["mean"],
            label=label,
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            linewidth=style.get("linewidth", 2.0),
            zorder=style.get("zorder", 3),
        )
        ax.fill_between(
            sub["time"],
            sub["mean"] - sub["ci"],
            sub["mean"] + sub["ci"],
            color=line.get_color(),
            alpha=0.15,
            linewidth=0,
            zorder=2,
        )

    if log_xscale:
        # time=0 は log で扱えないため 1 以上の範囲を表示する
        xmin = max(grouped["time"].min(), 1)
        ax.set_xscale("log")
        ax.set_xlim(left=xmin)

    if show_phase_boundaries and phase_summary is not None and not phase_summary.empty:
        _draw_phase_boundaries(ax, phase_summary, styles)

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


def _draw_phase_boundaries(
    ax: plt.Axes,
    summary: pd.DataFrame,
    curve_styles: Dict[Tuple, Dict],
) -> None:
    """
    algorithm/n ごとの平均 phase 終了時刻を縦線で描く。

    改善点:
    - フェーズ境界線を曲線と同色にし、linestyle のみで種類を示す。
    - linestyle は単純なパターンに統一して小サイズでも視認できるようにする。
    - linewidth と alpha を上げて存在感を出す。
    - フェーズ名のラベルを上端に付ける。
    """
    required_cols = ["algorithm", "n"] + [col for col, _, _ in _PHASE_SPECS]
    if any(col not in summary.columns for col in required_cols):
        return

    grouped = summary.groupby(["algorithm", "n"], as_index=False)[
        [col for col, _, _ in _PHASE_SPECS]
    ].mean()

    # フェーズ終了時刻（累積）を algorithm/n ごとに計算して描く
    for row in grouped.itertuples(index=False):
        algorithm = str(row.algorithm)
        n = int(row.n)
        series_label = algorithm if algorithm == "huang2022" else f"izumi2026 (n={n})"
        style = curve_styles.get((algorithm, n), {})
        color = style.get("color", "0.4")

        elapsed = 0.0
        for col, phase_label, linestyle in _PHASE_SPECS:
            elapsed += float(getattr(row, col))
            if elapsed <= 0:
                continue
            ax.axvline(
                elapsed,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                alpha=0.65,
                zorder=4,
            )

    # フェーズ名ラベルを全 algorithm の平均位置に 1 本だけ表示する
    # （全ラベルを詰め込むと重なるため、フェーズ名を上端に小さく添える）
    grouped_all_mean = summary.groupby("algorithm", as_index=False)[
        [col for col, _, _ in _PHASE_SPECS]
    ].mean().mean(numeric_only=True)

    elapsed_for_label = 0.0
    for col, phase_label, linestyle in _PHASE_SPECS:
        elapsed_for_label += float(grouped_all_mean[col])
        if elapsed_for_label <= 0:
            continue
        ax.text(
            elapsed_for_label,
            1.0,
            f" {phase_label}",
            transform=ax.get_xaxis_transform(),
            fontsize=7,
            ha="left",
            va="bottom",
            color="0.35",
            rotation=90,
            clip_on=False,
        )


def _add_combined_legend(ax: plt.Axes) -> None:
    """曲線の凡例とフェーズ境界線の凡例をまとめて表示する。"""
    curve_handles, curve_labels = ax.get_legend_handles_labels()

    # フェーズ境界線の凡例（線種のみ、色はグレーで統一して意味を示す）
    phase_handles = [
        Line2D([0], [0], color="0.35", linestyle=(0, (2, 2)), linewidth=2.0, alpha=0.8, label="find end"),
        Line2D([0], [0], color="0.35", linestyle=(0, (6, 2)), linewidth=2.0, alpha=0.8, label="rank end"),
        Line2D([0], [0], color="0.35", linestyle="--",        linewidth=2.0, alpha=0.8, label="count end"),
        Line2D([0], [0], color="0.35", linestyle="-.",        linewidth=2.0, alpha=0.8, label="explore end"),
    ]
    handles = curve_handles + phase_handles
    labels = curve_labels + [h.get_label() for h in phase_handles]
    ax.legend(handles, labels, fontsize=8, ncols=2, loc="upper left")


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
