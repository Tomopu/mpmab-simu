"""
plotter.py: 比較実験の CSV からグラフを生成するユーティリティ。

実験スクリプト側では pandas DataFrame を作り、このモジュールに渡して PNG を保存する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def save_regret_curve(
    curves: pd.DataFrame,
    output_path: str | Path,
    confidence: float = 0.95,
) -> None:
    """
    平均 cumulative regret の折れ線グラフを保存する。

    Args:
        curves: columns = algorithm, n, time, cumulative_regret を含む DataFrame。
        output_path: 保存先 PNG path。
        confidence: 信頼区間。現在は正規近似の 95% CI を想定する。
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

    ax.set_xlabel("time")
    ax.set_ylabel("average cumulative regret")
    ax.set_title(f"Cumulative Regret with {int(confidence * 100)}% CI")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


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
