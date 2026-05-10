"""
Huang 2022 と Izumi 2026 homogeneous 設定の比較実験。

実行例:
    python experiments/compare_homogeneous.py --experiment small --trials 20
    python experiments/compare_homogeneous.py --experiment speedup --trials 50

出力:
    results/<experiment>.csv
    results/<experiment>_curves.csv
    figures/regret_huang_vs_izumi.png
    figures/init_duration_by_n.png
    figures/collision_count_by_n.png
    figures/success_rate_by_n.png
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import HomogeneousHuang2022, HomogeneousMultiChannelIzumi2026
from core.runner import Runner
from envs import BernoulliMPMABEnv
from utils import compute_metrics, save_metric_bar, save_regret_curve


@dataclass(frozen=True)
class ExperimentConfig:
    """比較実験の設定。"""

    name: str
    K: int
    M: int
    T: int
    means: List[float]
    n_values: List[int]
    trials: int
    seed_base: int = 20260510

    @property
    def delta(self) -> float:
        # docs/20260510_implementation_and_experiments.md の設定に合わせる。
        return 1.0 / (self.T * math.log(self.T))


CONFIGS: Dict[str, ExperimentConfig] = {
    "small": ExperimentConfig(
        name="homogeneous_small_sanity",
        K=5,
        M=2,
        # docs の最小案は T=5_000 だが、現実装の初期化フェーズが horizon に
        # 届く seed があるため、デフォルト sanity check は完走しやすい 50_000 にする。
        T=50_000,
        means=[0.9, 0.8, 0.5, 0.2, 0.1],
        n_values=[1, 2],
        trials=20,
    ),
    "speedup": ExperimentConfig(
        name="homogeneous_multichannel_speedup",
        K=10,
        M=5,
        T=50_000,
        means=[0.9, 0.85, 0.8, 0.75, 0.7, 0.45, 0.35, 0.25, 0.15, 0.1],
        n_values=[1, 2, 3],
        trials=50,
    ),
    "tradeoff": ExperimentConfig(
        name="good_arm_tradeoff",
        K=20,
        M=5,
        T=100_000,
        means=[
            0.95,
            0.9,
            0.86,
            0.82,
            0.78,
            0.7,
            0.64,
            0.58,
            0.52,
            0.46,
            0.4,
            0.35,
            0.3,
            0.25,
            0.2,
            0.16,
            0.12,
            0.09,
            0.06,
            0.03,
        ],
        n_values=[1, 2, 4, 6],
        trials=100,
    ),
}


def main() -> None:
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
    args = parser.parse_args()

    config = build_config(args)

    summary_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []

    for trial in range(config.trials):
        seed = config.seed_base + trial
        summary, curves = run_huang_trial(config, trial, seed, args.sample_points)
        summary_rows.append(summary)
        curve_rows.extend(curves)

        for n in config.n_values:
            summary, curves = run_izumi_trial(config, trial, seed, n, args.sample_points)
            summary_rows.append(summary)
            curve_rows.extend(curves)

    summary_df = pd.DataFrame(summary_rows)
    curve_df = pd.DataFrame(curve_rows)

    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / f"{config.name}.csv"
    curves_path = results_dir / f"{config.name}_curves.csv"
    summary_df.to_csv(summary_path, index=False)
    curve_df.to_csv(curves_path, index=False)

    if not args.no_plots:
        save_regret_curve(
            curve_df,
            figures_dir / f"{config.name}_regret_huang_vs_izumi.png",
            confidence=args.ci,
        )
        save_metric_bar(
            summary_df,
            "init_duration",
            figures_dir / f"{config.name}_init_duration_by_n.png",
            ylabel="average init duration",
            title="Initialization Duration",
        )
        save_metric_bar(
            summary_df,
            "collision_count",
            figures_dir / f"{config.name}_collision_count_by_n.png",
            ylabel="average collision count",
            title="Collision Count",
        )
        save_metric_bar(
            summary_df,
            "final_assignment_success",
            figures_dir / f"{config.name}_success_rate_by_n.png",
            ylabel="success rate",
            title="Final Top-M Assignment Success Rate",
        )

    print(f"summary: {summary_path}")
    print(f"curves:  {curves_path}")
    if not args.no_plots:
        print(f"figures: {figures_dir}")
    print_summary(summary_df)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """CLI 引数で preset を上書きした ExperimentConfig を作る。"""
    base = CONFIGS[args.experiment]
    K = args.K or base.K
    M = args.M or base.M
    T = args.horizon or base.T
    means = parse_means(args.means) if args.means else generate_means(K, args.mean_high, args.mean_low)
    if len(means) != K:
        raise ValueError(f"means の長さは K と一致する必要がある。len(means)={len(means)}, K={K}")
    if M >= K:
        raise ValueError(f"M < K が必要。M={M}, K={K}")

    n_values = parse_int_list(args.n_values) if args.n_values else list(base.n_values)
    max_n = K - M - 1
    n_values = [n for n in n_values if 1 <= n <= max_n]
    if not n_values:
        raise ValueError(f"有効な n がない。Izumi 2026 実装では 1 <= n < K-M が必要。K-M={K-M}")

    suffix = args.name_suffix or f"K{K}_M{M}_T{T}"
    name = f"{base.name}_{suffix}" if suffix else base.name

    return ExperimentConfig(
        name=name,
        K=K,
        M=M,
        T=T,
        means=means,
        n_values=n_values,
        trials=args.trials or base.trials,
        seed_base=base.seed_base,
    )


def parse_means(raw: str) -> List[float]:
    """カンマ区切りの arm 平均報酬を読む。"""
    means = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if any(mu <= 0.0 or mu >= 1.0 for mu in means):
        raise ValueError("means はすべて (0, 1) の範囲にしてください。")
    return means


def parse_int_list(raw: str) -> List[int]:
    """カンマ区切り整数リストを読む。"""
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def generate_means(K: int, high: float, low: float) -> List[float]:
    """K 本の arm 平均報酬を降順に線形生成する。"""
    if not (0.0 < low < high < 1.0):
        raise ValueError("0 < mean-low < mean-high < 1 が必要。")
    if K == 1:
        return [high]
    step = (high - low) / (K - 1)
    return [high - step * i for i in range(K)]


def run_huang_trial(
    config: ExperimentConfig, trial: int, seed: int, sample_points: int
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Huang 2022 を 1 trial 実行する。"""
    env = BernoulliMPMABEnv(
        means=config.means,
        num_players=config.M,
        seed=seed,
    )
    runner = Runner(env=env, horizon=config.T)
    algo = HomogeneousHuang2022(
        K=config.K,
        M=config.M,
        delta=config.delta,
        seed=seed,
    )
    result = algo.run(runner)
    metrics = compute_metrics(
        trace=runner.trace,
        player_states=result["player_states"],
        means=config.means,
        M=config.M,
    )
    summary = build_summary_row(
        config=config,
        algorithm="huang2022",
        n=1,
        trial=trial,
        seed=seed,
        metrics=metrics,
    )
    curves = build_curve_rows(
        trace_records=runner.trace.to_records(),
        config=config,
        algorithm="huang2022",
        n=1,
        trial=trial,
        seed=seed,
        sample_points=sample_points,
    )
    return summary, curves


def run_izumi_trial(
    config: ExperimentConfig,
    trial: int,
    seed: int,
    n: int,
    sample_points: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Izumi 2026 を 1 trial 実行する。"""
    env = BernoulliMPMABEnv(
        means=config.means,
        num_players=config.M,
        seed=seed,
    )
    runner = Runner(env=env, horizon=config.T)
    algo = HomogeneousMultiChannelIzumi2026(
        K=config.K,
        M=config.M,
        n=n,
        delta=config.delta,
        seed=seed,
    )
    result = algo.run(runner)
    metrics = compute_metrics(
        trace=runner.trace,
        player_states=result["player_states"],
        means=config.means,
        M=config.M,
    )
    summary = build_summary_row(
        config=config,
        algorithm="izumi2026",
        n=n,
        trial=trial,
        seed=seed,
        metrics=metrics,
    )
    curves = build_curve_rows(
        trace_records=runner.trace.to_records(),
        config=config,
        algorithm="izumi2026",
        n=n,
        trial=trial,
        seed=seed,
        sample_points=sample_points,
    )
    return summary, curves


def build_summary_row(
    config: ExperimentConfig,
    algorithm: str,
    n: int,
    trial: int,
    seed: int,
    metrics: Dict[str, object],
) -> Dict[str, object]:
    """CSV summary 1 行分を作る。"""
    row: Dict[str, object] = {
        "algorithm": algorithm,
        "K": config.K,
        "M": config.M,
        "T": config.T,
        "delta": config.delta,
        "n": n,
        "trial": trial,
        "seed": seed,
    }

    for key in [
        "cumulative_regret",
        "total_reward",
        "total_steps",
        "init_duration",
        "find_good_duration",
        "rank_duration",
        "number_players_duration",
        "exploration_duration",
        "collision_count",
        "final_assignment_success",
        "player_count_success",
        "rank_assignment_success",
        "assignment_duplicate",
    ]:
        value = metrics.get(key)
        if isinstance(value, bool):
            value = int(value)
        row[key] = value

    return row


def build_curve_rows(
    trace_records: List[Dict[str, object]],
    config: ExperimentConfig,
    algorithm: str,
    n: int,
    trial: int,
    seed: int,
    sample_points: int,
) -> List[Dict[str, object]]:
    """Trace から平均 regret curve 用のサンプル行を作る。"""
    sample_times = list(_sample_times(config.T, sample_points))
    rows: List[Dict[str, object]] = []
    idx = 0
    last_regret = 0.0
    last_phase = "not_started"

    for t in sample_times:
        while idx < len(trace_records) and int(trace_records[idx]["time"]) <= t:
            last_regret = float(trace_records[idx]["cumulative_regret"])
            last_phase = str(trace_records[idx]["phase"])
            idx += 1
        rows.append(
            {
                "algorithm": algorithm,
                "K": config.K,
                "M": config.M,
                "T": config.T,
                "delta": config.delta,
                "n": n,
                "trial": trial,
                "seed": seed,
                "time": t,
                "cumulative_regret": last_regret,
                "phase": last_phase,
            }
        )

    return rows


def _sample_times(T: int, sample_points: int) -> Iterable[int]:
    """0 と T を含む等間隔 time grid を返す。"""
    sample_points = max(2, sample_points)
    seen = set()
    for i in range(sample_points):
        t = round(i * T / (sample_points - 1))
        if t not in seen:
            seen.add(t)
            yield t


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


if __name__ == "__main__":
    main()
