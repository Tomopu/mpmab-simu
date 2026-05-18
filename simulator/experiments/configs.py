"""Experiment presets and CLI override helpers."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List


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


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """CLI 引数で preset を上書きした ExperimentConfig を作る。"""
    base = CONFIGS[args.experiment]
    K = args.K or base.K
    M = args.M or base.M
    T = args.horizon or base.T
    if args.means:
        means = parse_means(args.means)
    elif args.K is not None and args.K != base.K:
        means = generate_means(K, args.mean_high, args.mean_low)
    else:
        means = list(base.means)
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
