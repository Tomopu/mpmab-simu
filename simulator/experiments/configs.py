"""Experiment presets and CLI override helpers."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List

# ---- Homogeneous ----


@dataclass(frozen=True)
class ExperimentConfig:
    """Homogeneous 比較実験の設定。"""

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
        M=8,
        T=1_000_000,
        means=[
            0.95, 0.9, 0.86, 0.82, 0.78,
            0.7, 0.64, 0.58, 0.52, 0.46,
            0.4, 0.35, 0.3, 0.25, 0.2,
            0.16, 0.12, 0.09, 0.06, 0.03,
        ],
        n_values=[1, 2, 3, 4, 5, 6],
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
        raise ValueError(f"有効な n がない。Izumi 2026 実装では 1 <= n < K-M が必要。K-M={K - M}")

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


# ---- Heterogeneous ----


@dataclass(frozen=True)
class HeteroExperimentConfig:
    """
    Heterogeneous 比較実験の設定。

    Attributes:
        name: 実験識別名
        K: arm 数
        M: プレイヤー数
        T: horizon
        means_matrix: 報酬行列 means_matrix[m][k]（プレイヤー × arm）
        n_values: Izumi 2026 の n（good arm 数 / チャネル数）候補リスト
        trials: trial 数
        seed_base: 1 trial 目の seed
    """

    name: str
    K: int
    M: int
    T: int
    means_matrix: List[List[float]]
    n_values: List[int]
    trials: int
    seed_base: int = 20260510

    @property
    def delta(self) -> float:
        return 1.0 / (self.T * math.log(self.T))


HETERO_CONFIGS: Dict[str, HeteroExperimentConfig] = {
    "small": HeteroExperimentConfig(
        name="hetero_small_sanity",
        K=5,
        M=2,
        T=500_000,
        means_matrix=[
            [0.9, 0.7, 0.5, 0.3, 0.1],  # player 0
            [0.1, 0.3, 0.5, 0.7, 0.9],  # player 1
        ],
        n_values=[2],
        trials=10,
    ),
    "asym": HeteroExperimentConfig(
        name="hetero_asymmetric",
        K=6,
        M=3,
        T=1_000_000,
        means_matrix=[
            [0.9, 0.6, 0.4, 0.3, 0.2, 0.1],  # player 0: prefers arm 0
            [0.2, 0.8, 0.5, 0.3, 0.1, 0.1],  # player 1: prefers arm 1
            [0.1, 0.2, 0.3, 0.7, 0.4, 0.1],  # player 2: prefers arm 3
        ],
        n_values=[2, 3],
        trials=20,
    ),
    "large": HeteroExperimentConfig(
        # homogeneous tradeoff と同じ means 値を shift=2 の巡回行列で配置する。
        # player m の means_matrix[m] = tradeoff_means を 2m だけ右ローテーションしたもの。
        # 各プレイヤーの最良 arm: p0→0, p1→18, p2→16, p3→14, p4→12, p5→10, p6→8, p7→6
        # （全員 means[0]=0.95 が最良だが arm 番号が異なり、最適割当は一意）。
        name="hetero_large_K20_M8",
        K=20,
        M=8,
        T=1_000_000,
        means_matrix=[
            [0.95, 0.9, 0.86, 0.82, 0.78, 0.7, 0.64, 0.58, 0.52, 0.46, 0.4, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06, 0.03],  # player 0: best arm 0
            [0.86, 0.82, 0.78, 0.7, 0.64, 0.58, 0.52, 0.46, 0.4, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06, 0.03, 0.95, 0.9],   # player 1: best arm 18
            [0.78, 0.7, 0.64, 0.58, 0.52, 0.46, 0.4, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06, 0.03, 0.95, 0.9, 0.86, 0.82],   # player 2: best arm 16
            [0.64, 0.58, 0.52, 0.46, 0.4, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06, 0.03, 0.95, 0.9, 0.86, 0.82, 0.78, 0.7],   # player 3: best arm 14
            [0.52, 0.46, 0.4, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06, 0.03, 0.95, 0.9, 0.86, 0.82, 0.78, 0.7, 0.64, 0.58],   # player 4: best arm 12
            [0.4, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06, 0.03, 0.95, 0.9, 0.86, 0.82, 0.78, 0.7, 0.64, 0.58, 0.52, 0.46],   # player 5: best arm 10
            [0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06, 0.03, 0.95, 0.9, 0.86, 0.82, 0.78, 0.7, 0.64, 0.58, 0.52, 0.46, 0.4, 0.35],   # player 6: best arm 8
            [0.2, 0.16, 0.12, 0.09, 0.06, 0.03, 0.95, 0.9, 0.86, 0.82, 0.78, 0.7, 0.64, 0.58, 0.52, 0.46, 0.4, 0.35, 0.3, 0.25],   # player 7: best arm 6
        ],
        n_values=[1, 2, 3, 4, 5, 6],
        # 100 trials にすると SE が ±4,256 になり n=5 vs n=6 の差（約 5,400）が 2.5σ 水準で検出可能。
        # 20 trials（SE ±9,516）では n=4〜6 の差は統計的に有意でない。
        trials=100,
    ),
    "large_converge": HeteroExperimentConfig(
        # large と同じ means_matrix だが T=100,000 に短縮したプリセット。
        # T=100,000 では UCB ボーナス(0.016) < min gap(0.030) となり割当が収束する。
        # 収束後の挙動や、学習曲線の「収束点」を観察するのに使う。
        name="hetero_large_converge_K20_M8",
        K=20,
        M=8,
        T=100_000,
        means_matrix=[
            [0.95, 0.9, 0.86, 0.82, 0.78, 0.7, 0.64, 0.58, 0.52, 0.46, 0.4, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06, 0.03],  # player 0: best arm 0
            [0.86, 0.82, 0.78, 0.7, 0.64, 0.58, 0.52, 0.46, 0.4, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06, 0.03, 0.95, 0.9],   # player 1: best arm 18
            [0.78, 0.7, 0.64, 0.58, 0.52, 0.46, 0.4, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06, 0.03, 0.95, 0.9, 0.86, 0.82],   # player 2: best arm 16
            [0.64, 0.58, 0.52, 0.46, 0.4, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06, 0.03, 0.95, 0.9, 0.86, 0.82, 0.78, 0.7],   # player 3: best arm 14
            [0.52, 0.46, 0.4, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06, 0.03, 0.95, 0.9, 0.86, 0.82, 0.78, 0.7, 0.64, 0.58],   # player 4: best arm 12
            [0.4, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06, 0.03, 0.95, 0.9, 0.86, 0.82, 0.78, 0.7, 0.64, 0.58, 0.52, 0.46],   # player 5: best arm 10
            [0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06, 0.03, 0.95, 0.9, 0.86, 0.82, 0.78, 0.7, 0.64, 0.58, 0.52, 0.46, 0.4, 0.35],   # player 6: best arm 8
            [0.2, 0.16, 0.12, 0.09, 0.06, 0.03, 0.95, 0.9, 0.86, 0.82, 0.78, 0.7, 0.64, 0.58, 0.52, 0.46, 0.4, 0.35, 0.3, 0.25],   # player 7: best arm 6
        ],
        n_values=[1, 2, 3, 4, 5, 6],
        trials=50,
    ),
}


def build_hetero_config(args: argparse.Namespace) -> HeteroExperimentConfig:
    """CLI 引数で preset を上書きした HeteroExperimentConfig を作る。"""
    base = HETERO_CONFIGS[args.experiment]

    K = args.K or base.K
    M = args.M or base.M
    T = args.horizon or base.T

    # --K や --M を変えたのに --means-matrix がない場合は不整合になるためエラーにする
    if (args.K is not None or args.M is not None) and not args.means_matrix:
        raise ValueError("--K または --M を指定する場合は --means-matrix も指定してください。")

    if args.means_matrix:
        means_matrix = json.loads(args.means_matrix)
        if len(means_matrix) != M:
            raise ValueError(f"means_matrix の行数 ({len(means_matrix)}) が M ({M}) と一致しない。")
        for row in means_matrix:
            if len(row) != K:
                raise ValueError(f"means_matrix の各行の長さ ({len(row)}) が K ({K}) と一致しない。")
    else:
        means_matrix = base.means_matrix

    if M >= K:
        raise ValueError(f"M < K が必要。M={M}, K={K}")
    if K < M + 1:
        raise ValueError(f"BEACON は K >= M+1 を必要とする。K={K}, M={M}")

    n_values = [int(x) for x in args.n_values.split(",") if x.strip()] if args.n_values else list(base.n_values)
    max_n = K - M - 1
    n_values = [n for n in n_values if 1 <= n <= max_n]
    if not n_values:
        raise ValueError(f"有効な n がない。ParallelBEACON では 1 <= n < K-M が必要。K-M={K - M}")

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


# ---- 共通ヘルパー ----


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
