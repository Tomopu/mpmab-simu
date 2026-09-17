"""Trial execution and tabular result builders for homogeneous experiments."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

from simulator.algorithms import HomogeneousHuang2022, HomogeneousMultiChannelIzumi2026
from simulator.algorithms.homogeneous.trinh2021 import simulate_rskl_batch
from simulator.core.runner import Runner
from simulator.envs import BernoulliMPMABEnv, shuffle_means
from simulator.experiments.configs import ExperimentConfig
from simulator.experiments.io import _run_with_retry, _sample_times, build_curve_rows
from simulator.utils import compute_metrics

# Randomized Selfish KL-UCB の algorithm 名（summary/curves の algorithm 列に入る）と、使う版の対応
#   trinh2021_rskl    : 論文の本文の定義（c = 0、指数は厳密に計算）
#   trinh2021_rskl_c3 : 著者の公開実装と同じ計算（c = 3、指数は幅 1e-3 の二分法）
RSKL_ALGORITHMS = {"trinh2021_rskl": "paper", "trinh2021_rskl_c3": "authors_code"}


def trial_means(config: ExperimentConfig, seed: int) -> Tuple[List[float], List[int]]:
    """
    trial で使う arm 期待値の並びを返す。

    Args:
        config: 実験設定。shuffle_arms=True なら並びを入れ替える。
        seed: trial の seed（retry で変わる attempt seed ではなく、trial 固有の seed を渡す）

    Returns:
        (means, order)。order は新しい arm k に元のどの arm を置いたか（入れ替えなしなら恒等順）。
    """
    if config.shuffle_arms:
        return shuffle_means(config.means, seed)
    return list(config.means), list(range(len(config.means)))


def run_huang_trial(
    config: ExperimentConfig,
    trial: int,
    seed: int,
    sample_points: int,
    means: Optional[List[float]] = None,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Huang 2022 を 1 trial 実行する。

    Args:
        means: この trial の arm 期待値の並び。None なら config.means を使う。
    """
    means = list(config.means) if means is None else means
    env = BernoulliMPMABEnv(
        means=means,
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
        means=means,
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
    means: Optional[List[float]] = None,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Izumi 2026 を 1 trial 実行する。

    Args:
        means: この trial の arm 期待値の並び。None なら config.means を使う。
    """
    means = list(config.means) if means is None else means
    env = BernoulliMPMABEnv(
        means=means,
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
        means=means,
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


def run_with_optional_retry(
    config: ExperimentConfig,
    algorithm: str,
    trial: int,
    seed: int,
    n: int,
    sample_points: int,
    retry_on_failure: bool,
    max_attempts: int,
    means: Optional[List[float]] = None,
    arm_order: Optional[List[int]] = None,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    1 trial を実行し、必要なら割当失敗時に seed を変えて再試行する。

    再試行で破棄した attempt は summary/curve には混ぜない。全 attempt が
    失敗した場合だけ、最後の失敗 attempt を retry_exhausted=1 として保存する。

    Args:
        means: この trial の arm 期待値の並び。再試行しても並びは変えない（trial 固有）。
        arm_order: summary に保存する並び（新しい arm k に元のどの arm を置いたか）
    """
    if algorithm == "huang2022":
        run_fn = lambda s: run_huang_trial(config, trial, s, sample_points, means)
    elif algorithm == "izumi2026":
        run_fn = lambda s: run_izumi_trial(config, trial, s, n, sample_points, means)
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    summary, curves = _run_with_retry(run_fn, seed, retry_on_failure, max_attempts, algorithm, n, trial)
    summary["arm_order"] = format_arm_order(arm_order, config.K)
    return summary, curves


def format_arm_order(arm_order: Optional[Sequence[int]], K: int) -> str:
    """arm の並びを CSV に保存しやすい空白区切りの文字列にする（None なら恒等順）。"""
    order = list(range(K)) if arm_order is None else list(arm_order)
    return " ".join(str(i) for i in order)


def _run_rskl_chunk(args: Tuple) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Randomized Selfish KL-UCB の trial をまとめて実行し、summary 行と curve 行を返す。

    ProcessPoolExecutor から呼べるよう、引数は 1 つのタプルで受け取る。
    """
    config, algorithm, trials, seeds, means_list, orders, sample_points, block = args
    times = list(_sample_times(config.T, sample_points))
    result = simulate_rskl_batch(
        means_per_trial=means_list,
        M=config.M,
        T=config.T,
        seeds=seeds,
        sample_times=times,
        block=block,
        variant=RSKL_ALGORITHMS[algorithm],
    )
    summaries: List[Dict[str, object]] = []
    curves: List[Dict[str, object]] = []
    for i, (trial, seed) in enumerate(zip(trials, seeds)):
        # Randomized Selfish KL-UCB には初期化フェーズも割当の確定もないので、
        # フェーズ所要時間は 0、final_assignment_success は「最終ステップで全員が相異なる top-M arm を
        # 選んでいたか」を参考値として入れる。
        metrics = {
            "cumulative_regret": float(result.cumulative_regret[i]),
            "total_reward": float(result.total_reward[i]),
            "total_steps": config.T,
            "init_duration": 0,
            "find_good_duration": 0,
            "rank_duration": 0,
            "number_players_duration": 0,
            "exploration_duration": config.T,
            "collision_count": int(result.collision_steps[i]),
            "final_assignment_success": bool(result.final_distinct_top_m[i]),
            "player_count_success": None,
            "rank_assignment_success": None,
            "assignment_duplicate": None,
        }
        row = build_summary_row(config, algorithm, 0, trial, seed, metrics)
        row["pseudo_regret"] = float(result.pseudo_regret[i])
        row["attempt"] = 0
        row["attempts_used"] = 1
        row["retry_exhausted"] = 0
        row["arm_order"] = format_arm_order(orders[i], config.K)
        summaries.append(row)
        for j, t in enumerate(times):
            curves.append(
                {
                    "algorithm": algorithm,
                    "K": config.K,
                    "M": config.M,
                    "T": config.T,
                    "delta": config.delta,
                    "n": 0,
                    "trial": trial,
                    "seed": seed,
                    "time": t,
                    "cumulative_regret": float(result.curve[i, j]),
                    "phase": "randomized_selfish_klucb",
                    "attempt": 0,
                    "attempts_used": 1,
                }
            )
    return summaries, curves


def run_rskl_trials(
    config: ExperimentConfig,
    sample_points: int,
    batch_size: int = 25,
    workers: int = 1,
    block: int = 1024,
    algorithm: str = "trinh2021_rskl",
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Randomized Selfish KL-UCB を config.trials 回実行する（trial をバッチにまとめ、必要なら並列に回す）。

    Args:
        config: 実験設定（shuffle_arms も反映する）
        algorithm: RSKL_ALGORITHMS のキー（使う版を決める）
        sample_points: regret curve のサンプル点数
        batch_size: 1 プロセスでまとめて計算する trial 数
        workers: 並列に動かすプロセス数（1 なら逐次）
        block: 乱数をまとめて生成するステップ数（結果には影響しない）

    Returns:
        (summary 行, curve 行)。trial の順に並べて返す。
    """
    # 1. trial ごとの seed と arm の並びを、Huang/Izumi と同じ規則で作る
    trials = list(range(config.trials))
    seeds = [config.seed_base + trial for trial in trials]
    orders_means = [trial_means(config, seed) for seed in seeds]

    # 2. batch_size ごとにまとめる（各 trial の乱数は trial 固有なので、まとめ方で結果は変わらない）
    chunks = []
    for start in range(0, len(trials), max(1, batch_size)):
        end = start + max(1, batch_size)
        chunks.append(
            (
                config,
                algorithm,
                trials[start:end],
                seeds[start:end],
                [m for m, _ in orders_means[start:end]],
                [o for _, o in orders_means[start:end]],
                sample_points,
                block,
            )
        )

    # 3. 逐次または並列に実行する
    if workers <= 1:
        outputs = [_run_rskl_chunk(chunk) for chunk in chunks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(_run_rskl_chunk, chunks))

    summaries: List[Dict[str, object]] = []
    curves: List[Dict[str, object]] = []
    for chunk_summaries, chunk_curves in outputs:
        summaries.extend(chunk_summaries)
        curves.extend(chunk_curves)
    return summaries, curves


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
