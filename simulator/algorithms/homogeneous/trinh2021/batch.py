"""
Randomized Selfish KL-UCB を複数 trial まとめて numpy で実行する（比較実験用の高速版）。

Randomized Selfish KL-UCB は割当が確定する手法ではないため、horizon T まで毎ステップ
全 (プレイヤー, arm) の指数を計算する。Runner と Trace を使う 1 ステップずつの実行では
T = 10^6 に時間がかかりすぎるので、ここでは環境も含めて trial 方向にベクトル化する。

環境のモデルは BernoulliMPMABEnv と同じ:
    - 2 人以上が同じ arm を選ぶと、その全員の報酬が 0（衝突）
    - 衝突がなければ Bernoulli(means[arm]) の報酬
    - 即時 regret = top-M arm の期待値和 - その時刻の観測報酬の合計

乱数は trial ごとに独立な系列を使う（seed から SeedSequence で行動用と環境用の 2 系列を作る）。
そのため、どの trial を同じバッチに入れても、各 trial の結果は変わらない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from simulator.algorithms.homogeneous.trinh2021.policy import select_actions


@dataclass
class RSKLBatchResult:
    """
    バッチ実行の結果（配列の先頭の次元は trial）。

    Attributes:
        cumulative_regret: 観測報酬ベースの累積 regret（BernoulliMPMABEnv と同じ定義）
        pseudo_regret: 期待値ベースの累積 regret（衝突しなかった player の arm 期待値で数える）
        total_reward: 観測報酬の合計
        collision_steps: 少なくとも 1 人が衝突したステップ数
        final_distinct_top_m: 最終ステップで全員が相異なる top-M arm を選んでいたか
        curve: sample_times の各時刻での累積 regret（形は (trial, len(sample_times))）
        actions: record_actions=True のときだけ、全ステップの行動（形は (T, trial, M)）
    """

    cumulative_regret: np.ndarray
    pseudo_regret: np.ndarray
    total_reward: np.ndarray
    collision_steps: np.ndarray
    final_distinct_top_m: np.ndarray
    curve: np.ndarray
    actions: Optional[np.ndarray] = None


def trial_generators(seed: int):
    """
    1 trial 分の乱数生成器を 2 つ作る。

    Args:
        seed: trial の seed

    Returns:
        (policy_rng, env_rng)。policy_rng は行動選択の正規乱数、env_rng は報酬の一様乱数に使う。
    """
    policy_seq, env_seq = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(policy_seq), np.random.default_rng(env_seq)


def simulate_rskl_batch(
    means_per_trial: Sequence[Sequence[float]],
    M: int,
    T: int,
    seeds: Sequence[int],
    sample_times: Sequence[int],
    c: float = 0.0,
    block: int = 1024,
    record_actions: bool = False,
    variant: str = "paper",
) -> RSKLBatchResult:
    """
    Randomized Selfish KL-UCB を複数 trial まとめて horizon T まで実行する。

    Args:
        means_per_trial: trial ごとの arm 期待値（形 (B, K)）。arm の並びは trial ごとに違ってよい。
        M: プレイヤー数
        T: horizon
        seeds: trial ごとの seed（長さ B）
        sample_times: 累積 regret を記録する時刻（0 を含んでよい）
        c: 探索関数の log log t の係数（論文の実験に合わせて既定は 0）
        block: 乱数をまとめて生成するステップ数（結果には影響しない。速さとメモリだけに効く）
        record_actions: True なら全ステップの行動を返す（テスト用。T が大きいと重い）
        variant: "paper"（論文の本文）または "authors_code"（著者の公開実装）。policy.select_actions を参照。

    Returns:
        RSKLBatchResult
    """
    means = np.asarray(means_per_trial, dtype=float)
    B, K = means.shape
    if len(seeds) != B:
        raise ValueError("seeds の長さは trial 数と一致する必要がある。")
    if not (1 <= M <= K):
        raise ValueError("1 <= M <= K が必要。")

    # top-M arm の期待値和（最適な総報酬）。trial ごとに arm の並びが違っても値は同じになる
    optimal_total = np.sort(means, axis=1)[:, ::-1][:, :M].sum(axis=1)
    top_m_mask = np.zeros((B, K), dtype=bool)
    top_m_idx = np.argsort(-means, axis=1, kind="stable")[:, :M]
    np.put_along_axis(top_m_mask, top_m_idx, True, axis=1)

    rngs = [trial_generators(int(s)) for s in seeds]

    # player m が arm k を選んだ回数と得た報酬の合計（形 (B, M, K)）
    counts = np.zeros((B, M, K))
    reward_sums = np.zeros((B, M, K))
    prev_index: Optional[np.ndarray] = None

    cumulative_regret = np.zeros(B)
    pseudo_regret = np.zeros(B)
    total_reward = np.zeros(B)
    collision_steps = np.zeros(B, dtype=np.int64)

    sample_list = sorted(set(int(s) for s in sample_times))
    curve = np.zeros((B, len(sample_list)))
    sample_pos = 0
    while sample_pos < len(sample_list) and sample_list[sample_pos] <= 0:
        sample_pos += 1  # t = 0 の累積 regret は 0 のまま

    actions_log = np.zeros((T, B, M), dtype=np.int16) if record_actions else None
    arms = np.arange(K)
    flat_rows = np.arange(B * M) * K  # (B, M, K) を 1 次元に並べたときの各 (trial, player) の先頭位置
    actions = np.zeros((B, M), dtype=np.int64)

    for block_start in range(1, T + 1, block):
        length = min(block, T - block_start + 1)
        # 1. trial ごとの乱数系列から、このブロック分の乱数をまとめて作る
        noise_block = np.stack([pr.standard_normal((length, M, K)) for pr, _ in rngs], axis=1)
        uniform_block = np.stack([er.random((length, M)) for _, er in rngs], axis=1)

        for s in range(length):
            t = block_start + s
            # 2. 各プレイヤーが自分の統計だけから行動を選ぶ（衝突フラグは使わない）
            actions, prev_index = select_actions(
                counts, reward_sums, t, noise_block[s], prev_index=prev_index, c=c, variant=variant
            )
            # 3. 環境: arm ごとの選択人数から衝突を判定し、報酬を生成する
            chosen = actions[:, :, None] == arms  # (B, M, K)
            arm_counts = chosen.sum(axis=1)  # (B, K)
            collided = np.take_along_axis(arm_counts, actions, axis=1) >= 2  # (B, M)
            arm_means = np.take_along_axis(means, actions, axis=1)  # (B, M)
            rewards = (uniform_block[s] < arm_means) & ~collided

            # 4. 各プレイヤーの統計を更新する（観測した報酬だけを使う）
            flat = flat_rows + actions.ravel()
            counts.reshape(-1)[flat] += 1.0
            reward_sums.reshape(-1)[flat] += rewards.ravel()

            # 5. 評価指標を更新する（アルゴリズムの意思決定には使わない）
            step_reward = rewards.sum(axis=1)
            total_reward += step_reward
            cumulative_regret += optimal_total - step_reward
            pseudo_regret += optimal_total - np.where(collided, 0.0, arm_means).sum(axis=1)
            collision_steps += collided.any(axis=1)
            if actions_log is not None:
                actions_log[t - 1] = actions

            while sample_pos < len(sample_list) and sample_list[sample_pos] == t:
                curve[:, sample_pos] = cumulative_regret
                sample_pos += 1

    # 6. 最終ステップで全員が相異なる top-M arm を選んでいたか（参考指標）
    distinct = np.array([len(set(row)) == M for row in actions])
    on_top = np.take_along_axis(top_m_mask, actions, axis=1).all(axis=1)
    final_distinct_top_m = distinct & on_top

    return RSKLBatchResult(
        cumulative_regret=cumulative_regret,
        pseudo_regret=pseudo_regret,
        total_reward=total_reward,
        collision_steps=collision_steps,
        final_distinct_top_m=final_distinct_top_m,
        curve=curve,
        actions=actions_log,
    )
