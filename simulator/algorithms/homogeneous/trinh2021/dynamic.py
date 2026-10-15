"""
プレイヤーが途中で入る・出る動的な設定（Trinh and Combes, 2021 の 5 節）を、複数 trial まとめて実行する。

著者の公開実装（environment.DynamicEnvironment と algorithms.selfishucb）に合わせた約束:
    - プレイヤーの席は K 個（M = K）。最初の席の 1 人だけが時刻 0 から参加している
    - 入る時刻 t_entries は昇順。先頭は 0 に置き換える（最初の 1 人）
    - 時刻 e に入るプレイヤーは、時刻 e の手番のあとに席に着き、次の手番から行動する
    - 時刻 e に出るプレイヤーは、時刻 e の手番を終えてから出る。席の統計は 0 に戻す
    - 席が埋まっているときに入ろうとしたプレイヤーは入れない（出ないモデルでは、以降だれも入らない）
    - 行動の選び方は simulate_rskl_batch と同じ policy.select_actions。時刻 t は全体の時刻を使う（著者実装と同じ）
    - 評価: 参加人数 n(t) に対する最適な期待報酬 mu_opt(t) = 上位 n(t) 本の期待値の和と、
      衝突しなかった参加者の arm の期待値の和 system(t) を比べる
        * 累積 regret = sum_t (mu_opt(t) - system(t))
        * 性能比 = sum_t system(t) / sum_t mu_opt(t)（論文の表 2 の R / R*）

時刻の対応: 著者実装の時刻 t_a は 0 始まり、このモジュールの時刻 t は 1 始まり（t = t_a + 1）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from simulator.algorithms.homogeneous.trinh2021.batch import trial_generators
from simulator.algorithms.homogeneous.trinh2021.policy import select_actions


@dataclass
class DynamicBatchResult:
    """
    動的な設定のバッチ実行の結果（配列の先頭の次元は trial）。

    Attributes:
        cumulative_regret: sum_t (mu_opt(t) - system(t))
        optimal_reward: sum_t mu_opt(t)
        system_reward: sum_t system(t)
        ratio: system_reward / optimal_reward
        collision_steps: 参加者のうち少なくとも 1 人が衝突したステップ数
        curve: sample_times の各時刻での累積 regret（形は (trial, len(sample_times))）
        n_entered: 実際に席に着いた延べ人数（最初の 1 人を含む）
    """

    cumulative_regret: np.ndarray
    optimal_reward: np.ndarray
    system_reward: np.ndarray
    ratio: np.ndarray
    collision_steps: np.ndarray
    curve: np.ndarray
    n_entered: np.ndarray


def sample_arrivals(rng: np.random.Generator, K: int, T: int, lam: float, mean_stay: Optional[float]):
    """
    著者実装と同じ方法で、入る時刻（と出る時刻）を引く。

    Args:
        rng: 乱数生成器
        K: 席の数（= arm の数）
        T: horizon
        lam: 到着率（1 ステップあたり）
        mean_stay: 滞在時間の平均（指数分布）。None なら出ないモデル

    Returns:
        (t_entries, t_leaving)。t_leaving は出ないモデルでは None
    """
    n_entries = max(K, int(rng.poisson(lam * T)))
    t_entries = np.sort(rng.integers(T, size=n_entries))
    t_entries[0] = 0
    if mean_stay is None:
        return t_entries, None
    stay = rng.exponential(mean_stay, size=n_entries).astype(int)
    t_leaving = np.minimum(t_entries + stay, T)
    t_leaving = np.where(t_leaving <= t_entries, t_entries + 1, t_leaving)
    return t_entries, t_leaving


def _schedule(t_entries: np.ndarray, t_leaving: Optional[np.ndarray]) -> Dict[int, List[tuple]]:
    """入る・出る出来事を、著者実装の時刻 t_a ごとの (種類, 入る順番) の列にまとめる。"""
    events: Dict[int, List[tuple]] = {}
    for i, e in enumerate(t_entries[1:], start=1):
        events.setdefault(int(e), []).append(("enter", i))
    if t_leaving is not None:
        for i, e in enumerate(t_leaving):
            events.setdefault(int(e), []).append(("leave", i))
    return events


def simulate_dynamic_batch(
    means: Sequence[float],
    T: int,
    seeds: Sequence[int],
    entries_per_trial: Sequence[Sequence[int]],
    leaving_per_trial: Optional[Sequence[Optional[Sequence[int]]]] = None,
    sample_times: Sequence[int] = (),
    c: float = 0.0,
    variant: str = "paper",
) -> DynamicBatchResult:
    """
    動的な設定で、R-SKL（variant="paper"）などを複数 trial まとめて horizon T まで実行する。

    Args:
        means: arm の期待値（長さ K。全 trial で共通）
        T: horizon
        seeds: trial ごとの seed（長さ B）
        entries_per_trial: trial ごとの入る時刻（著者実装の 0 始まりの時刻）
        leaving_per_trial: trial ごとの出る時刻（入る時刻と同じ順番）。None なら出ないモデル
        sample_times: 累積 regret を記録する時刻（1 始まり）
        c: 探索関数の log log t の係数
        variant: policy.select_actions の variant（"paper"・"selfish_klucb"・"selfish_ts"）

    Returns:
        DynamicBatchResult
    """
    means = np.asarray(means, dtype=float)
    K = len(means)
    B = len(seeds)
    if len(entries_per_trial) != B:
        raise ValueError("entries_per_trial の長さは trial 数と一致する必要がある。")
    sorted_means = np.sort(means)[::-1]
    prefix_opt = np.concatenate([[0.0], np.cumsum(sorted_means)])  # 参加人数 n の最適な期待報酬

    rngs = [trial_generators(int(s)) for s in seeds]
    counts = np.zeros((B, K, K))  # (trial, 席, arm)
    reward_sums = np.zeros((B, K, K))
    active = np.zeros((B, K), dtype=bool)
    active[:, 0] = True
    slot_of = [dict() for _ in range(B)]  # 入る順番 -> 席
    blocked = [set() for _ in range(B)]  # 席が埋まっていて入れなかった順番
    closed = np.zeros(B, dtype=bool)  # 出ないモデルで、席が埋まって以降の入場を止めた trial
    n_entered = np.ones(B, dtype=np.int64)
    for b in range(B):
        slot_of[b][0] = 0
    leaving = leaving_per_trial if leaving_per_trial is not None else [None] * B
    schedules = [_schedule(np.asarray(entries_per_trial[b]), None if leaving[b] is None else np.asarray(leaving[b]))
                 for b in range(B)]
    # 出来事の起きる時刻（1 始まりの手番 t の直後に処理する。t = t_a + 1）
    event_steps: Dict[int, List[int]] = {}
    for b, sch in enumerate(schedules):
        for ta in sch:
            if 0 <= ta < T:
                event_steps.setdefault(ta + 1, []).append(b)

    cumulative_regret = np.zeros(B)
    optimal_reward = np.zeros(B)
    system_reward = np.zeros(B)
    collision_steps = np.zeros(B, dtype=np.int64)
    sample_list = sorted(set(int(s) for s in sample_times if s > 0))
    curve = np.zeros((B, len(sample_list)))
    sample_pos = 0
    arms = np.arange(K)
    prev_index: Optional[np.ndarray] = None

    for t in range(1, T + 1):
        # 1. 全席の行動を選ぶ（参加していない席の行動は使わない）
        if variant == "selfish_ts":
            failures = counts - reward_sums
            noise = np.stack([pr.beta(1.0 + reward_sums[b], 1.0 + failures[b]) for b, (pr, _) in enumerate(rngs)])
        else:
            noise = np.stack([pr.standard_normal((K, K)) for pr, _ in rngs])
        actions, prev_index = select_actions(counts, reward_sums, t, noise, prev_index=prev_index, c=c, variant=variant)

        # 2. 環境: 参加者だけで衝突を判定し、報酬を生成する
        chosen = (actions[:, :, None] == arms) & active[:, :, None]
        arm_counts = chosen.sum(axis=1)
        collided = (np.take_along_axis(arm_counts, actions, axis=1) >= 2) & active
        arm_means = means[actions]
        uniform = np.stack([er.random(K) for _, er in rngs])
        rewards = (uniform < arm_means) & ~collided & active

        # 3. 参加者の統計を更新する
        upd = chosen.astype(float)
        counts += upd
        reward_sums += upd * rewards[:, :, None]

        # 4. 評価指標
        n_active = active.sum(axis=1)
        opt = prefix_opt[n_active]
        sysr = np.where(active & ~collided, arm_means, 0.0).sum(axis=1)
        optimal_reward += opt
        system_reward += sysr
        cumulative_regret += opt - sysr
        collision_steps += collided.any(axis=1)
        while sample_pos < len(sample_list) and sample_list[sample_pos] == t:
            curve[:, sample_pos] = cumulative_regret
            sample_pos += 1

        # 5. 手番 t（著者の時刻 t - 1）のあとに出る・入る
        for b in event_steps.get(t, ()):
            for kind, i in schedules[b][t - 1]:
                if kind != "leave" or i not in slot_of[b]:
                    continue
                s = slot_of[b].pop(i)
                active[b, s] = False
                counts[b, s] = 0.0
                reward_sums[b, s] = 0.0
            for kind, i in schedules[b][t - 1]:
                if kind != "enter" or closed[b]:
                    continue
                free = np.flatnonzero(~active[b])
                if len(free) == 0:
                    if leaving[b] is None:
                        closed[b] = True
                    else:
                        blocked[b].add(i)
                    continue
                s = int(free[0])
                active[b, s] = True
                counts[b, s] = 0.0
                reward_sums[b, s] = 0.0
                slot_of[b][i] = s
                n_entered[b] += 1
            # prev_index は戻さなくてよい。統計を 0 にした席の指数は次の手番で 1（未試行）になり、
            # 静的な場合の 1 手番目と同じ状態から Newton 法の初期値を使い直す（trial ごとに独立のまま）

    return DynamicBatchResult(
        cumulative_regret=cumulative_regret,
        optimal_reward=optimal_reward,
        system_reward=system_reward,
        ratio=system_reward / optimal_reward,
        collision_steps=collision_steps,
        curve=curve,
        n_entered=n_entered,
    )
