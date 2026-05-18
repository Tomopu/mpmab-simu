"""
metrics.py: MPMAB シミュレーション結果の評価指標。

regret は observed reward ベースで計算する（期待値ベースの regret は別途 expected_regret を使う）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from simulator.core.trace import Trace


def compute_metrics(
    trace: Trace,
    player_states: Optional[List[Any]] = None,
    means: Optional[List[float]] = None,
    M: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Trace と player_states から評価指標を計算する。

    Args:
        trace: シミュレーション全体の Trace
        player_states: 各プレイヤーの PlayerState（初期化結果）。None なら初期化指標をスキップ。
        means: 各 arm の真の平均報酬。None なら期待値ベース regret を省略。
        M: プレイヤー数。None なら player_states から推定。

    Returns:
        metrics: 評価指標の dict
          - cumulative_regret: 累積 regret（observed reward ベース）
          - total_reward: 累積報酬合計
          - total_steps: 総ステップ数
          - collision_count: collision が発生したステップ数（少なくとも 1 人が collision）
          - phase_durations: フェーズごとの所要ステップ数
          - init_duration: 初期化フェーズ合計（find_good_arm + VMC + VNP）
          - find_good_duration: FindGoodArm の所要ステップ数
          - rank_duration: VirtualMusicalChairs の所要ステップ数
          - number_players_duration: VirtualNumberPlayers の所要ステップ数
          - exploration_duration: DistributedExploration の所要ステップ数
          - rank_assignment_success: 全プレイヤーに重複なく rank が割り当てられたか（bool）
          - player_count_success: 全プレイヤーの M_hat が真の M と一致するか（bool）
          - final_assignment_success: 全プレイヤーの assigned_arm が top-M arm に含まれるか（bool）
          - assignment_duplicate: 割当腕に重複があるか（bool）
    """
    phase_durations = trace.phase_durations

    # 累積 regret と総報酬（observed reward ベース）
    cumulative_regret = trace.cumulative_regret
    total_reward = trace.cumulative_reward
    total_steps = trace.total_steps

    # collision カウント: 少なくとも 1 人が collision したステップ数
    records = trace.to_records()
    collision_count = sum(
        1 for r in records if any(r["collisions"])
    )

    # フェーズ別所要時間の集計
    # FindGoodArm / FindMultipleGoodArms フェーズ（Huang 2022 / Izumi 2026 共通）
    find_good_duration = sum(
        v for k, v in phase_durations.items()
        if k.startswith("find_good_arm") or k.startswith("find_multiple_good_arms")
    )
    # VirtualMusicalChairs / ParallelVirtualMusicalChairs フェーズ
    rank_duration = sum(
        v for k, v in phase_durations.items()
        if k.startswith("virtual_musical_chairs")
        or k.startswith("parallel_virtual_musical_chairs")
    )
    # VirtualNumberPlayers / ParallelVirtualNumberPlayers フェーズ
    number_players_duration = sum(
        v for k, v in phase_durations.items()
        if k.startswith("virtual_number_players")
        or k.startswith("parallel_virtual_number_players")
    )
    # DistributedExploration フェーズ（phase サブフェーズも含む）
    exploration_duration = sum(
        v for k, v in phase_durations.items()
        if k.startswith("distributed_exploration")
    )
    # HierarchicalDistributedExploration フェーズ（Izumi 2026 用, hde_phase* も含む）
    exploration_duration += sum(
        v for k, v in phase_durations.items()
        if k.startswith("hierarchical_distributed_exploration")
        or k.startswith("hde_phase")
    )

    # 初期化合計
    init_duration = find_good_duration + rank_duration + number_players_duration

    metrics: Dict[str, Any] = {
        "cumulative_regret": cumulative_regret,
        "total_reward": total_reward,
        "total_steps": total_steps,
        "collision_count": collision_count,
        "phase_durations": phase_durations,
        "init_duration": init_duration,
        "find_good_duration": find_good_duration,
        "rank_duration": rank_duration,
        "number_players_duration": number_players_duration,
        "exploration_duration": exploration_duration,
        "rank_assignment_success": None,
        "player_count_success": None,
        "final_assignment_success": None,
        "assignment_duplicate": None,
    }

    # player_states がある場合はさらに詳細な指標を計算
    if player_states is not None:
        num_players = M if M is not None else len(player_states)

        # rank assignment success: 全プレイヤーの external_rank_s が重複なし
        s_values = [ps.external_rank_s for ps in player_states]
        rank_assignment_success = (
            len(set(s_values)) == num_players and all(s >= 0 for s in s_values)
        )
        metrics["rank_assignment_success"] = rank_assignment_success

        # player count success: 全プレイヤーの M_hat が真の M と一致
        if hasattr(player_states[0], "M_hat"):
            m_hats = [ps.M_hat for ps in player_states]
            player_count_success = all(mh == num_players for mh in m_hats)
            metrics["player_count_success"] = player_count_success

        # final assignment: assigned_arm の重複確認
        assigned = [ps.assigned_arm for ps in player_states if ps.assigned_arm >= 0]
        assignment_duplicate = len(assigned) != len(set(assigned))
        metrics["assignment_duplicate"] = assignment_duplicate

        # top-M arm に全員が割り当てられているか確認
        if means is not None:
            top_m_arms = set(
                sorted(range(len(means)), key=lambda k: means[k], reverse=True)[:num_players]
            )
            final_assignment_success = (
                len(assigned) == num_players and all(a in top_m_arms for a in assigned)
            )
            metrics["final_assignment_success"] = final_assignment_success

    return metrics
