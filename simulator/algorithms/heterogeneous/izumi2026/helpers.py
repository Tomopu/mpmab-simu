"""heterogeneous Izumi2026 実装で共有するヘルパー。"""

from __future__ import annotations

import math
from typing import Dict, List

from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner


def build_group_map(rank_list: List[int], n: int) -> Dict[int, List[int]]:
    """rank_list から group_map[g] = プレイヤー index のリストを構築する。"""
    group_map: Dict[int, List[int]] = {g: [] for g in range(1, n + 1)}
    for m, j in enumerate(rank_list):
        g = player_group(j, n)
        group_map[g].append(m)
    return group_map


def player_group(rank: int, n: int) -> int:
    """player の rank から所属グループ (1-based) を返す。"""
    if rank <= n:
        return rank
    return ((rank - 1) % n) + 1


def consume_dummy_steps(
    runner: HeterogeneousRunner,
    n_steps: int,
    dummy_arm: int,
    M: int,
) -> None:
    """通信コストとして n_steps ステップを消費する。全員同じ arm を引くため collision が発生する。"""
    if n_steps <= 0:
        return
    actions = [dummy_arm] * M
    for _ in range(n_steps):
        runner.step(actions)


def consume_comm_steps(
    runner: HeterogeneousRunner,
    n_steps: int,
    state_arms: List[int],
) -> None:
    """
    通信コストとして n_steps ステップを消費する。

    BEACON の通信中は idle な player が自分の state arm を引くため、
    全員が異なる arm を引き collision が発生しない。
    state_arms[m]: player m が引く arm（CSVMC の s_list から設定、全員で重複なし）。

    Args:
        runner: HeterogeneousRunner
        n_steps: 消費するステップ数
        state_arms: 各 player の state arm（長さ M, 重複なし推奨）
    """
    if n_steps <= 0:
        return
    for _ in range(n_steps):
        runner.step(state_arms)


def floor_log2(n: int) -> int:
    """floor(log2(n)) を返す。n<=0 なら -1。"""
    if n <= 0:
        return -1
    return int(math.floor(math.log2(n)))
