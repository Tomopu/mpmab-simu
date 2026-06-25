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
    """通信コストとして n_steps ステップを消費する。"""
    if n_steps <= 0:
        return
    actions = [dummy_arm] * M
    for _ in range(n_steps):
        runner.step(actions)


def floor_log2(n: int) -> int:
    """floor(log2(n)) を返す。n<=0 なら -1。"""
    if n <= 0:
        return -1
    return int(math.floor(math.log2(n)))
