"""
BEACON の Matching Oracle。

UCB 行列 mu_bar[k][m] を受け取り、最大重み二部マッチング（player-arm 割当）を返す。
"""

from __future__ import annotations

from typing import List


def matching_oracle(mu_bar: List[List[float]], M: int) -> List[int]:
    """
    最大重み二部マッチングで各プレイヤーへの arm 割当を決定する。

    BEACON Algorithm 1 の Oracle(mu_bar) に相当する。
    mu_bar[k][m] = player m の arm k の UCB 値。

    scipy が利用可能なら Hungarian 法を使う。
    利用不可の場合は全通り探索（M<=8 程度まで有効）。

    Args:
        mu_bar: UCB 行列。mu_bar[k][m] = player m が arm k に割り当てられたときの UCB 値。
            shape (K, M)。
        M: プレイヤー数

    Returns:
        assignment: assignment[m] = player m に割り当てられた arm（0-based）
            長さ M のリスト
    """
    K = len(mu_bar)

    # player-arm コスト行列を作る: cost[m][k] = mu_bar[k][m]
    cost = [[mu_bar[k][m] for k in range(K)] for m in range(M)]

    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np

        cost_np = np.array(cost)
        # linear_sum_assignment は最小化を解くので符号反転
        row_ind, col_ind = linear_sum_assignment(-cost_np)
        assignment = [-1] * M
        for i, m in enumerate(row_ind):
            assignment[m] = int(col_ind[i])
        return assignment

    except ImportError:
        pass

    # scipy がない場合のフォールバック
    from itertools import permutations

    best_val = -1.0
    best_assign = list(range(M))
    for perm in permutations(range(K), M):
        val = sum(cost[m][perm[m]] for m in range(M))
        if val > best_val:
            best_val = val
            best_assign = list(perm)

    return best_assign
