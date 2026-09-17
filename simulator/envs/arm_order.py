"""
arm の並び（index と期待値の対応）を trial ごとにランダムに入れ替える補助関数。

Huang 2022 と Izumi 2026 の Good Arm 探索は arm index の小さい順に arm を確認するため、
期待値の降順に index を並べた環境では Good Arm が上位 n 本になりやすい。
並びをランダムにした環境でも性能を確かめられるよう、期待値の列を並べ替える。
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

# trial の seed から並べ替え用の乱数系列を作るときに足す定数。
# アルゴリズムや環境の報酬乱数（seed そのもの）とは別の系列にするために使う。
ARM_ORDER_SEED_OFFSET = 7_000_003


def shuffle_means(means: Sequence[float], seed: int) -> Tuple[List[float], List[int]]:
    """
    期待値の列を seed に応じて並べ替える。

    Args:
        means: 元の arm 期待値（0-based index）
        seed: trial の seed。同じ trial の全アルゴリズムで同じ並びになるよう、trial の seed を渡す。

    Returns:
        (shuffled_means, order)
        - shuffled_means[k] = means[order[k]]
        - order: 新しい arm k に元のどの arm を置いたかを表す並び
    """
    order = list(range(len(means)))
    random.Random(ARM_ORDER_SEED_OFFSET + seed).shuffle(order)
    return [float(means[i]) for i in order], order
