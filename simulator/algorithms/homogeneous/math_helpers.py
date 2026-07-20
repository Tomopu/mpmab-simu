"""homogeneous アルゴリズムで共有する小さな数学ヘルパー。"""

from __future__ import annotations

import math


def ceil_int(x: float) -> int:
    """``math.ceil(x)`` を int として返す。"""
    return int(math.ceil(x))


def checked_log(x: float) -> float:
    """引数を検証してから自然対数を返す。"""
    if x <= 0.0:
        raise ValueError(f"log の引数は正でなければならない。got {x}")
    return math.log(x)
