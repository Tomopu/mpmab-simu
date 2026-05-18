"""Small math helpers shared by homogeneous algorithms."""

from __future__ import annotations

import math


def ceil_int(x: float) -> int:
    """Return ``math.ceil(x)`` as an int."""
    return int(math.ceil(x))


def checked_log(x: float) -> float:
    """Return natural log after validating the argument."""
    if x <= 0.0:
        raise ValueError(f"log の引数は正でなければならない。got {x}")
    return math.log(x)

