"""Backward-compatible import for Izumi 2026."""

from simulator.algorithms.homogeneous.izumi2026 import HomogeneousMultiChannelIzumi2026
from simulator.algorithms.homogeneous.states import PlayerStateIzumi

__all__ = ["HomogeneousMultiChannelIzumi2026", "PlayerStateIzumi"]

