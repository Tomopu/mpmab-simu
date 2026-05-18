"""
アルゴリズム基底クラス。

全アルゴリズムは run() を実装し、Runner を通して環境と対話する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from simulator.core.runner import Runner


class BaseAlgorithm(ABC):
    """
    MPMAB アルゴリズムの基底クラス。

    サブクラスは run() を実装し、Runner.step() を通して環境に行動を送る。
    """

    @abstractmethod
    def run(self, runner: Runner, **kwargs: Any) -> Dict[str, Any]:
        """
        アルゴリズムを実行する。

        Args:
            runner: Runner インスタンス（環境への window）
            **kwargs: アルゴリズム固有のパラメータ

        Returns:
            結果サマリー（assigned_arms など）
        """
        ...
