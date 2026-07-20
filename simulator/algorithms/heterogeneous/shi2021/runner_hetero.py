"""
HeterogeneousRunner: HeterogeneousMPMABEnv と通信するための Runner。

BernoulliMPMABEnv の Runner と同じインターフェースを持つが、
HeterogeneousMPMABEnv.step() を使い HeterogeneousStepResult を返す。
collision_sensing=True の環境で BEACON が collision フラグを参照できる。
"""

from __future__ import annotations

from typing import List, Optional

from simulator.envs.heterogeneous_mpmab import HeterogeneousMPMABEnv, HeterogeneousStepResult
from simulator.core.trace import Trace


class HorizonReachedHetero(Exception):
    """horizon に達したことを通知する例外（Heterogeneous Runner 用）。"""
    pass


class HeterogeneousRunner:
    """
    Heterogeneous 環境との同期インターフェースと Trace への記録を担うクラス。

    BEACON などの collision-sensing アルゴリズムは、戻り値の
    collisions フィールドを意思決定に使ってよい（collision_sensing=True の場合）。

    Args:
        env: HeterogeneousMPMABEnv のインスタンス
        horizon: 最大ステップ数 T
        trace: Trace のインスタンス（None なら内部で作成）
    """

    def __init__(
        self,
        env: HeterogeneousMPMABEnv,
        horizon: int,
        trace: Optional[Trace] = None,
    ) -> None:
        self.env = env
        self.horizon = horizon
        self.trace = trace if trace is not None else Trace(env.M)
        self._elapsed = 0

    @property
    def elapsed(self) -> int:
        """これまでの経過ステップ数。"""
        return self._elapsed

    @property
    def remaining(self) -> int:
        """残りステップ数。"""
        return self.horizon - self._elapsed

    def set_phase(self, phase: str) -> None:
        """Trace の現在フェーズを設定する。"""
        self.trace.set_phase(phase)

    def step(self, actions: List[int], phase: Optional[str] = None) -> HeterogeneousStepResult:
        """
        1 ステップ実行する。

        horizon に達していた場合は HorizonReachedHetero を送出する。

        Args:
            actions: 各プレイヤーの arm index（0-based, 長さ M）
            phase: フェーズ名（None なら現在のフェーズを引き継ぐ）

        Returns:
            HeterogeneousStepResult

        Raises:
            HorizonReachedHetero: horizon に達した場合
        """
        if self._elapsed >= self.horizon:
            raise HorizonReachedHetero(
                f"horizon={self.horizon} に達した（elapsed={self._elapsed}）"
            )

        result = self.env.step(actions)
        self._elapsed += 1

        self.trace.append_step(
            time=self._elapsed,
            actions=result.actions,
            rewards=result.rewards,
            collisions=result.collisions,
            total_reward=result.total_reward,
            optimal_total_reward=result.optimal_total_reward,
            instant_regret=result.instant_regret,
            phase=phase,
        )

        return result
