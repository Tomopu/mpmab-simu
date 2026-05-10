"""
Runner: アルゴリズムと環境を繋ぐ同期シミュレーション実行クラス。

アルゴリズム側は Runner.step() を呼ぶことで環境に action を渡し、
StepResult を受け取りつつ Trace に記録を積む。

horizon 到達時には HorizonReached 例外を送出して途中フェーズでも停止できる。
"""

from __future__ import annotations

from typing import List, Optional

from envs.bernoulli_mpmab import BernoulliMPMABEnv, StepResult
from core.trace import Trace


class HorizonReached(Exception):
    """horizon に達したことを通知する例外。"""
    pass


class Runner:
    """
    環境との同期インターフェースと Trace への記録を担うクラス。

    アルゴリズムは Runner.step() を通して環境と対話する。
    horizon に達すると HorizonReached が送出され、アルゴリズムは途中停止できる。

    Args:
        env: BernoulliMPMABEnv のインスタンス
        horizon: 最大ステップ数 T
        trace: Trace のインスタンス（None なら内部で作成）
    """

    def __init__(
        self,
        env: BernoulliMPMABEnv,
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

    def step(self, actions: List[int], phase: Optional[str] = None) -> StepResult:
        """
        1 ステップ実行する。

        horizon に達していた場合は HorizonReached を送出する（アルゴリズムが
        途中フェーズを即座に停止できるよう、step 呼び出しの最初に検査する）。

        Args:
            actions: 各プレイヤーの arm index（0-based, 長さ M）
            phase: フェーズ名（None なら現在のフェーズを引き継ぐ）

        Returns:
            StepResult

        Raises:
            HorizonReached: horizon に達した場合
        """
        if self._elapsed >= self.horizon:
            raise HorizonReached(f"horizon={self.horizon} に達した（elapsed={self._elapsed}）")

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
