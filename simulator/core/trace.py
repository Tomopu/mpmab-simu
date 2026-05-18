"""
Trace: シミュレーション履歴を保持するデータクラス。

各ステップの行動・報酬・collision・フェーズ名などを記録し、
実験後に DataFrame や dict list として取り出せるようにする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    """1 ステップの記録。"""

    time: int
    phase: str
    actions: List[int]
    rewards: List[float]
    collisions: List[bool]
    total_reward: float
    optimal_total_reward: float
    instant_regret: float
    cumulative_reward: float
    cumulative_regret: float


class Trace:
    """
    シミュレーション全体の履歴を保持するクラス。

    Runner がステップごとに append_step を呼び出し、
    終了後に to_records() や to_dataframe() で結果を取り出す。

    Args:
        num_players: プレイヤー数 M
    """

    def __init__(self, num_players: int) -> None:
        self.M = num_players
        self._records: List[StepRecord] = []
        self._cumulative_reward: float = 0.0
        self._cumulative_regret: float = 0.0

        # フェーズごとの所要ステップ数
        self._phase_durations: Dict[str, int] = {}
        self._current_phase: str = "unknown"

    def set_phase(self, phase: str) -> None:
        """現在のフェーズ名を設定する。"""
        self._current_phase = phase

    def append_step(
        self,
        time: int,
        actions: List[int],
        rewards: List[float],
        collisions: List[bool],
        total_reward: float,
        optimal_total_reward: float,
        instant_regret: float,
        phase: Optional[str] = None,
    ) -> StepRecord:
        """
        1 ステップの記録を追加する。

        Args:
            time: 時刻（1-based が多いが、呼び出し側に合わせる）
            actions: 各プレイヤーの行動
            rewards: 各プレイヤーの報酬
            collisions: 各プレイヤーの collision フラグ
            total_reward: 報酬合計
            optimal_total_reward: 最適報酬合計
            instant_regret: 即時 regret
            phase: フェーズ名（None なら現在のフェーズを使う）

        Returns:
            追加した StepRecord
        """
        if phase is not None:
            self._current_phase = phase

        self._cumulative_reward += total_reward
        self._cumulative_regret += instant_regret

        # フェーズごとの所要ステップ数を集計
        p = self._current_phase
        self._phase_durations[p] = self._phase_durations.get(p, 0) + 1

        record = StepRecord(
            time=time,
            phase=p,
            actions=list(actions),
            rewards=list(rewards),
            collisions=list(collisions),
            total_reward=total_reward,
            optimal_total_reward=optimal_total_reward,
            instant_regret=instant_regret,
            cumulative_reward=self._cumulative_reward,
            cumulative_regret=self._cumulative_regret,
        )
        self._records.append(record)
        return record

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    @property
    def cumulative_regret(self) -> float:
        return self._cumulative_regret

    @property
    def total_steps(self) -> int:
        return len(self._records)

    @property
    def phase_durations(self) -> Dict[str, int]:
        """フェーズごとの所要ステップ数。"""
        return dict(self._phase_durations)

    def to_records(self) -> List[Dict[str, Any]]:
        """
        全記録を dict のリストとして返す。

        Returns:
            各ステップの記録を dict にしたリスト
        """
        return [
            {
                "time": r.time,
                "phase": r.phase,
                "actions": r.actions,
                "rewards": r.rewards,
                "collisions": r.collisions,
                "total_reward": r.total_reward,
                "optimal_total_reward": r.optimal_total_reward,
                "instant_regret": r.instant_regret,
                "cumulative_reward": r.cumulative_reward,
                "cumulative_regret": r.cumulative_regret,
            }
            for r in self._records
        ]

    def to_dataframe(self):
        """
        全記録を pandas DataFrame として返す。

        Returns:
            pandas.DataFrame
        """
        import pandas as pd

        return pd.DataFrame(self.to_records())
