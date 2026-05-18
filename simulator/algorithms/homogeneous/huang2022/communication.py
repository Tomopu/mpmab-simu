from __future__ import annotations

import math
from typing import Dict, List, Tuple

from simulator.algorithms.homogeneous.math_helpers import checked_log as _ln
from simulator.core.runner import Runner

class Huang2022CommunicationMixin:
    """Simplified communication and assignment helpers."""

    def _consume_comm_steps(self, runner: Runner, n_steps: int, dummy_arm: int) -> None:
        """
        通信時間コストとして n_steps ステップを消費する（簡略実装）。

        全プレイヤーが dummy_arm を選ぶダミーアクションで runner.step() を呼ぶ。
        通信コストを Trace に記録するための簡略化実装。
        実際の forced collision bit 伝送は行わない。

        Args:
            runner: Runner
            n_steps: 消費するステップ数
            dummy_arm: 全プレイヤーが選ぶダミー腕（0-based）
        """
        actions = [dummy_arm] * self.M
        for _ in range(n_steps):
            runner.step(actions)

    def _compute_accept_reject(
        self,
        mu_hat: List[List[float]],
        N_mat: List[List[int]],
        active_arms: List[int],
        M0: int,
        delta: float,
        p: int,
    ) -> Tuple[List[int], List[int]]:
        """
        ρ[k] と B[k] を計算し accept/reject 集合を返す。

        論文 Algorithm 6 の集約推定と accept/reject 判定:
          ρ[k] = (Σ_i µ̂[k,i]*N[k,i]) / (Σ_i N[k,i])
          B[k] = sqrt(2*ln(1/δ) / Σ_i N[k,i]) + 2^{-p/2-3}

        accept 条件: |{i in K : ρ[k]-B[k] >= ρ[i]+B[i]}| >= |K| - M0
        reject 条件: |{i in K : ρ[i]-B[i] >= ρ[k]+B[k]}| >= M0

        Args:
            mu_hat: mu_hat[k][pid] = player pid による arm k の推定平均報酬
            N_mat: N_mat[k][pid] = player pid による arm k のサンプル数
            active_arms: 現在の active arms（0-based）
            M0: 現在の active players 数
            delta: 信頼度
            p: フェーズ番号

        Returns:
            (C_accept, C_reject): accept された腕のリスト、reject された腕のリスト
        """
        rho: Dict[int, float] = {}
        B: Dict[int, float] = {}

        for k in active_arms:
            num = sum(mu_hat[k][pid] * N_mat[k][pid] for pid in range(self.M))
            den = sum(N_mat[k][pid] for pid in range(self.M))
            rho[k] = (num / den) if den > 0 else 0.0
            B[k] = math.sqrt(2.0 * _ln(1.0 / delta) / max(1, den)) + 2.0 ** (-p / 2.0 - 3)

        C_accept = []
        C_reject = []

        for k in active_arms:
            # accept: k より明確に劣る腕の数が |K| - M0 以上
            cnt_acc = sum(
                1 for i in active_arms if (rho[k] - B[k]) >= (rho[i] + B[i])
            )
            if cnt_acc >= len(active_arms) - M0:
                C_accept.append(k)

            # reject: k より明確に優る腕の数が M0 以上
            cnt_rej = sum(
                1 for i in active_arms if (rho[i] - B[i]) >= (rho[k] + B[k])
            )
            if cnt_rej >= M0:
                C_reject.append(k)

        return C_accept, C_reject

    def _fallback_assign_remaining_accepts(
        self,
        f: List[int],
        j_list: List[int],
        M0: int,
        C_accept: List[int],
        assigned_before: set[int],
    ) -> None:
        """
        未使用の accepted arm を未割当 player に補完割当する旧実験用 fallback。

        現在の論文再現経路からは呼び出さない。簡略通信実装で未割当が残る
        小規模実験を継続させるために使っていた処理を、将来の比較用に隔離して残す。
        """
        newly_used = set(a for a in f if a >= 0) - assigned_before
        remaining_accept = [
            a for a in C_accept if a not in assigned_before and a not in newly_used
        ]
        remaining_players = [
            m for m in range(self.M) if f[m] == -1 and 1 <= j_list[m] <= M0
        ]
        remaining_players.sort(key=lambda m: (j_list[m], m), reverse=True)
        for m, arm in zip(remaining_players, remaining_accept):
            f[m] = arm
            newly_used.add(arm)


# ------------------------------------------------------------------
# モジュールレベルのヘルパー
# ------------------------------------------------------------------

