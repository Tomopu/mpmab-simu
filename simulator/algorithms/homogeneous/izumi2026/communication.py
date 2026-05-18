from __future__ import annotations

import math
from typing import Dict, List, Tuple

from simulator.algorithms.homogeneous.math_helpers import checked_log as _ln
from simulator.core.runner import Runner


class Izumi2026CommunicationMixin:
    """Simplified communication and assignment helpers."""

    def _consume_comm_steps(self, runner: Runner, n_steps: int, dummy_arm: int) -> None:
        """
        通信時間コストとして n_steps ステップを消費する（簡略実装）。

        全プレイヤーが dummy_arm を選ぶダミーアクションで runner.step() を呼ぶ。
        forced collision bit 伝送の完全再現の代わりに時間コストのみをシミュレートする。

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
        AcceptReject ヘルパー: ρ[k] と B[k] を計算し accept/reject 集合を返す。

        Huang 2022 の _compute_accept_reject と同じ式を使う（Supplemental Pseudocode に従う）:
            ρ[k] = (Σ_m μ̂[k,m]*N[k,m]) / (Σ_m N[k,m])
            B[k] = sqrt(2*ln(1/δ) / Σ_m N[k,m]) + 2^{-p/2-3}

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
            (C_accept, C_reject)
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
            cnt_acc = sum(1 for i in active_arms if (rho[k] - B[k]) >= (rho[i] + B[i]))
            if cnt_acc >= len(active_arms) - M0:
                C_accept.append(k)

            # reject: k より明確に優る腕の数が M0 以上
            cnt_rej = sum(1 for i in active_arms if (rho[i] - B[i]) >= (rho[k] + B[k]))
            if cnt_rej >= M0:
                C_reject.append(k)

        return C_accept, C_reject

    def _try_assign(
        self,
        j: int,
        good_arms: List[int],
        M_active: int,
        C_accept: List[int],
    ) -> int:
        """
        AssignAndUpdate の割当部分: プレイヤー j に arm を割り当てて返す。

        Supplemental Pseudocode の AssignAndUpdate に従う:
            C_assign = C_accept - G（good arms を通信チャンネルとして除外）
            rank が高い player（j が大きい）から C_assign の先頭を割り当てる

        論文の変数対応:
            論文 M_active - j + 1 (1-based position) → 実装 idx = M_active - j (0-based)
            割当条件: M_active - j + 1 <= |C_assign| → 0 <= idx < len(C_assign)

        Args:
            j: 1-based internal rank
            good_arms: G（通信チャンネル, 通常は割当除外対象）
            M_active: 現在の active players 数
            C_accept: accept された腕

        Returns:
            割当腕（0-based）。割当なしは -1。
        """
        # C_assign: good arms を除いた accept 腕（follower に割り当て可能な腕）
        good_set = set(good_arms)
        C_assign = [a for a in C_accept if a not in good_set]

        # 0-based index: j=M_active が idx=0（最初の腕）、j=1 が idx=M_active-1（最後の腕）
        idx = M_active - j

        # good arms を除いた C_assign からのみ割り当てる（論文に従う通常経路）
        if 0 <= idx < len(C_assign):
            return C_assign[idx]

        return -1
