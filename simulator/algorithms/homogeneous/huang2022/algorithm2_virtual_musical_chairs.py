from __future__ import annotations

from typing import List

from simulator.core.runner import Runner


class Huang2022VirtualMusicalChairsMixin:
    """Algorithm 2: VirtualMusicalChairs."""

    def virtual_musical_chairs(self, runner: Runner, k_tilde: int, tau: int) -> List[int]:
        """
        Algorithm 2 VirtualMusicalChairs を M 人同期実行する。

        good arm k_tilde を K 個の仮想スロットに時間分割し、
        各プレイヤーに外部 rank s を割り当てる。

        論文の変数対応:
            論文 t=1..K*tau (1-based) → 実装 t=1..K*tau (1-based のまま)
            論文 s (1..K の 1-based) → 実装 s_list (0-based: 論文の s-1 に相当)
            論文の mod 条件 t mod K == ell (1-based) → 実装 (t-1) % K == ell (0-based)

        Args:
            runner: Runner
            k_tilde: good arm (0-based)
            tau: サンプリング時間 τ（tau_rank = K * ln(1/delta) / mu_tilde）

        Returns:
            s_list: 各プレイヤーの external rank（0-based）
                    未確定のプレイヤーは -1 のまま残ることがある
        """
        runner.set_phase("virtual_musical_chairs")
        K = self.K
        M = self.M

        s_list = [-1] * M
        current_slot = [0] * M

        # k_tilde 以外の任意のダミー腕（固定）
        dummy = 0 if k_tilde != 0 else 1

        total_steps = K * tau

        for t in range(1, total_steps + 1):
            # ブロック先頭（(t-1) % K == 0）でスロット決定
            if (t - 1) % K == 0:
                for m in range(M):
                    if s_list[m] == -1:
                        # 未確定: 乱択スロット（0-based）
                        current_slot[m] = self._player_rngs[m].randrange(K)
                    else:
                        # 確定済み: rank を維持
                        current_slot[m] = s_list[m]

            actions = []
            for m in range(M):
                # 自分のスロットのタイムステップのときだけ k_tilde を選ぶ
                # 論文: t mod K == ell (1-based) → 実装: (t-1) % K == ell (0-based)
                if (t - 1) % K == current_slot[m]:
                    actions.append(k_tilde)
                else:
                    actions.append(dummy)

            result = runner.step(actions)

            # 報酬チェック: k_tilde を選んで r > 0 → collision なし → rank 確定
            for m in range(M):
                if (t - 1) % K == current_slot[m]:
                    if result.rewards[m] > 0 and s_list[m] == -1:
                        s_list[m] = current_slot[m]

        return s_list
