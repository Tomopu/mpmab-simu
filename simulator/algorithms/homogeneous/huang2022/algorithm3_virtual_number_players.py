from __future__ import annotations

from typing import List, Tuple

from simulator.core.runner import Runner


class Huang2022VirtualNumberPlayersMixin:
    """Algorithm 3: VirtualNumberPlayers の処理を提供する補助クラス。"""

    def virtual_number_players(
        self, runner: Runner, k_tilde: int, s_list: List[int], tau: int
    ) -> Tuple[List[int], List[int]]:
        """
        Algorithm 3 VirtualNumberPlayers を M 人同期実行する。

        外部 rank s から内部 rank j と推定プレイヤー数 M_hat を決定する。

        論文の変数対応:
            論文 n=1..2K (1-based) → 実装 n=1..2K (1-based のまま)
            論文 k=1..K (1-based の仮想スロット) → 実装 k_virtual=0..K-1 (0-based)
            論文 s (1-based) → 実装: s_list[m] は 0-based なので比較時に +1 して 1-based に戻す
            論文条件 n > 2s → 実装 n > 2*(s_list[m]+1)
            論文条件 n <= 2s → 実装 n <= 2*(s_list[m]+1)

        Args:
            runner: Runner
            k_tilde: good arm (0-based)
            s_list: 各プレイヤーの external rank（0-based）
            tau: サンプリング時間 τ（tau_comm = ln(1/delta) / mu_tilde）

        Returns:
            (M_hat_list, j_list):
                M_hat_list: 各プレイヤーの推定プレイヤー数
                j_list: 各プレイヤーの internal rank（1-based, 論文と同じ）
        """
        runner.set_phase("virtual_number_players")
        K = self.K
        M = self.M

        M_hat = [1] * M
        ell = list(s_list)  # 各プレイヤーの現在の仮想スロット（0-based）
        j = [1] * M  # 1-based（論文と同じ）

        dummy = 0 if k_tilde != 0 else 1

        # 論文: n=1..2K, k=1..K, t=1..tau の 3 重ループ
        for n in range(1, 2 * K + 1):
            # 論文: if n > 2s then ell <- (ell+1) mod K
            # s_list は 0-based なので論文の s = s_list[m]+1 として比較
            for m in range(M):
                s_1based = s_list[m] + 1  # 論文の 1-based s
                if n > 2 * s_1based:
                    ell[m] = (ell[m] + 1) % K

            # 論文: for k=1..K (1-based) → 実装: k_virtual=0..K-1 (0-based)
            for k_virtual in range(K):
                R_sum = [0] * M

                for _ in range(tau):
                    actions = []
                    for m in range(M):
                        if ell[m] == k_virtual:
                            actions.append(k_tilde)
                        else:
                            actions.append(dummy)
                    result = runner.step(actions)
                    for m in range(M):
                        if ell[m] == k_virtual:
                            R_sum[m] += result.rewards[m]

                # τ 回終了後: R==0 なら衝突 → M_hat +1、n<=2s なら j +1
                for m in range(M):
                    if ell[m] == k_virtual:
                        if R_sum[m] == 0:
                            M_hat[m] += 1
                            s_1based = s_list[m] + 1
                            if n <= 2 * s_1based:
                                j[m] += 1

        return M_hat, j
