from __future__ import annotations

from typing import List, Tuple

from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.core.runner import Runner


class Izumi2026ParallelVirtualNumberPlayersMixin:
    """Algorithm 3: ParallelVirtualNumberPlayers の処理を提供する補助クラス。"""

    def parallel_virtual_number_players(
        self,
        runner: Runner,
        good_arms: List[int],
        s_list: List[int],
        tau: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Algorithm 3 ParallelVirtualNumberPlayers を M 人同期実行する。

        仮想時刻 v=1..2K を n 本の good arm で並列処理し、
        推定プレイヤー数 M_hat と内部 rank j を決定する。
        Huang 2022 の VirtualNumberPlayers を n 倍高速化する。

        論文の変数対応:
            論文 h=1..ceil(2K/n) (1-based) → 実装 h_0=0..ceil(2K/n)-1 (0-based)
            論文 i=1..n (1-based) → 実装 i0=0..n-1 (0-based)
            論文 v=(h-1)*n+i (1-based 1..2K) → 実装 v = h_0*n+i0+1 (1-based で計算)
            論文 s (1-based) → 実装 s_1based = s_list[m]+1 (0-based s_list から変換)
            論文 l_i (1-based 1..K, -1 は skip) → 実装 ell[m][i0] (0-based, -1 は skip)
            論文条件 v > 2s → 実装 v > 2*s_1based
            論文 l_i の hopping: l_i = ((s + (v-2s) - 1) mod K) + 1
                → 0-based: ell[m][i0] = (v - s_1based - 1) % K
            論文 k=1..K (仮想スロット) → 実装 k_0=0..K-1 (0-based)
            論文の slot 条件 ((l_i-1 + (i-1)) mod K)+1 == k
                → 0-based: (ell[m][i0] + i0) % K == k_0

        Args:
            runner: Runner
            good_arms: G（0-based arm index リスト, 長さ n）
            s_list: 各プレイヤーの external rank（0-based）。未確定は 0 にフォールバック済み。
            tau: サンプリング時間 τ

        Returns:
            (M_hat_list, j_list):
                M_hat_list: 各プレイヤーの推定プレイヤー数
                j_list: 各プレイヤーの internal rank（1-based, 論文と同じ）
        """
        runner.set_phase("parallel_virtual_number_players")
        K = self.K
        M = self.M
        n = len(good_arms)

        M_hat = [1] * M
        j = [1] * M  # 1-based（論文と同じ）

        # good arms 以外のダミー腕
        good_set = set(good_arms)
        non_good_arms = [a for a in range(K) if a not in good_set]
        dummy = non_good_arms[0] if non_good_arms else 0

        # 論文: h=1..ceil(2K/n) → 実装: h_0=0..ceil(2K/n)-1
        H = _ceil(2 * K / n)

        for h_0 in range(H):
            # 1. n 本分の仮想時刻 v と対応するスロット ell を計算する
            ell = [[-1] * n for _ in range(M)]  # ell[m][i0]: 0-based スロット (-1 は skip)

            for m in range(M):
                # s_1based: 論文の 1-based s (実装の s_list[m] は 0-based なので +1)
                s_1based = s_list[m] + 1
                for i0 in range(n):
                    # 論文の 1-based 仮想時刻 v
                    v = h_0 * n + i0 + 1
                    if v > 2 * K:
                        # v が 2K を超えたらスキップ
                        ell[m][i0] = -1
                        continue
                    if v > 2 * s_1based:
                        # hopping フェーズ: s から (v - 2s) 回分ずらしたスロット
                        # 論文: l_i = ((s + (v-2s) - 1) mod K) + 1
                        # 0-based: ell = (v - s_1based - 1) % K
                        ell[m][i0] = (v - s_1based - 1) % K
                    else:
                        # wait フェーズ: スロット s で待機
                        # 0-based: ell = s_list[m]
                        ell[m][i0] = s_list[m]

            # 2. 論文: for k=1..K (1-based) → 実装: k_0=0..K-1 (0-based)
            for k_0 in range(K):
                R_sum = [0] * M

                # 各プレイヤーのアクション決定
                # arm i0 がスロット k_0 にマップされるか確認する
                # 条件（0-based）: ell[m][i0] != -1 かつ (ell[m][i0] + i0) % K == k_0
                actions_base = []
                matched_i0_per_player = [-1] * M

                for m in range(M):
                    found = False
                    for i0 in range(n):
                        if ell[m][i0] != -1 and (ell[m][i0] + i0) % K == k_0:
                            matched_i0_per_player[m] = i0
                            found = True
                            break
                    if found:
                        actions_base.append(good_arms[matched_i0_per_player[m]])
                    else:
                        actions_base.append(dummy)

                # τ 回サンプリングする
                for _ in range(tau):
                    result = runner.step(list(actions_base))
                    for m in range(M):
                        if matched_i0_per_player[m] >= 0:
                            R_sum[m] += result.rewards[m]

                # 3. τ 回後: R == 0 なら衝突 → M_hat, j を更新する
                for m in range(M):
                    if matched_i0_per_player[m] >= 0 and R_sum[m] == 0:
                        M_hat[m] += 1
                        # 1-based 仮想時刻 v を再計算
                        i_tilde_0 = matched_i0_per_player[m]
                        v_collision = h_0 * n + i_tilde_0 + 1
                        # 論文: if v <= 2s then j += 1
                        s_1based = s_list[m] + 1
                        if v_collision <= 2 * s_1based:
                            j[m] += 1

        return M_hat, j
