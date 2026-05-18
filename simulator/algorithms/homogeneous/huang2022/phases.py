from __future__ import annotations

from typing import Dict, List, Tuple

from simulator.algorithms.homogeneous.math_helpers import checked_log as _ln
from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.core.runner import Runner

class Huang2022PhaseMixin:
    """Initialization phases for HomogeneousHuang2022."""

    def find_good_arm(self, runner: Runner) -> Tuple[int, float]:
        """
        Algorithm 1 FindGoodArm を M 人同期実行する。

        全プレイヤーが独立に同じアルゴリズムを実行し、同じ good arm k_tilde と
        報酬下界 mu_tilde を出力するまで phase を繰り返す。

        論文の変数対応:
          論文 p (1-based) → 実装 p (1-based のまま)
          論文 k=1..K (1-based) → 実装 ell=0..K-1 (0-based)
          閾値 2^{1-p} → 実装 thr = 2.0 ** (1 - p)

        Args:
            runner: Runner インスタンス

        Returns:
            (k_tilde, mu_tilde): good arm の 0-based index と報酬下界

        Raises:
            HorizonReached: horizon に達した場合
        """
        runner.set_phase("find_good_arm")
        K = self.K
        M = self.M
        delta = self.delta

        # 全プレイヤーの出力（-1 は未決定）
        k_tilde_list = [-1] * M
        mu_tilde_list = [0.0] * M

        p = 0  # フェーズカウンタ

        while any(kt == -1 for kt in k_tilde_list):
            p += 1

            # フェーズごとに各プレイヤーの R[k], N[k] を初期化
            R = [[0] * K for _ in range(M)]
            N = [[0] * K for _ in range(M)]

            # ---- sub-phase 1: 各腕を一様ランダムに探索 ----
            # 論文: for t = 1,..., 6K^2 * 2^p * ceil(ln(2/delta))
            # （TeX の表記では 6K 2^p ln(2/delta) だが、腕あたりの期待サンプル数を
            #   十分にするため K^2 が正しい。参照実装も K^2 を使用。）
            T1 = 6 * K * K * (2**p) * _ceil(_ln(2.0 / delta))
            for _ in range(T1):
                actions = [self._player_rngs[m].randrange(K) for m in range(M)]
                result = runner.step(actions)
                # no-sensing: rewards のみ記録。collision フラグは使わない。
                for m in range(M):
                    a = actions[m]
                    R[m][a] += result.rewards[m]
                    N[m][a] += 1

            # ---- sub-phase 2: 各腕 ell を確認フェーズで accept/reject ----
            # 論文: for ell = 1,...,K (1-based) → 実装: ell=0..K-1 (0-based)
            T2 = (2**p) * K * _ceil(_ln(2.0 / delta))
            for ell in range(K):
                Rprime = [[0] * K for _ in range(M)]

                # 閾値: 2^{1-p}
                thr = 2.0 ** (1 - p)

                # accept 判定: R[ell] / N[ell] >= 2^{1-p}
                accept = [
                    (N[m][ell] > 0 and R[m][ell] / N[m][ell] >= thr)
                    for m in range(M)
                ]

                # accept なら一様ランダム、reject なら ell のみ選ぶ
                for _ in range(T2):
                    actions = []
                    for m in range(M):
                        if accept[m]:
                            actions.append(self._player_rngs[m].randrange(K))
                        else:
                            actions.append(ell)
                    result = runner.step(actions)
                    for m in range(M):
                        a = actions[m]
                        Rprime[m][a] += result.rewards[m]

                # R'[ell] >= 1 なら ell を confirm
                for m in range(M):
                    if k_tilde_list[m] == -1 and Rprime[m][ell] >= 1:
                        k_tilde_list[m] = ell
                        mu_tilde_list[m] = 2.0 ** (-p)

                if all(kt != -1 for kt in k_tilde_list):
                    break

        # 同期設計上、全員が同じ k_tilde を得る（Lemma 1 の保証）
        # 代表として player 0 の値を返す
        return k_tilde_list[0], mu_tilde_list[0]

    # ------------------------------------------------------------------
    # Algorithm 2: VirtualMusicalChairs
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Algorithm 3: VirtualNumberPlayers
    # ------------------------------------------------------------------

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

                # τ 回終了後: R==0 なら collision → M_hat +1、n<=2s なら j +1
                for m in range(M):
                    if ell[m] == k_virtual:
                        if R_sum[m] == 0:
                            M_hat[m] += 1
                            s_1based = s_list[m] + 1
                            if n <= 2 * s_1based:
                                j[m] += 1

        return M_hat, j

    # ------------------------------------------------------------------
    # Algorithm 4: DistributedExploration
    # ------------------------------------------------------------------

