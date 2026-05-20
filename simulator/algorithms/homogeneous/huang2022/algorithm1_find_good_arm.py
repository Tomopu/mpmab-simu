from __future__ import annotations

from typing import Tuple

from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.algorithms.homogeneous.math_helpers import checked_log as _ln
from simulator.core.runner import Runner


class Huang2022FindGoodArmMixin:
    """Algorithm 1: FindGoodArm."""

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
            # 論文: for t = 1,..., 6K * 2^p * ceil(ln(2/delta))
            T1 = 6 * K * (2**p) * _ceil(_ln(2.0 / delta))
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
                accept = [(N[m][ell] > 0 and R[m][ell] / N[m][ell] >= thr) for m in range(M)]

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

                # accept された arm のみ confirm する（TeX: check は accept branch 内にしかない）
                for m in range(M):
                    if k_tilde_list[m] == -1 and accept[m] and Rprime[m][ell] >= 1:
                        k_tilde_list[m] = ell
                        mu_tilde_list[m] = 2.0 ** (-p)

                if all(kt != -1 for kt in k_tilde_list):
                    break

        # 同期設計上、全員が同じ k_tilde を得る（Lemma 1 の保証）
        # 代表として player 0 の値を返す
        return k_tilde_list[0], mu_tilde_list[0]
