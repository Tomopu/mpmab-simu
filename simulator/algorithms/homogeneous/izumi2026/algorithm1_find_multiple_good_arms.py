from __future__ import annotations

from typing import Dict, List, Tuple

from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.algorithms.homogeneous.math_helpers import checked_log as _ln
from simulator.core.runner import Runner


class Izumi2026FindMultipleGoodArmsMixin:
    """Algorithm 1: FindMultipleGoodArms."""

    def find_multiple_good_arms(self, runner: Runner) -> Tuple[List[int], Dict[int, float]]:
        """
        Algorithm 1 FindMultipleGoodArms を M 人同期実行する。

        各プレイヤーが独立に n 個の good arm を探し、player 0 の結果を代表として返す。
        ループは全プレイヤーが n 個確定するか active set が 1 本以下になるまで続ける。

        論文の変数対応:
            論文 K (active set) → 実装 active_arms (0-based)
            論文 G, tilde_mu → 実装 G_list[m], mu_tilde_map[m]
            論文 T1 = 6|K|*2^p*ceil(ln(2n/delta)) → 実装 T1 = 6*Ka*2^p*ceil(ln(2n/delta))
            論文 T2 = |K|*2^p*ceil(ln(2n/delta))  → 実装 T2 = Ka*2^p*ceil(ln(2n/delta))
            論文 ell=1..K (1-based) → 実装 ell=0..Ka-1 (0-based active_arms の index)
            論文の p=1 における閾値 2^{1-p} → 実装 thr = 2^{1-p}

        Args:
            runner: Runner インスタンス

        Returns:
            (good_arms, mu_tilde_map):
                good_arms: player 0 が確定した good arm の 0-based index リスト（長さ n）
                mu_tilde_map: 各 good arm の報酬下界 dict {arm_idx: mu_tilde}

        Raises:
            HorizonReached: horizon に達した場合
        """
        runner.set_phase("find_multiple_good_arms")
        K = self.K
        M = self.M
        n = self.n
        delta = self.delta

        # 各プレイヤーが独立に追跡する good arm set と下界
        G_list: List[List[int]] = [[] for _ in range(M)]
        mu_tilde_map: List[Dict[int, float]] = [{} for _ in range(M)]

        # 共通のアクティブ腕集合（player 0 の G を基準に更新する）
        active_arms: List[int] = list(range(K))

        p = 0

        # 全プレイヤーが n 本確定するか active_arms が 1 本以下になるまで繰り返す
        while any(len(G_list[m]) < n for m in range(M)) and len(active_arms) > 1:
            p += 1
            Ka = len(active_arms)

            # フェーズ先頭で各プレイヤーの R[k], N[k] を初期化
            # active_arms 内のインデックスで管理する
            R = [[0] * Ka for _ in range(M)]
            N = [[0] * Ka for _ in range(M)]
            # active_arms の値 -> active_arms 内の位置（0-based）の逆引き
            arm_to_idx = {a: i for i, a in enumerate(active_arms)}

            # ---- sub-phase 1: 一様ランダム探索 ----
            # 論文: T1 = 6|K|*2^p*ceil(ln(2n/delta))
            T1 = 6 * Ka * (2**p) * _ceil(_ln(2.0 * n / delta))
            for _ in range(T1):
                actions = []
                for m in range(M):
                    if len(G_list[m]) < n:
                        # 未確定: active_arms から一様ランダムに選ぶ
                        actions.append(self._player_rngs[m].choice(active_arms))
                    else:
                        # 確定済み: 最初の good arm をダミーとして使う
                        actions.append(G_list[m][0])
                result = runner.step(actions)
                # no-sensing: rewards のみ記録。collision フラグは使わない。
                for m in range(M):
                    if len(G_list[m]) < n:
                        a = actions[m]
                        if a in arm_to_idx:
                            idx = arm_to_idx[a]
                            R[m][idx] += result.rewards[m]
                            N[m][idx] += 1

            # ---- sub-phase 2: 各腕を昇順で確認フェーズ ----
            # 論文: for ell in K ascending order → 実装: ell_idx=0..Ka-1 (active_arms 内の位置)
            T2 = Ka * (2**p) * _ceil(_ln(2.0 * n / delta))
            thr = 2.0 ** (1 - p)  # accept 閾値 2^{1-p}

            for ell_idx, ell in enumerate(active_arms):
                # 確認フェーズ用の報酬配列
                Rprime = [[0] * Ka for _ in range(M)]

                # accept 判定: R[ell] / N[ell] >= 2^{1-p}
                accept = [(N[m][ell_idx] > 0 and R[m][ell_idx] / N[m][ell_idx] >= thr) for m in range(M)]

                for _ in range(T2):
                    actions = []
                    for m in range(M):
                        if len(G_list[m]) < n:
                            if accept[m]:
                                # accept: active_arms から一様ランダムに選ぶ
                                a = self._player_rngs[m].choice(active_arms)
                            else:
                                # reject: arm ell を単体で選ぶ
                                a = ell
                            actions.append(a)
                        else:
                            # 確定済み: 最初の good arm をダミーとして使う
                            actions.append(G_list[m][0])
                    result = runner.step(actions)
                    for m in range(M):
                        if len(G_list[m]) < n:
                            a = actions[m]
                            if a in arm_to_idx:
                                idx = arm_to_idx[a]
                                Rprime[m][idx] += result.rewards[m]

                # accept された arm のみ G に追加する（TeX: check は accept branch 内にしかない）
                for m in range(M):
                    if len(G_list[m]) < n and accept[m] and Rprime[m][ell_idx] >= 1:
                        G_list[m].append(ell)
                        mu_tilde_map[m][ell] = 2.0 ** (-p)

                # 全プレイヤーが n 本確定したら内側の for を抜ける
                if all(len(G_list[m]) >= n for m in range(M)):
                    break

            # フェーズ終了: player 0 の G を基準に active_arms を更新する
            # 同期設計上、全プレイヤーが同じ good arms を確定するはずであり、
            # player 0 の G をシミュレーションの canonical G として使う。
            confirmed_p0 = set(G_list[0])
            active_arms = [a for a in active_arms if a not in confirmed_p0]

        return G_list[0], mu_tilde_map[0]
