from __future__ import annotations

from typing import List

from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.core.runner import Runner


class Izumi2026ParallelVirtualMusicalChairsMixin:
    """Algorithm 2: ParallelVirtualMusicalChairs."""

    def parallel_virtual_musical_chairs(self, runner: Runner, good_arms: List[int], tau: int) -> List[int]:
        """
        Algorithm 2 ParallelVirtualMusicalChairs を M 人同期実行する。

        n 本の good arm を K 個の仮想スロットに時間分割し、各プレイヤーに外部 rank s を割り当てる。
        Huang 2022 の VirtualMusicalChairs を n 本並列化し、ランク確定が最大 n 倍高速になる。

        ブロック構造（K ステップ = 1 ブロック）:
            - ブロック先頭（t0 % K == 0）でスロット l[m][i0] を決定する。
            - spreading 条件: l[m][i0] ∉ F = {(l[m][j0] + i0 - j0) % K | j0 < i0}
            これにより 1 ブロック内で同一プレイヤーが 2 本以上の good arm を同時に引かない。

        論文の変数対応:
            論文 t=1..ceil(K*tau/n) (1-based) → 実装 t0=0..total_steps-1 (0-based)
            論文 s (1-based slot 1..K) → 実装 s_list[m] (0-based slot 0..K-1, 論文の s-1)
            論文 l_i (1-based 1..K) → 実装 l[m][i0] (0-based 0..K-1)
            論文 block start: t mod K == 1 → 実装 t0 % K == 0
            論文 arm i が引かれる条件: ((t+i-2) mod K)+1 == l_i
                → 0-based: (t0 + i0) % K == l[m][i0]

        Args:
            runner: Runner
            good_arms: FindMultipleGoodArms の出力 G（0-based arm index リスト）
            tau: サンプリング時間 τ = ceil(ln(1/delta) / mu_tilde_min)

        Returns:
            s_list: 各プレイヤーの external rank（0-based）。未確定は -1。
        """
        runner.set_phase("parallel_virtual_musical_chairs")
        K = self.K
        M = self.M
        n = len(good_arms)

        s_list = [-1] * M

        # 各プレイヤーの n 本の good arm に対するスロット候補 l[m][i0] (0-based)
        l = [[-1] * n for _ in range(M)]

        # good arms 以外のダミー腕（1 本固定で使う）
        good_set = set(good_arms)
        non_good_arms = [a for a in range(K) if a not in good_set]
        dummy = non_good_arms[0] if non_good_arms else 0

        # 総ステップ数: ceil(K * tau / n)（Huang 2022 の K * tau より n 倍少ない）
        total_steps = _ceil(K * tau / n)

        for t0 in range(total_steps):
            # ---- ブロック先頭でスロット決定 ----
            # 論文: t mod K == 1 → 実装: t0 % K == 0
            if t0 % K == 0:
                for m in range(M):
                    if s_list[m] == -1:
                        # 未確定: spreading 条件を満たしながら n 本のスロットを乱択する
                        # 1. F を空で初期化
                        # 2. i0 番目のスロット l[m][i0] を F の外から選ぶ
                        # 3. F を更新: F_{i0+1} = {(l[m][j0] + i0-j0) % K | j0 <= i0}
                        #    （0-based の spreading formula: 論文 1-based の +1 を省いた形）
                        F: List[int] = []
                        forbidden_set = set()
                        for i0 in range(n):
                            # F の外にある候補スロットを列挙
                            available = [slot for slot in range(K) if slot not in forbidden_set]
                            if not available:
                                # フォールバック: spreading を緩めて全スロットから選ぶ
                                available = list(range(K))
                            l[m][i0] = self._player_rngs[m].choice(available)
                            # 次の arm i0+1 のための forbidden set を更新する
                            # 0-based: {(l[m][j0] + (i0+1-j0)) % K | j0 <= i0}
                            # これは arm j0 と arm i0+1 が同一ブロック内の同じタイムステップに
                            # 割り当てられることを防ぐための禁止リストを更新する。
                            forbidden_set = {(l[m][j0] + (i0 + 1 - j0)) % K for j0 in range(i0 + 1)}
                    else:
                        # 確定済み: 全スロットを rank s に揃える（"wait at rank s" 状態）
                        for i0 in range(n):
                            l[m][i0] = s_list[m]

            # ---- 各プレイヤーのアクションを決定 ----
            actions = []
            pulled_i0_per_player = [-1] * M

            for m in range(M):
                pulled = False
                for i0 in range(n):
                    # arm i0 が時刻 t0 に引かれる条件（0-based）:
                    # (t0 + i0) % K == l[m][i0]
                    # spreading 条件により、各 t0 で最大 1 つの i0 のみ一致する
                    if l[m][i0] >= 0 and (t0 + i0) % K == l[m][i0]:
                        actions.append(good_arms[i0])
                        pulled_i0_per_player[m] = i0
                        pulled = True
                        break  # spreading 条件より最大 1 本のみ一致する
                if not pulled:
                    # いずれの good arm の条件も満たさないステップ: ダミー腕を引く
                    actions.append(dummy)

            result = runner.step(actions)

            # ---- 報酬チェック: r > 0 かつ未確定 → rank 確定 ----
            for m in range(M):
                i0 = pulled_i0_per_player[m]
                if i0 >= 0:
                    if result.rewards[m] > 0 and s_list[m] == -1:
                        # collision なし（no-sensing: reward > 0 で判断）
                        s_list[m] = l[m][i0]
                        # 論文 Algorithm 2:
                        #   s <- ell_i
                        #   ell_j <- s for all j in {1,...,n}
                        # rank 確定後ただちに全 good arm の仮想スロットを s に揃える。
                        # これにより、同じ virtual rank s は全 channel 上で占有され、
                        # 他 player が同じ s に座ることを防ぐ。
                        for j0 in range(n):
                            l[m][j0] = s_list[m]

        return s_list
