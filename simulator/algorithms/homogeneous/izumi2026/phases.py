from __future__ import annotations

from typing import Dict, List, Tuple

from simulator.algorithms.homogeneous.math_helpers import checked_log as _ln
from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.core.runner import Runner

class Izumi2026PhaseMixin:
    """Initialization phases for HomogeneousMultiChannelIzumi2026."""

    def find_multiple_good_arms(
        self, runner: Runner
    ) -> Tuple[List[int], Dict[int, float]]:
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
                accept = [
                    (N[m][ell_idx] > 0 and R[m][ell_idx] / N[m][ell_idx] >= thr)
                    for m in range(M)
                ]

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

                # R'[ell] >= 1 なら ell を G に追加する
                for m in range(M):
                    if len(G_list[m]) < n and Rprime[m][ell_idx] >= 1:
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

    # ------------------------------------------------------------------
    # Algorithm 2: ParallelVirtualMusicalChairs
    # ------------------------------------------------------------------

    def parallel_virtual_musical_chairs(
        self, runner: Runner, good_arms: List[int], tau: int
    ) -> List[int]:
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
                            available = [
                                slot for slot in range(K) if slot not in forbidden_set
                            ]
                            if not available:
                                # フォールバック: spreading を緩めて全スロットから選ぶ
                                available = list(range(K))
                            l[m][i0] = self._player_rngs[m].choice(available)
                            # 次の arm i0+1 のための forbidden set を更新する
                            # 0-based: {(l[m][j0] + (i0+1-j0)) % K | j0 <= i0}
                            # これは arm j0 と arm i0+1 が同一ブロック内の同じタイムステップに
                            # 割り当てられることを防ぐための禁止リストを更新する。
                            forbidden_set = {
                                (l[m][j0] + (i0 + 1 - j0)) % K
                                for j0 in range(i0 + 1)
                            }
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

    # ------------------------------------------------------------------
    # Algorithm 3: ParallelVirtualNumberPlayers
    # ------------------------------------------------------------------

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
                        # v が 2K を超えたら skip
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
                # 条件（0-based）: ell[m][i0] != -1 AND (ell[m][i0] + i0) % K == k_0
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

                # 3. τ 回後: R == 0 なら collision → M_hat, j を更新する
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

    # ------------------------------------------------------------------
    # Algorithm 4: HierarchicalDistributedExploration
    # ------------------------------------------------------------------

