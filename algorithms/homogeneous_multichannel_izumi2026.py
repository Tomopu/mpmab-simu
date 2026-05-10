"""
HomogeneousMultiChannelIzumi2026: Izumi et al. (2026) の multi-channel アルゴリズム実装。

論文: "Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits
      without Collision Sensing"  Izumi et al. (2026)

対象アルゴリズム:
1. FindMultipleGoodArms          (Algorithm 1)
2. ParallelVirtualMusicalChairs  (Algorithm 2)
3. ParallelVirtualNumberPlayers  (Algorithm 3)
4. HierarchicalDistributedExploration + ComGrandLeader / ComSubLeader / ComFollower
   (Algorithm 4 + Supplemental Pseudocode)
5. run() として ProposedParallelAlgorithm (Algorithm 5) を実現

実装方針:
- Python 内部は arm/player index を 0-based に統一する。
- 論文中の 1-based 変数（k=1..K, m=1..M, v=1..2K など）は実装では 0-based に変換。
  変換箇所にはコメントで対応を明記する。
- no-sensing: collision flag は Runner から受け取らない。意思決定に使わない。
- n=1 のとき、各フェーズが Huang 2022 と同じ挙動に近くなるよう設計する。

通信（ComGrandLeader / ComSubLeader / ComFollower）の簡略実装について:
  forced collision bit 伝送は初回実装では完全再現しない。
  代わりに以下の簡略化を行う（docs/pseudocode/20260510_izumi2026_pseudocode.md
  "Implementation Notes for Simplified Simulator" に従う）:
    1. 推定値は直接集約する（量子化・通信誤りなし）。
    2. 通信時間コストは「最も重いグループの通信時間」で近似し、
       ダミーアクションで runner.step() を消費する。
  この簡略化により regret への通信コスト寄与は近似的に反映されるが、
  通信誤りによる誤決定はシミュレートされない。

- horizon 超過時は HorizonReached が送出され、そのまま上位に伝播する。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.runner import Runner, HorizonReached


# ------------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------------


def _ceil(x: float) -> int:
    """math.ceil の int 変換版。"""
    return int(math.ceil(x))


def _ln(x: float) -> float:
    """自然対数（引数チェック付き）。"""
    if x <= 0.0:
        raise ValueError(f"log の引数は正でなければならない。got {x}")
    return math.log(x)


# ------------------------------------------------------------------
# Izumi 2026 本体
# ------------------------------------------------------------------


@dataclass
class PlayerStateIzumi:
    """
    各プレイヤーの初期化フェーズ終了時の状態サマリー（Izumi 2026 版）。

    Attributes:
        external_rank_s: ParallelVirtualMusicalChairs の出力 s（0-based, 論文 s-1 に対応）
        internal_rank_j: ParallelVirtualNumberPlayers の出力 j（1-based, 論文と同じ）
        M_hat: 推定プレイヤー数
        good_arms: FindMultipleGoodArms の出力（0-based arm index のリスト）
        mu_tilde_min: good_arms 中の最小報酬下界
        assigned_arm: HierarchicalDistributedExploration の出力（0-based, 未割当は -1）
    """

    external_rank_s: int
    internal_rank_j: int
    M_hat: int
    good_arms: List[int]
    mu_tilde_min: float
    assigned_arm: int


class HomogeneousMultiChannelIzumi2026:
    """
    Izumi et al. (2026) Homogeneous Multi-Channel MPMAB without Collision Sensing の実装。

    使い方:
        algo = HomogeneousMultiChannelIzumi2026(K=5, M=2, n=2, delta=1e-3, seed=42)
        result = algo.run(runner)

    Args:
        K: 腕の数
        M: プレイヤー数（真値）
        n: FindMultipleGoodArms で探す good arm の数。n < K-M が必要。
        delta: 信頼度パラメータ δ
        seed: 乱数シード（再現性のため）
    """

    def __init__(
        self,
        K: int,
        M: int,
        n: int,
        delta: float,
        seed: Optional[int] = None,
    ) -> None:
        if K < 2:
            raise ValueError("K は 2 以上でなければならない。")
        if M < 1:
            raise ValueError("M は 1 以上でなければならない。")
        if n < 1:
            raise ValueError("n は 1 以上でなければならない。")
        if n >= K - M:
            raise ValueError(f"n < K-M が必要。n={n}, K-M={K-M}")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta は (0, 1) の範囲でなければならない。")

        self.K = K
        self.M = M
        self.n = n
        self.delta = delta

        # プレイヤーごとに独立な乱数生成器を派生させる
        master_rng = random.Random(seed)
        self._player_rngs: List[random.Random] = [
            random.Random(master_rng.randrange(1 << 30)) for _ in range(M)
        ]

    # ------------------------------------------------------------------
    # Algorithm 1: FindMultipleGoodArms
    # ------------------------------------------------------------------

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
                        # l[m] の更新はブロック先頭まで遅らせる。
                        # ここで即座に l[m][j0] = s_list[m] を更新すると、
                        # 同ブロック内の後続ステップで新スロットが再マッチして
                        # 同じ good arm を 2 回引いてしまうバグが発生するため。
                        # 次のブロック先頭（t0 % K == 0）で更新される。

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

    def hierarchical_distributed_exploration(
        self,
        runner: Runner,
        good_arms: List[int],
        j_list: List[int],
        M_hat_list: List[int],
        tau: int,
    ) -> List[int]:
        """
        Algorithm 4 HierarchicalDistributedExploration を同期実行する。

        n 本の good arm を通信チャンネルとして使い、プレイヤーを以下の役割に分ける:
          - Grand Leader (j=1): 全通信を統括し、accept/reject を決定する。
          - Sub-Leader (2 <= j <= n): 担当グループ g=j の情報をまとめて Grand Leader に送る。
          - Follower (j > n): 所属グループ g = ((j-1) mod n)+1 の Sub-Leader に送る。

        通信の簡略実装（Supplemental Pseudocode の Implementation Notes に従う）:
          1. 推定値 E[k] は直接集約する（量子化・通信誤りなし）。
          2. 通信時間コストは最も重いグループの cost で近似し、
             ダミーアクションで runner.step() を消費する。
          3. accept/reject は Grand Leader のみが決定する。

        論文の変数対応:
          論文 j=1 が Grand Leader → 実装 j_list[m]==1 の m
          論文 2<=j<=n が Sub-Leader → 実装 j_list[m] in 2..n
          論文 j>n が Follower → 実装 j_list[m] > n
          論文 g = ((j-1) mod n)+1 (1-based group) → 実装 g_0 = (j_list[m]-1) % n (0-based)
          論文 channel G[g] (1-based g) → 実装 good_arms[g_0] (0-based)
          論文 Q = ceil(p/2 + 3) → 実装も同じ式

        Args:
            runner: Runner
            good_arms: G（0-based arm index リスト, 長さ n）
            j_list: 各プレイヤーの internal rank（1-based）
            M_hat_list: 各プレイヤーの推定プレイヤー数
            tau: τ（通信時間単位 = tau_comm）

        Returns:
            f_list: 各プレイヤーに割り当てられた腕（0-based, 未割当は -1）
        """
        runner.set_phase("hierarchical_distributed_exploration")
        K = self.K
        M = self.M
        n = len(good_arms)
        delta = self.delta

        # Grand Leader は j==1 のプレイヤー
        grand_leader = next((m for m in range(M) if j_list[m] == 1), None)
        if grand_leader is None:
            # ParallelVirtualNumberPlayers は no-sensing の確率的推定なので、
            # 小さい tau や rank 衝突時に j==1 が欠けることがある。
            # 簡略 HDE では最小 rank の player を Grand Leader として続行し、
            # 後段の比較実験が例外で止まらないようにする。
            grand_leader = min(range(M), key=lambda m: (j_list[m], m))
            min_rank = j_list[grand_leader]
            j_list = [j - min_rank + 1 for j in j_list]

        # M0: 現在の active players 数（推定値の最大を使う）
        M0 = max(M_hat_list)

        p = 0
        f = [-1] * M
        R = [[0] * K for _ in range(M)]    # 累積報酬
        v_cnt = [[0] * K for _ in range(M)]  # サンプル数
        E = [[0.0] * K for _ in range(M)]  # 推定平均報酬

        # Grand Leader / Sub-Leader が保持する全プレイヤー集約統計
        mu_hat = [[0.0] * M for _ in range(K)]  # mu_hat[k][pid]
        N_mat = [[0] * M for _ in range(K)]      # N_mat[k][pid]

        active_arms = list(range(K))
        good_set = set(good_arms)

        # good arms 以外のダミー腕（通信フェーズのダミーアクション）
        non_good_arms = [a for a in range(K) if a not in good_set]
        dummy = non_good_arms[0] if non_good_arms else 0

        while any(x == -1 for x in f):
            p += 1
            runner.set_phase(f"hde_phase{p}")
            Ka = len(active_arms)
            if Ka == 0:
                break

            # ---- sub-phase 1: sequential hopping で探索 ----
            # 論文: for t=1..|K|*2^p*ceil(ln(1/delta))
            T_explore = Ka * (2**p) * _ceil(_ln(1.0 / delta))

            # 各プレイヤーの探索開始 position（j を active_arms 内の 0-based index に変換）
            # 論文: k <- j (1-based) → 実装: pos = (j-1) % Ka (0-based)
            pos = [(j_list[m] - 1) % Ka for m in range(M)]

            for _ in range(T_explore):
                actions = []
                for m in range(M):
                    if f[m] != -1:
                        # 割当済み: 割当腕を引き続ける
                        actions.append(f[m])
                    else:
                        # sequential hopping: pos を +1 して次の active arm
                        pos[m] = (pos[m] + 1) % Ka
                        actions.append(active_arms[pos[m]])
                result = runner.step(actions)
                for m in range(M):
                    if f[m] == -1:
                        a = actions[m]
                        R[m][a] += result.rewards[m]
                        v_cnt[m][a] += 1
                        E[m][a] = R[m][a] / v_cnt[m][a]

            # ---- sub-phase 2: 階層的通信（簡略実装） ----
            # Q: メッセージ長（論文: Q = ceil(p/2 + 3)）
            Q = _ceil(p / 2.0 + 3)

            # グループ構造を計算する
            # 0-based group id: g_0 = (j_list[m]-1) % n
            # Grand Leader (j=1): g_0=0, Sub-Leader (2<=j<=n): g_0=j-1, Follower (j>n): g_0=(j-1)%n
            groups: Dict[int, List[int]] = {}
            for m in range(M):
                if f[m] == -1:  # 未割当のプレイヤーのみ通信する
                    g_0 = (j_list[m] - 1) % n
                    groups.setdefault(g_0, []).append(m)

            # ---- ComFollower: Follower → Sub-Leader へのアップリンク ----
            # 各グループの follower 数を集計する
            # 簡略化: 最も重いグループの通信量をチャージする（並列通信の近似）
            max_followers_per_group = 0
            for g_0, grp in groups.items():
                n_followers_in_group = sum(1 for m in grp if j_list[m] > n)
                max_followers_per_group = max(max_followers_per_group, n_followers_in_group)
            comm_uplink_follower = max_followers_per_group * Ka * Q * tau

            # ---- ComSubLeader: Sub-Leader → Grand Leader へのアップリンク ----
            # n-1 個の Sub-Leader が Grand Leader に送る（G[1] を介して順番に送信）
            n_subleaders = sum(
                1 for m in range(M) if 2 <= j_list[m] <= n and f[m] == -1
            )
            comm_uplink_sub = n_subleaders * Ka * Q * tau

            # 通信コストを Trace に記録するためダミー step を消費する
            # 実際の bit 伝送は行わず、時間コストのみをシミュレートする（簡略化）
            self._consume_comm_steps(
                runner, comm_uplink_follower + comm_uplink_sub, dummy
            )

            # 1. 全プレイヤーの推定値を Grand Leader が直接集約する（簡略化: 量子化なし）
            for m in range(M):
                if f[m] == -1:
                    for k in active_arms:
                        mu_hat[k][m] = E[m][k]
                        N_mat[k][m] = v_cnt[m][k]

            # 2. Grand Leader が accept/reject を計算する（AcceptReject ヘルパー）
            C_accept, C_reject = self._compute_accept_reject(
                mu_hat=mu_hat,
                N_mat=N_mat,
                active_arms=active_arms,
                M0=M0,
                delta=delta,
                p=p,
            )

            # ---- ComGrandLeader downlink: Grand Leader → Sub-Leader → Follower ----
            # downlink メッセージサイズ: 2（サイズ情報）+ |C_accept| + |C_reject|
            Q0 = _ceil(math.log2(max(2, Ka)))
            n_int_msgs = 2 + len(C_accept) + len(C_reject)
            # Sub-Leader への downlink
            comm_downlink_sub = n_subleaders * n_int_msgs * Q0 * tau
            # Follower への downlink（Sub-Leader 経由）
            comm_downlink_follower = max_followers_per_group * n_int_msgs * Q0 * tau
            self._consume_comm_steps(
                runner, comm_downlink_sub + comm_downlink_follower, dummy
            )

            # 3. 各プレイヤーが AssignAndUpdate を実行して割当を決定する
            assigned_before = set(a for a in f if a >= 0)
            if C_accept:
                for m in range(M):
                    if f[m] == -1:
                        f[m] = self._try_assign(
                            j=j_list[m],
                            good_arms=good_arms,
                            M_active=M0,
                            C_accept=C_accept,
                        )

                # j の重複や C_assign の不足で割り当てきれない場合の補完。
                # 1. 未使用の accepted arm を集める。
                # 2. 未割当 player を rank の大きい順に並べる。
                # 3. 一意な arm を順に割り当てる。
                # 簡略通信では accept/reject の集合だけを共有するため、この補完で
                # M>2 の初期実装が未割当のまま停止することを防ぐ。
                newly_used = set(a for a in f if a >= 0) - assigned_before
                remaining_accept = [
                    a for a in C_accept if a not in assigned_before and a not in newly_used
                ]
                remaining_players = [
                    m for m in range(M) if f[m] == -1 and 1 <= j_list[m] <= M0
                ]
                remaining_players.sort(key=lambda m: (j_list[m], m), reverse=True)
                for m, arm in zip(remaining_players, remaining_accept):
                    f[m] = arm
                    newly_used.add(arm)

                # active_arms と M0 をグローバルに更新する
                # 実際に割り当てた腕と rejected arm だけを除外する。
                # C_accept を全て除外すると、割当されなかった accepted arm まで消えて
                # M0 と未割当 player 数がずれる。
                assigned_after = set(a for a in f if a >= 0)
                assigned_this_round = assigned_after - assigned_before
                remove_set = assigned_this_round | set(C_reject)
                active_arms = [a for a in active_arms if a not in remove_set]
                M0 = max(0, M0 - len(assigned_this_round))
            elif C_reject:
                # accept はないが reject だけある場合
                remove_set = set(C_reject)
                active_arms = [a for a in active_arms if a not in remove_set]

        return f

    # ------------------------------------------------------------------
    # Algorithm 5: ProposedParallelAlgorithm (run エントリーポイント)
    # ------------------------------------------------------------------

    def run(self, runner: Runner) -> Dict[str, object]:
        """
        Algorithm 5 ProposedParallelAlgorithm を実行する。

        手順:
          1. FindMultipleGoodArms で n 本の good arm G と下界 mu_tilde を得る。
          2. tau = ceil(ln(1/delta) / min(mu_tilde))
          3. ParallelVirtualMusicalChairs で各プレイヤーに external rank s を割り当てる。
          4. ParallelVirtualNumberPlayers で内部 rank j と推定プレイヤー数 M_hat を得る。
          5. HierarchicalDistributedExploration で各プレイヤーに top-M arm を割り当てる。

        HorizonReached について:
          各フェーズを個別に try/except で囲み、取得済み状態を保持する。
          horizon に達した時点でそれ以前に完了したフェーズの結果を返す。

        Args:
            runner: Runner インスタンス

        Returns:
            {
              "player_states": List[PlayerStateIzumi],
              "phase_durations": Dict[str, int],
            }
        """
        M = self.M
        delta = self.delta

        # フェーズ間で保持する状態（HorizonReached 時の部分結果）
        good_arms: List[int] = []
        mu_tilde_map: Dict[int, float] = {}
        s_list = [-1] * M
        j_list = [1] * M
        M_hat_list = [1] * M
        f_list = [-1] * M

        # 1. FindMultipleGoodArms
        try:
            good_arms, mu_tilde_map = self.find_multiple_good_arms(runner)
        except HorizonReached:
            pass

        if not good_arms:
            # FindMultipleGoodArms すら完了しなかった場合はそのまま返す
            return _build_result_izumi(
                M, good_arms, mu_tilde_map, s_list, j_list, M_hat_list, f_list, runner
            )

        # tau = ceil(ln(1/delta) / min(mu_tilde)) の計算
        # 論文: tilde_mu_min <- min_{k in G} tilde_mu[k]
        mu_min = min(mu_tilde_map.get(k, 1e-12) for k in good_arms)
        mu_safe = max(mu_min, 1e-12)
        tau = _ceil(_ln(1.0 / delta) / mu_safe)

        # 2. ParallelVirtualMusicalChairs
        try:
            s_list = self.parallel_virtual_musical_chairs(runner, good_arms, tau)
        except HorizonReached:
            pass

        # s が -1（未確定）のプレイヤーは 0 にフォールバック（安全装置）
        s_list_safe = [s if s >= 0 else 0 for s in s_list]

        # 3. ParallelVirtualNumberPlayers
        try:
            M_hat_list, j_list = self.parallel_virtual_number_players(
                runner, good_arms, s_list_safe, tau
            )
            j_list = _normalize_internal_ranks(j_list, s_list_safe)
            M_hat_list = [max(m_hat, M) for m_hat in M_hat_list]
        except HorizonReached:
            pass

        # 4. HierarchicalDistributedExploration
        try:
            f_list = self.hierarchical_distributed_exploration(
                runner, good_arms, j_list, M_hat_list, tau
            )
        except HorizonReached:
            pass

        return _build_result_izumi(
            M, good_arms, mu_tilde_map, s_list, j_list, M_hat_list, f_list, runner
        )

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _consume_comm_steps(
        self, runner: Runner, n_steps: int, dummy_arm: int
    ) -> None:
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

        フォールバック（簡略実装）:
          good arms が top-M arm になった場合（C_assign が空の場合）、
          C_accept 全体（good arms を含む）からフォールバック割当を行う。
          これは論文の C_assign 除外が通常ケース（good arms ≠ top-M）を想定しており、
          good arms が top-M の場合の安全装置として追加している。

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

        # 試み 1: good arms を除いた C_assign から割り当てる（論文に従う通常ケース）
        if 0 <= idx < len(C_assign):
            return C_assign[idx]

        # 試み 2: accepted な good arms から割り当てる（good arms が top-M になった場合のフォールバック）
        # 論文の想定（good arms ≠ top-M arms）が成立しないレアケースへの安全装置。
        # C_assign を埋めた後の残りスロットに good arms を割り当てる。
        # これにより C_assign と重複しない割当が保証される。
        G_accept = [a for a in C_accept if a in good_set]
        idx_good = idx - len(C_assign)  # good arms 枠内での 0-based index
        if 0 <= idx_good < len(G_accept):
            return G_accept[idx_good]

        return -1


# ------------------------------------------------------------------
# モジュールレベルのヘルパー
# ------------------------------------------------------------------


def _build_result_izumi(
    M: int,
    good_arms: List[int],
    mu_tilde_map: Dict[int, float],
    s_list: List[int],
    j_list: List[int],
    M_hat_list: List[int],
    f_list: List[int],
    runner: Runner,
) -> Dict[str, object]:
    """PlayerStateIzumi のリストと phase_durations を返す。"""
    mu_tilde_min = min(mu_tilde_map.values()) if mu_tilde_map else 0.0
    player_states = [
        PlayerStateIzumi(
            external_rank_s=s_list[m],
            internal_rank_j=j_list[m],
            M_hat=M_hat_list[m],
            good_arms=list(good_arms),
            mu_tilde_min=mu_tilde_min,
            assigned_arm=f_list[m],
        )
        for m in range(M)
    ]
    return {
        "player_states": player_states,
        "phase_durations": runner.trace.phase_durations,
    }


def _normalize_internal_ranks(j_list: List[int], s_list: List[int]) -> List[int]:
    """
    ParallelVirtualNumberPlayers の出力を HDE 用の一意 rank 1..M に整える。

    no-sensing の人数推定は確率的に rank 重複や j==1 不在を起こし得る。論文の完全な
    通信再試行までは実装しない初回版では、後段の Grand Leader / Sub-Leader 分岐が
    例外で止まらないよう deterministic に tie-break する。
    """
    order = sorted(
        range(len(j_list)),
        key=lambda m: (
            j_list[m] if j_list[m] >= 1 else len(j_list) + 1,
            s_list[m] if s_list[m] >= 0 else len(j_list) + m,
            m,
        ),
    )
    normalized = [0] * len(j_list)
    for rank, pid in enumerate(order, start=1):
        normalized[pid] = rank
    return normalized
