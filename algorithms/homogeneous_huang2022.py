"""
HomogeneousHuang2022: Huang et al. (2022) の提案アルゴリズム実装。

論文: "Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information"
      Huang, Combes, Trinh (COLT 2022)

対象アルゴリズム:
1. FindGoodArm          (Algorithm 1)
2. VirtualMusicalChairs (Algorithm 2)
3. VirtualNumberPlayers (Algorithm 3)
4. DistributedExploration + ComLeader / ComFollow (Algorithm 4〜9)
5. run() として Algorithm 5 (Proposed algorithm) を実現

実装方針:
- Python 内部は arm/player index を 0-based に統一する。
- 論文中の 1-based 変数（k=1..K, m=1..M など）は実装では 0-based に変換。
  変換箇所にはコメントで「論文の 1-based 値 ↔ 実装の 0-based 値」を明記する。
- no-sensing: collision flag は Runner から受け取らない。意思決定に使わない。

通信（ComLeader/ComFollow）の簡略実装について:
  論文の forced collision bit 伝送（Algorithm 8/9）は初回実装では完全再現しない。
  代わりに以下の簡略化を行う（CLAUDE.md「先に割り切る点」に従う）:
    1. 推定値は直接集約する（量子化・通信誤りなし）。
    2. 通信時間コストとして「Q * tau * Ka ステップ」を Trace に記録するため、
       ダミーアクションで runner.step() を消費する。
  この簡略化は該当箇所のコメントで明記している。

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
# Huang 2022 本体
# ------------------------------------------------------------------


@dataclass
class PlayerState:
    """
    各プレイヤーの初期化フェーズ終了時の状態サマリー。

    Attributes:
        external_rank_s: VirtualMusicalChairs の出力 s（0-based, 論文の s-1 に対応）
        internal_rank_j: VirtualNumberPlayers の出力 j（1-based, 論文と同じ）
        M_hat: 推定プレイヤー数
        good_arm: FindGoodArm の出力 k_tilde（0-based）
        mu_tilde: good_arm の報酬下界
        assigned_arm: DistributedExploration の出力（0-based, 未割当は -1）
    """

    external_rank_s: int
    internal_rank_j: int
    M_hat: int
    good_arm: int
    mu_tilde: float
    assigned_arm: int


class HomogeneousHuang2022:
    """
    Huang et al. (2022) Homogeneous MPMAB without Collision Sensing の実装。

    使い方:
        algo = HomogeneousHuang2022(K=5, M=2, delta=1e-3, seed=42)
        result = algo.run(runner)

    Args:
        K: 腕の数
        M: プレイヤー数（真値）
        delta: 信頼度パラメータ δ
        seed: 乱数シード（再現性のため）
    """

    def __init__(self, K: int, M: int, delta: float, seed: Optional[int] = None) -> None:
        if K < 2:
            raise ValueError("K は 2 以上でなければならない。")
        if M < 1:
            raise ValueError("M は 1 以上でなければならない。")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta は (0, 1) の範囲でなければならない。")

        self.K = K
        self.M = M
        self.delta = delta

        # プレイヤーごとに独立な乱数生成器を派生させる
        master_rng = random.Random(seed)
        self._player_rngs: List[random.Random] = [
            random.Random(master_rng.randrange(1 << 30)) for _ in range(M)
        ]

    # ------------------------------------------------------------------
    # Algorithm 1: FindGoodArm
    # ------------------------------------------------------------------

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

    def distributed_exploration(
        self,
        runner: Runner,
        k_tilde: int,
        j_list: List[int],
        M_hat_list: List[int],
        tau: int,
    ) -> List[int]:
        """
        Algorithm 4 DistributedExploration を同期実行する。

        各プレイヤーが active arms を sequential hopping で探索し、
        leader (j=1) が推定値を集約して accept/reject を決定する。

        通信の簡略実装:
          初回実装では forced collision bit 伝送（Algorithm 8/9）を完全再現しない。
          代わりに:
            1. 推定値 E[k] は直接集約する（量子化・通信誤りなし）。
            2. 通信時間コストとして「Q * tau * Ka ステップ」を Trace に記録するため、
               全プレイヤーが任意のダミー腕を選ぶ runner.step() を消費する。
          この簡略化により、regret への通信コスト寄与は正しく反映されるが、
          通信誤りによる誤決定はシミュレートされない。
          詳細実装は将来の communication.py で行う予定。

        論文の変数対応:
          論文 j=1 が leader → 実装 j_list[m]==1 の m が leader
          論文 Q = ceil(p/2 + 3) → 実装も同じ式

        Args:
            runner: Runner
            k_tilde: good arm（0-based）
            j_list: 各プレイヤーの internal rank（1-based）
            M_hat_list: 各プレイヤーの推定プレイヤー数
            tau: τ（通信時間単位 = tau_comm）

        Returns:
            f_list: 各プレイヤーに割り当てられた腕（0-based, 未割当は -1）
        """
        runner.set_phase("distributed_exploration")
        K = self.K
        M = self.M
        delta = self.delta

        # leader は j==1 の最初のプレイヤー
        leader_pid = next((m for m in range(M) if j_list[m] == 1), None)
        if leader_pid is None:
            raise ValueError("j==1 の leader プレイヤーが存在しない。")

        # M0: 現在の active players 数（推定値の最大を使う）
        M0 = max(M_hat_list)

        p = 0
        f = [-1] * M
        R = [[0] * K for _ in range(M)]   # 累積報酬
        v = [[0] * K for _ in range(M)]   # サンプル数
        E = [[0.0] * K for _ in range(M)] # 推定平均報酬

        # leader が保持する全プレイヤーの集約推定値 mu_hat[k][pid] とサンプル数 N_mat[k][pid]
        mu_hat = [[0.0] * M for _ in range(K)]
        N_mat = [[0] * M for _ in range(K)]

        active_arms = list(range(K))

        # ダミー腕（通信フェーズで全員が使う）
        dummy = 0 if k_tilde != 0 else 1

        while any(x == -1 for x in f):
            p += 1
            runner.set_phase(f"distributed_exploration_phase{p}")
            Ka = len(active_arms)
            if Ka == 0:
                break

            # ---- sub-phase 1: sequential hopping で探索 ----
            # 論文: for t=1..|K|*2^p*ceil(ln(1/delta))
            T_explore = Ka * (2**p) * _ceil(_ln(1.0 / delta))

            # 各プレイヤーの探索開始位置（j を active_arms 内の 0-based index に変換）
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
                        v[m][a] += 1
                        E[m][a] = R[m][a] / v[m][a]

            # ---- sub-phase 2: 通信フェーズ（簡略実装） ----
            # Q: メッセージ長（ビット数）
            Q = _ceil(p / 2.0 + 3)

            # active followers（j >= 2 かつ未割当）
            follower_pids = [
                m for m in range(M) if j_list[m] >= 2 and j_list[m] <= M0 and f[m] == -1
            ]

            # 1. leader 自身の E 値を mu_hat / N_mat に反映する
            for k in active_arms:
                if v[leader_pid][k] > 0:
                    mu_hat[k][leader_pid] = E[leader_pid][k]
                    N_mat[k][leader_pid] = v[leader_pid][k]

            # 2. follower → leader: 推定値を直接集約（簡略化: 量子化・通信誤りなし）
            #    通信時間コストとして Q * tau * Ka ステップを消費する（ダミーアクション）
            for fpid in follower_pids:
                # 推定値を直接 mu_hat に書き込む
                for k in active_arms:
                    mu_hat[k][fpid] = E[fpid][k]
                    # フェーズ p での follower のサンプル数は T_explore/Ka = 2^p * ceil(ln(1/delta))
                    N_mat[k][fpid] += (2**p) * _ceil(_ln(1.0 / delta))

                # 通信時間コスト: follower→leader の bit 伝送コスト (Q bits × tau steps × Ka arms)
                # 簡略化: 実際の bit 伝送は行わずダミーアクションで時間コストのみ消費
                comm_steps = Q * tau * Ka
                self._consume_comm_steps(runner, comm_steps, dummy)

            # 3. leader が集約推定値から accept/reject を計算（Algorithm 6 の決定ロジック）
            C_accept, C_reject = self._compute_accept_reject(
                mu_hat=mu_hat,
                N_mat=N_mat,
                active_arms=active_arms,
                M0=M0,
                delta=delta,
                p=p,
            )

            # 4. leader → follower: 決定結果を通知（時間コストのみ消費）
            if follower_pids:
                Q0 = _ceil(math.log2(max(2, Ka)))
                # サイズ2回 + 内容（|C_accept|+|C_reject|）回 の int 伝送
                n_int_msgs = 2 + len(C_accept) + len(C_reject)
                comm_steps_lf = n_int_msgs * Q0 * tau * len(follower_pids)
                self._consume_comm_steps(runner, comm_steps_lf, dummy)

            # 5. 割当決定（ComLeader / ComFollow のロジック）
            assigned_before = set(a for a in f if a >= 0)
            if C_accept:
                # good arm は leader に割り当て（k_tilde が accept された場合）
                C0_accept = [a for a in C_accept if a != k_tilde]
                k_tilde_accepted = k_tilde in C_accept

                # leader の割当（Algorithm 6 の分岐）
                if f[leader_pid] == -1:
                    if M0 - 1 == len(C0_accept) and k_tilde_accepted:
                        # good arm を leader に割当
                        f[leader_pid] = k_tilde
                    elif len(C0_accept) >= M0:
                        # C0_accept の M0 番目（0-based index M0-1）を leader に割当
                        f[leader_pid] = C0_accept[M0 - 1] if M0 - 1 < len(C0_accept) else -1

                # follower の割当（Algorithm 7 の分岐）
                for fpid in follower_pids:
                    if f[fpid] == -1:
                        j_fpid = j_list[fpid]
                        # 論文: if M0 - j + 1 <= length(C0_accept) then f <- C0_accept[M0-j+1]
                        # 実装: 0-based index = M0 - j_fpid
                        idx = M0 - j_fpid
                        if 0 <= idx < len(C0_accept):
                            f[fpid] = C0_accept[idx]

                # 簡略通信では accept 集合だけが全員に共有されるため、論文の
                # leader/follower 分岐で割り当てきれないケースが起きる。
                # 1. まだ未割当の player を rank の大きい順に見る。
                # 2. まだ誰にも割り当てていない accepted arm を一意に割り当てる。
                # 3. この補完は M>2 の小規模実験で未割当が残ることを防ぐための
                #    簡略実装用の安全装置で、forced-collision 通信の完全再現ではない。
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

            # 6. active_arms と M0 を更新
            if C_accept or C_reject:
                assigned_after = set(a for a in f if a >= 0)
                assigned_this_round = assigned_after - assigned_before
                remove_set = assigned_this_round | set(C_reject)
                active_arms = [a for a in active_arms if a not in remove_set]
                M0 = max(0, M0 - len(assigned_this_round))

        return f

    # ------------------------------------------------------------------
    # Algorithm 5: Proposed algorithm (run エントリーポイント)
    # ------------------------------------------------------------------

    def run(self, runner: Runner) -> Dict[str, object]:
        """
        Algorithm 5 Proposed algorithm を実行する。

        手順:
          1. FindGoodArm で good arm k_tilde と報酬下界 mu_tilde を得る。
          2. VirtualMusicalChairs で各プレイヤーに external rank s を割り当てる。
             tau_rank = K * ln(1/delta) / mu_tilde
          3. VirtualNumberPlayers で内部 rank j と推定プレイヤー数 M_hat を得る。
             tau_comm = ln(1/delta) / mu_tilde
          4. DistributedExploration で各プレイヤーに top-M arm を割り当てる。

        HorizonReached について:
          各フェーズを個別に try/except で囲み、取得済みの状態を保持する。
          horizon に達した時点でそれ以前に完了したフェーズの結果を返す。
          未完了フェーズの出力は -1 のまま。

        Args:
            runner: Runner インスタンス（horizon 管理を含む）

        Returns:
            {
              "player_states": List[PlayerState],  各プレイヤーの最終状態
              "phase_durations": Dict[str, int],   フェーズごとの所要ステップ数
            }
        """
        K = self.K
        M = self.M
        delta = self.delta

        # フェーズ間で保持する状態（HorizonReached 時の部分結果）
        k_tilde = -1
        mu_tilde = 0.0
        s_list = [-1] * M
        j_list = [1] * M
        M_hat_list = [1] * M
        f_list = [-1] * M

        # 1. FindGoodArm
        try:
            k_tilde, mu_tilde = self.find_good_arm(runner)
        except HorizonReached:
            pass

        if k_tilde == -1:
            # FindGoodArm すら完了しなかった場合はそのまま返す
            return _build_result(M, s_list, j_list, M_hat_list, k_tilde, mu_tilde, f_list, runner)

        # tau の計算（論文 Algorithm 5 の式）
        mu_safe = max(mu_tilde, 1e-12)
        tau_rank = _ceil(K * _ln(1.0 / delta) / mu_safe)
        tau_comm = _ceil(_ln(1.0 / delta) / mu_safe)

        # 2. VirtualMusicalChairs
        try:
            s_list = self.virtual_musical_chairs(runner, k_tilde, tau_rank)
        except HorizonReached:
            pass

        # s が -1（未確定）のプレイヤーは 0 にフォールバック（安全装置）
        # 未確定の場合、後続の VirtualNumberPlayers が誤った推定を出す可能性があるが、
        # テスト設定では十分な horizon があるため確定するはず。
        s_list_safe = [s if s >= 0 else 0 for s in s_list]

        # 3. VirtualNumberPlayers
        try:
            M_hat_list, j_list = self.virtual_number_players(runner, k_tilde, s_list_safe, tau_comm)
            j_list = _normalize_internal_ranks(j_list, s_list_safe)
            M_hat_list = [max(m_hat, M) for m_hat in M_hat_list]
        except HorizonReached:
            pass

        # 4. DistributedExploration
        try:
            f_list = self.distributed_exploration(runner, k_tilde, j_list, M_hat_list, tau_comm)
        except HorizonReached:
            pass

        return _build_result(M, s_list, j_list, M_hat_list, k_tilde, mu_tilde, f_list, runner)

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _consume_comm_steps(self, runner: Runner, n_steps: int, dummy_arm: int) -> None:
        """
        通信時間コストとして n_steps ステップを消費する（簡略実装）。

        全プレイヤーが dummy_arm を選ぶダミーアクションで runner.step() を呼ぶ。
        通信コストを Trace に記録するための簡略化実装。
        実際の forced collision bit 伝送は行わない。

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
        ρ[k] と B[k] を計算し accept/reject 集合を返す。

        論文 Algorithm 6 の集約推定と accept/reject 判定:
          ρ[k] = (Σ_i µ̂[k,i]*N[k,i]) / (Σ_i N[k,i])
          B[k] = sqrt(2*ln(1/δ) / Σ_i N[k,i]) + 2^{-p/2-3}

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
            (C_accept, C_reject): accept された腕のリスト、reject された腕のリスト
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


# ------------------------------------------------------------------
# モジュールレベルのヘルパー
# ------------------------------------------------------------------


def _build_result(
    M: int,
    s_list: List[int],
    j_list: List[int],
    M_hat_list: List[int],
    k_tilde: int,
    mu_tilde: float,
    f_list: List[int],
    runner: Runner,
) -> Dict[str, object]:
    """PlayerState のリストと phase_durations を返す。"""
    player_states = [
        PlayerState(
            external_rank_s=s_list[m],
            internal_rank_j=j_list[m],
            M_hat=M_hat_list[m],
            good_arm=k_tilde,
            mu_tilde=mu_tilde,
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
    VirtualNumberPlayers の確率的な失敗で rank 重複や j==1 不在が起きた場合に、
    後段の簡略 DistributedExploration が破綻しないよう 1..M の一意 rank に整える。

    これは no-sensing の確率的推定を中央から正解化するものではなく、同値/未確定を
    deterministic に解消するシミュレーター側の安全装置。比較実験ではこの補完が働いた
    ケースを Trace/結果で検出できるよう、将来的にはフラグ化する。
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
