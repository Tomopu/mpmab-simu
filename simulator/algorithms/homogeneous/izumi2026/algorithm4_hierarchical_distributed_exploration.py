from __future__ import annotations

import math
from typing import Dict, List

from simulator.algorithms.homogeneous.izumi2026.communication import Izumi2026CommunicationMixin
from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.algorithms.homogeneous.math_helpers import checked_log as _ln
from simulator.core.runner import Runner


class Izumi2026HierarchicalDistributedExplorationMixin(Izumi2026CommunicationMixin):
    """Algorithm 4: HierarchicalDistributedExploration."""

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
            raise ValueError("j==1 の Grand Leader が存在しない。")

        # M0: 現在の active players 数（推定値の最大を使う）
        M0 = max(M_hat_list)

        p = 0
        f = [-1] * M
        R = [[0] * K for _ in range(M)]  # 累積報酬
        v_cnt = [[0] * K for _ in range(M)]  # サンプル数
        E = [[0.0] * K for _ in range(M)]  # 推定平均報酬

        # Grand Leader / Sub-Leader が保持する全プレイヤー集約統計
        mu_hat = [[0.0] * M for _ in range(K)]  # mu_hat[k][pid]
        N_mat = [[0] * M for _ in range(K)]  # N_mat[k][pid]

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
            n_subleaders = sum(1 for m in range(M) if 2 <= j_list[m] <= n and f[m] == -1)
            comm_uplink_sub = n_subleaders * Ka * Q * tau

            # 通信コストを Trace に記録するためダミー step を消費する
            # 実際の bit 伝送は行わず、時間コストのみをシミュレートする（簡略化）
            self._consume_comm_steps(runner, comm_uplink_follower + comm_uplink_sub, dummy)

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
            self._consume_comm_steps(runner, comm_downlink_sub + comm_downlink_follower, dummy)

            # 3. 各プレイヤーが AssignAndUpdate を実行して割当を決定する
            assigned_before = set(a for a in f if a >= 0)
            if C_accept:
                if n == 1:
                    # n=1 は Huang 2022 と同じ通信チャンネル 1 本のケース。
                    # good arm が accept された場合、leader (j=1) だけが good arm を
                    # 受け取り、follower は good arm を除いた accept 集合から割り当てる。
                    good_arm = good_arms[0]
                    C0_accept = [a for a in C_accept if a != good_arm]
                    good_arm_accepted = good_arm in C_accept

                    if f[grand_leader] == -1:
                        if M0 - 1 == len(C0_accept) and good_arm_accepted:
                            f[grand_leader] = good_arm
                        elif len(C0_accept) >= M0:
                            f[grand_leader] = C0_accept[M0 - 1]

                    for m in range(M):
                        if m == grand_leader or f[m] != -1:
                            continue
                        idx = M0 - j_list[m]
                        if 0 <= idx < len(C0_accept):
                            f[m] = C0_accept[idx]
                else:
                    # n>1: Grand Leader (j=1) と Sub-Leader (2<=j<=n) は
                    # 担当チャンネル good_arms[j-1] が C_accept に入ったとき割り当て。
                    # Follower (j>n) は未割当フォロワー内の相対 rank ベースで割り当て。
                    #
                    # _try_assign（idx = M0 - j）を使わない理由:
                    #   leader が先に割り当てられると M0 が減少し、
                    #   idx = M0 - j が負になってフォロワーが永久に未割当になる。
                    C_accept_set = set(C_accept)
                    C_assign_list = [a for a in C_accept if a not in good_set]

                    # 1. leader 割り当て（j<=n）
                    for m in range(M):
                        if f[m] == -1 and j_list[m] <= n:
                            channel_arm = good_arms[j_list[m] - 1]
                            if channel_arm in C_accept_set:
                                f[m] = channel_arm

                    # 2. follower 割り当て（j>n）: 未割当フォロワーを j の降順にソートし、
                    #    C_assign_list の先頭から順に割り当てる（j が大きいほど先に割り当て）
                    unassigned_followers = sorted(
                        [m for m in range(M) if f[m] == -1 and j_list[m] > n],
                        key=lambda m: j_list[m],
                        reverse=True,
                    )
                    for idx, m in enumerate(unassigned_followers):
                        if idx < len(C_assign_list):
                            f[m] = C_assign_list[idx]

                # 未割当が残る場合は次 phase に進め、実験側で success=False として扱う。

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
