from __future__ import annotations

import math
from typing import Dict, List

from simulator.algorithms.homogeneous.huang2022.communication import Huang2022CommunicationMixin
from simulator.algorithms.homogeneous.math_helpers import checked_log as _ln
from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.core.runner import Runner

class Huang2022ExplorationMixin(Huang2022CommunicationMixin):
    """Exploration phase for HomogeneousHuang2022."""

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

                # 補完割当 fallback は論文手順ではないため通常経路からは呼ばない。
                # 未割当が残る場合は次 phase に進め、実験側で success=False として扱う。

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

