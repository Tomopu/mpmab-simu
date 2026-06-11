"""
BEACON エポック処理: Communication Phase + Exploration Phase。

BEACON: Leader (Algorithm 1) と BEACON: Follower m (Algorithm 2) を
同期シミュレーションとして実装する。

同期シミュレーションの方針:
    - 1 エポックあたり「通信フェーズ」+「探索フェーズ」の 2 段階を全員同期で実行する。
    - 通信フェーズでは follower→leader のデータ送信と leader→follower の割当送信を逐次実行する。
    - 探索フェーズでは全プレイヤーが割当 arm を 2^{p_r} ステップ引く。
    - 「stop signal」はシミュレーション上 2^{p_r} ステップで暗黙的に処理される。
    - HorizonReachedHetero は呼び出し元に伝播させ、last_assigned_arms を介して
      直前の割当を取得できる設計にする。

論文との対応:
    BEACON: Leader  (arXiv-2110.14622v2:208-237)
    BEACON: Follower m (arXiv-2110.14622v2:466-492)
    Send (Algorithm 3), Receive (Algorithm 4)

UCB 計算:
    mu_bar[k][m] = mu_tilde[k][m] + sqrt(3 * ln(t_r) / 2^{p[k][m]+1})
    t_r は各エポック開始時の累計ステップ数。

量子化:
    follower m が arm k を量子化するビット数: Q = ceil(1 + p[k][m] / 2)
    差分送信: sign(1bit) + abs(Q bits) = Q+1 bits
"""

from __future__ import annotations

import math
from typing import List

from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner
from simulator.algorithms.heterogeneous.shi2021.shi2021_beacon_communication import (
    BeaconCommunicationMixin,
)
from simulator.algorithms.heterogeneous.shi2021.oracle import matching_oracle


class BeaconExploreState:
    """
    BEACON エポックループで共有される探索状態。

    Attributes:
        T[k][m]: player m による arm k の exploratory サンプル数
        R[k][m]: player m による arm k の累積報酬
        samples[k][m]: player m による arm k の個別サンプル列（UCB に先頭 2^p 個を使う）
        K: arm 数
        M: プレイヤー数
        leader_pid: leader の player index（0-based）
        last_assigned_arms: 最終エポックの Oracle 割当（HorizonReached 時の取得用）
    """

    def __init__(
        self,
        T: List[List[int]],
        R: List[List[float]],
        samples: List[List[List[float]]],
        K: int,
        M: int,
        leader_pid: int,
    ) -> None:
        self.T = T
        self.R = R
        self.samples = samples
        self.K = K
        self.M = M
        self.leader_pid = leader_pid
        # 最終エポックの割当。HorizonReachedHetero が送出されたとき呼び出し元が参照する。
        self.last_assigned_arms: List[int] = list(range(M))


class BeaconEpochMixin(BeaconCommunicationMixin):
    """
    BEACON の初期化サンプリングとエポックループを実行する mixin。
    """

    def beacon_initial_sampling(
        self,
        runner: HeterogeneousRunner,
        state_list: List[int],
        rank_list: List[int],
    ) -> BeaconExploreState:
        """
        BEACON の初期サンプリングフェーズ（メインループ前の K ステップ）。

        Leader は各 arm k を 1 回ずつ引く（k=0..K-1）。
        Follower m (rank=r) は arm (r-1+k) mod K を引く（round-robin 探索）。

        論文:
            Leader:   Line 2 "Play each arm k in [K] and T^{r+1}_{k,1} += 1"
            Follower: Line 2 "In order k in [K], play arm [(m-1+k) mod K]"
                      1-based m → 0-based では rank_list[m]-1

        Args:
            runner: HeterogeneousRunner
            state_list: 各プレイヤーの state（0-based）
            rank_list: 各プレイヤーの rank（1-based）

        Returns:
            BeaconExploreState: 初期サンプリング後の状態
        """
        runner.set_phase("beacon_initial_sampling")
        K = self.K
        M = self.M

        T = [[0] * M for _ in range(K)]
        R = [[0.0] * M for _ in range(K)]
        # samples[k][m]: arm k の player m によるサンプル列
        samples: List[List[List[float]]] = [[[] for _ in range(M)] for _ in range(K)]

        leader_pid = next((m for m in range(M) if rank_list[m] == 1), 0)

        for k in range(K):
            actions = []
            for m in range(M):
                if rank_list[m] == 1:
                    # leader は arm k を順に引く
                    actions.append(k)
                else:
                    # follower (rank r, 1-based) は arm (r-1+k) mod K を引く
                    r_zero = rank_list[m] - 1  # 0-based follower index
                    actions.append((r_zero + k) % K)

            result = runner.step(actions)

            for m in range(M):
                a = actions[m]
                T[a][m] += 1
                R[a][m] += result.rewards[m]
                samples[a][m].append(result.rewards[m])

        return BeaconExploreState(
            T=T, R=R, samples=samples, K=K, M=M, leader_pid=leader_pid
        )

    def beacon_run_epochs(
        self,
        runner: HeterogeneousRunner,
        state_list: List[int],
        rank_list: List[int],
        explore_state: BeaconExploreState,
    ) -> List[int]:
        """
        BEACON のメインエポックループを実行する。

        horizon に達すると HorizonReachedHetero が送出される。
        その時点での最終割当は explore_state.last_assigned_arms に保存されており、
        呼び出し元が except で参照できる。

        各エポックの処理:
            1. p[k][m] = floor(log2(T[k][m])) を計算
            2. 通信フェーズ（follower→leader）:
                   p が更新された arm k について quantized delta を送信
            3. leader が mu_tilde を更新し UCB (mu_bar) を計算
            4. Oracle で最適マッチング S_r を決定
            5. 通信フェーズ（leader→follower）:
                   割当 arm s_r_m を各 follower に送信
            6. explore_state.last_assigned_arms を S_r で更新（HorizonReached 保険）
            7. 探索フェーズ: 2^{p_r} ステップ引く

        Args:
            runner: HeterogeneousRunner
            state_list: 各プレイヤーの state（通信 arm として使う, 0-based）
            rank_list: 各プレイヤーの rank（1-based）
            explore_state: BeaconExploreState（更新される）

        Returns:
            最終エポック終了時の割当 arm（0-based, 長さ M）
            ※ HorizonReachedHetero が送出される場合はここには到達しないため、
               呼び出し元は explore_state.last_assigned_arms を参照すること。

        Raises:
            HorizonReachedHetero: horizon に達した場合
        """
        K = self.K
        M = self.M
        es = explore_state
        c = state_list  # c[m] = 通信 arm（state arm）
        leader_pid = es.leader_pid

        prev_p = [[-1] * M for _ in range(K)]
        mu_tilde = [[0.0] * M for _ in range(K)]
        # prev_mu_tilde_q[k][m]: 前エポックの量子化済み整数値（delta 計算用）
        prev_mu_tilde_q = [[0] * M for _ in range(K)]

        arm_bits_needed = max(1, math.ceil(math.log2(max(2, K))))

        r = 0
        while True:
            r += 1
            runner.set_phase(f"beacon_epoch_{r}_comm")

            # 1. p[k][m] = floor(log2(T[k][m])) を計算
            curr_p = [
                [_floor_log2(es.T[k][m]) for m in range(M)] for k in range(K)
            ]

            # 2. 通信フェーズ: follower → leader
            for m in range(M):
                if rank_list[m] == 1:
                    continue  # leader はスキップ

                for k in range(K):
                    if curr_p[k][m] <= prev_p[k][m]:
                        # p が更新されていない arm はスキップ
                        continue

                    # Q = ceil(1 + p/2) ビットで量子化
                    Q = math.ceil(1.0 + curr_p[k][m] / 2.0)
                    n_use = 1 << curr_p[k][m]  # 2^p 個のサンプルを使う
                    samps = es.samples[k][m]
                    if samps:
                        mu_hat_km = (
                            sum(samps[: min(n_use, len(samps))])
                            / min(n_use, len(samps))
                        )
                    else:
                        mu_hat_km = 0.0

                    new_q = self._quantize_mean(mu_hat_km, Q)
                    delta_q = new_q - prev_mu_tilde_q[k][m]
                    prev_mu_tilde_q[k][m] = new_q

                    # 符号付き delta を Q+1 ビットで伝送
                    sign_bit = 0 if delta_q >= 0 else 1
                    abs_val = abs(delta_q)
                    bits_to_send = [sign_bit] + self._int_to_bits(abs_val, Q)

                    self._send_receive(
                        runner, bits_to_send,
                        sender=m, receiver=leader_pid,
                        state_list=c,
                    )

                    # mu_tilde[k][m] を leader 側で更新（同期シミュレーション）
                    mu_tilde[k][m] = self._dequantize_mean(prev_mu_tilde_q[k][m], Q)

            # 3. leader 自身の mu_hat を mu_tilde に反映（通信不要）
            for k in range(K):
                n_use = max(1, 1 << curr_p[k][leader_pid])
                samps = es.samples[k][leader_pid]
                if samps:
                    mu_tilde[k][leader_pid] = (
                        sum(samps[: min(n_use, len(samps))])
                        / min(n_use, len(samps))
                    )

            # 4. UCB 計算: mu_bar[k][m] = mu_tilde[k][m] + sqrt(3*ln(t_r) / 2^{p+1})
            t_r = runner.elapsed
            ln_tr = math.log(max(2, t_r))
            mu_bar = [[0.0] * M for _ in range(K)]
            for k in range(K):
                for m in range(M):
                    n_eff = max(1, 1 << (curr_p[k][m] + 1))
                    mu_bar[k][m] = mu_tilde[k][m] + math.sqrt(3.0 * ln_tr / n_eff)

            # 5. Oracle で割当決定
            assignment = matching_oracle(mu_bar, M)

            # 6. leader → follower: 割当 arm を送信
            for m in range(M):
                if rank_list[m] == 1:
                    continue
                arm_bits = self._int_to_bits(assignment[m], arm_bits_needed)
                self._send_receive(
                    runner, arm_bits,
                    sender=leader_pid, receiver=m,
                    state_list=c,
                )

            # 7. last_assigned_arms を更新（探索前に保存しておく）
            es.last_assigned_arms = list(assignment)

            # 8. 探索フェーズ: p_r = min_m p[r][s_r_m, m]; 2^{p_r} ステップ
            runner.set_phase(f"beacon_epoch_{r}_explore")
            p_r = min(curr_p[assignment[m]][m] for m in range(M))
            n_explore = max(1, 1 << p_r)

            for _ in range(n_explore):
                actions = [assignment[m] for m in range(M)]
                result = runner.step(actions)  # HorizonReachedHetero が伝播する

                for m in range(M):
                    k_a = assignment[m]
                    es.T[k_a][m] += 1
                    es.R[k_a][m] += result.rewards[m]
                    es.samples[k_a][m].append(result.rewards[m])

            prev_p = [row[:] for row in curr_p]

        # horizon に達した時点で HorizonReachedHetero が伝播するためここには到達しない
        return es.last_assigned_arms  # type: ignore[return-value]


def _floor_log2(n: int) -> int:
    """floor(log2(n)) を返す。n<=0 なら -1。"""
    if n <= 0:
        return -1
    return int(math.floor(math.log2(n)))
