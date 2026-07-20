"""ParallelBEACON のメインエポックループ。

疑似コード: docs/pseudocode/20260626_izumi2026_heterogeneous_pseudocode.md
Algorithm 5a〜5c に対応する実装。

通信コスト計算方針（簡略版シミュレーション）:
  実際の forced-collision ビット伝送は行わず、通信コストのみを consume_dummy_steps
  で課金する。推定値は samples から直接計算する。

  Uplink Phase A（n グループ並列）:
    各グループの通信コスト = フォロワーごとに p が増加した arm の (Q+1) ビットの合計
    全体コスト = max over groups
  Uplink Phase B（sub-leader → grand leader、逐次）:
    コスト = グループ 2..n について、そのグループ全員の changed arms の合計ビット数
  Downlink Phase A（grand leader → sub-leaders、逐次）:
    コスト = グループ 2..n について、グループサイズ × arm_bits_needed の合計
  Downlink Phase B（sub-leaders / grand leader → フォロワー、並列）:
    コスト = max over groups of (n_followers_in_group * arm_bits_needed)

差分エンコード:
  prev_p[k][m] を保持し、curr_p[k][m] > prev_p[k][m] のときのみ課金する。
  量子化ビット数 Q_{k,m} = ceil(1 + p[k,m] / 2)。
  1 arm あたりの送信コスト = Q_{k,m} + 1 ビット（符号ビット + 差分 magnitude）。
"""

from __future__ import annotations

import math
from typing import Dict, List

from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.algorithms.heterogeneous.izumi2026.algorithm4_parallel_beacon_initial_sampling import (
    ParallelBeaconExploreState,
)
from simulator.algorithms.heterogeneous.izumi2026.helpers import (
    consume_comm_steps,
    floor_log2,
)
from simulator.algorithms.heterogeneous.shi2021.oracle import matching_oracle
from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner


def _q_bits(p_val: int) -> int:
    """量子化ビット数 Q = ceil(1 + p/2) を返す。p < 0 なら 1。"""
    if p_val < 0:
        return 1
    return max(1, _ceil(1.0 + p_val / 2.0))


class ParallelBeaconEpochMixin:
    """ParallelBEACON のエポックループを実行する補助クラス。"""

    def parallel_beacon_run_epochs(
        self,
        runner: HeterogeneousRunner,
        state: ParallelBeaconExploreState,
    ) -> List[int]:
        """
        ParallelBEACON のメインエポックループを実行する。

        通信フェーズ（comm）と探索フェーズ（explore）を繰り返す。
        grand leader が全プレイヤーの個人推定値から最適割当を計算し、
        階層的通信で全フォロワーに割当を配布する。

        Args:
            runner: HeterogeneousRunner
            state: ParallelBeaconExploreState（初期サンプリング後の状態）

        Returns:
            最終割当 arm（0-based, 長さ M）
        """
        K = state.K
        M = state.M
        n = state.n
        grand_leader = state.grand_leader_pid
        group_map = state.group_map  # group_map[g] = player index リスト (1-based g)

        # assignment を送るのに必要なビット数: ceil(log2(K))
        arm_bits_needed = max(1, math.ceil(math.log2(max(2, K))))

        # 差分エンコードのため前エポックの p[k][m] を保持する（初期値 -1）
        # prev_p[k][m] = -1: 前エポックで p が未計算（初回 or サンプル数 0）
        prev_p: List[List[int]] = [[-1] * M for _ in range(K)]

        r = 0
        while True:
            r += 1
            runner.set_phase(f"parallel_beacon_epoch_{r}_comm")

            # 1. p[k][m] = floor(log2(T[k][m])) を計算する
            curr_p: List[List[int]] = [
                [floor_log2(state.T[k][m]) for m in range(M)] for k in range(K)
            ]

            # ================================================================
            # Uplink Phase A（parallel）: フォロワー → グループ leader
            # n グループが n チャネルで同時に通信する。
            # 所要ステップ数はグループごとに計算し、全グループの max を課金する。
            # ================================================================
            # 各グループのフォロワー一覧を事前に準備する
            # leader_of_g: group g の leader の player index
            leader_of_g: Dict[int, int] = {}
            for g in range(1, n + 1):
                if g == 1:
                    leader_of_g[g] = grand_leader
                else:
                    leader_of_g[g] = state.subleader_pids.get(g, -1)

            phase_a_group_bits: List[int] = []
            for g in range(1, n + 1):
                group_pids = group_map.get(g, [])
                lg = leader_of_g[g]
                # フォロワー = グループ leader 以外のメンバー
                followers_g = [m for m in group_pids if m != lg]
                bits_g = 0
                for f in followers_g:
                    for k in range(K):
                        if curr_p[k][f] > prev_p[k][f]:
                            # 差分エンコード: 符号ビット + Q ビット magnitude
                            Q = _q_bits(curr_p[k][f])
                            bits_g += Q + 1
                phase_a_group_bits.append(bits_g)

            # n グループが並列なので max が実際の所要ステップ数
            comm_steps_phase_a = max(phase_a_group_bits) if phase_a_group_bits else 0
            consume_comm_steps(runner, comm_steps_phase_a, state.state_arms)

            # ================================================================
            # Uplink Phase B（sequential）: sub-leader → grand leader
            # グループ 2..n の sub-leader が、グループ内全プレイヤーの
            # 個人推定値を grand leader へ逐次中継する。
            # ================================================================
            comm_steps_phase_b = 0
            for g in range(2, n + 1):
                group_pids = group_map.get(g, [])
                for m in group_pids:
                    for k in range(K):
                        if curr_p[k][m] > prev_p[k][m]:
                            # sub-leader が中継するのは自分のデータも含む
                            Q = _q_bits(curr_p[k][m])
                            comm_steps_phase_b += Q + 1

            consume_comm_steps(runner, comm_steps_phase_b, state.state_arms)

            # ================================================================
            # Grand leader が全プレイヤーの個人推定値 mu_tilde[k][m] を更新する。
            # シミュレーションでは samples から直接計算する（通信コストは上記で課金済み）。
            # mu_hat[k][m] = 最初の 2^{p[k][m]} サンプルの平均。
            # ================================================================
            mu_tilde = [[0.0] * M for _ in range(K)]
            for k in range(K):
                for m in range(M):
                    p_km = curr_p[k][m]
                    if p_km < 0:
                        # サンプルなし
                        mu_tilde[k][m] = 0.0
                        continue
                    n_use = 1 << p_km  # 2^{p_km} サンプルを使う
                    samps = state.samples[k][m]
                    if samps:
                        mu_tilde[k][m] = sum(samps[:n_use]) / min(n_use, len(samps))
                    else:
                        mu_tilde[k][m] = 0.0

            # ================================================================
            # UCB インデックス計算
            # mu_bar[k][m] = mu_tilde[k][m] + sqrt(3 * ln(t_r) / 2^{p[k][m]+1})
            # ================================================================
            t_r = runner.elapsed
            ln_tr = math.log(max(2, t_r))
            mu_bar = [[0.0] * M for _ in range(K)]
            for k in range(K):
                for m in range(M):
                    p_km = curr_p[k][m] if curr_p[k][m] >= 0 else 0
                    # n_eff = 2^{p_km+1}: UCB 探索ボーナスの分母
                    n_eff = max(1, 1 << (p_km + 1))
                    mu_bar[k][m] = mu_tilde[k][m] + math.sqrt(3.0 * ln_tr / n_eff)

            # Matching oracle: mu_bar を最大化する player-arm 割当を求める
            assignment = matching_oracle(mu_bar, M)

            # ================================================================
            # Downlink Phase A（sequential）: grand leader → sub-leaders
            # grand leader が各 sub-leader のグループ全員分の割当を逐次送信する。
            # ================================================================
            comm_steps_down_a = 0
            for g in range(2, n + 1):
                group_size_g = len(group_map.get(g, []))
                comm_steps_down_a += group_size_g * arm_bits_needed

            # ================================================================
            # Downlink Phase B（parallel）: sub-leaders / grand leader → フォロワー
            # 全グループが同時にフォロワーへ割当を送信する。
            # 所要ステップ数は各グループのフォロワー数の max。
            # ================================================================
            max_followers_per_group = 0
            for g in range(1, n + 1):
                group_pids = group_map.get(g, [])
                lg = leader_of_g[g]
                n_followers_g = sum(1 for m in group_pids if m != lg)
                max_followers_per_group = max(max_followers_per_group, n_followers_g)

            comm_steps_down_b = max_followers_per_group * arm_bits_needed

            consume_comm_steps(runner, comm_steps_down_a + comm_steps_down_b, state.state_arms)

            state.last_assigned_arms = list(assignment)

            # ================================================================
            # Exploration phase: 全プレイヤーが割当 arm を 2^{p_r} ステップ引く
            # p_r = min over m of p[assignment[m]][m]
            # ================================================================
            runner.set_phase(f"parallel_beacon_epoch_{r}_explore")
            p_r = min(
                (curr_p[assignment[m]][m] if curr_p[assignment[m]][m] >= 0 else 0)
                for m in range(M)
            )
            n_explore = max(1, 1 << p_r)

            for _ in range(n_explore):
                actions = [assignment[m] for m in range(M)]
                result = runner.step(actions)
                for m in range(M):
                    k_a = assignment[m]
                    state.T[k_a][m] += 1
                    state.R[k_a][m] += result.rewards[m]
                    state.samples[k_a][m].append(result.rewards[m])

            # 次エポックの差分エンコードのため、今エポック開始時の p を記録する
            prev_p = [row[:] for row in curr_p]

        return state.last_assigned_arms  # type: ignore[return-value]
