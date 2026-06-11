"""
ParallelBEACON エポック処理。

Shi et al. (2021) BEACON を Izumi et al. (2026) のマルチチャネル設定に拡張する。
n 本の good arm を通信チャネルとして使い、n グループ（grand leader / sub-leader / follower）
の階層通信で全プレイヤーの統計を集約し、最適マッチング Oracle で割当を決定する。

グループ構造:
    grand leader : rank j=1
    sub-leader g : rank j=g (2 <= g <= n)
    follower j>n : グループ g = ((j-1) mod n) + 1 に所属

通信の流れ (各エポック):
    アップリンク:
        follower → sub-leader (チャネル G[g-1])
        sub-leader → grand leader (チャネル G[0])
    ダウンリンク:
        grand leader → sub-leader (チャネル G[0])
        sub-leader → follower (チャネル G[g-1])

簡略実装の方針:
    - 統計値は直接集約する（forced-collision bit 伝送を再現しない）
    - 通信コストは runner.step() のダミーアクションで消費する
    - グループ並列通信はグループ最大サイズで時間をまとめる
    - HorizonReachedHetero は呼び出し元に伝播させる

論文との対応:
    papers/README.md: Model 4 学習フェーズ (ParallelBEACON: Leader / Follower m)
    ComGrandLeader / ComSubLeader / ComFollower (Izumi 2026 補完擬似コード)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner
from simulator.algorithms.heterogeneous.shi2021.oracle import matching_oracle
from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil


class ParallelBeaconExploreState:
    """
    ParallelBEACON エポックループで共有される探索状態。

    Attributes:
        T[k][m]: player m による arm k の exploratory サンプル数
        R[k][m]: player m による arm k の累積報酬
        samples[k][m]: player m による arm k の個別サンプル列
        K, M, n: arm 数・プレイヤー数・good arm 数
        good_arms: 通信チャネルとして使う arm の 0-based index リスト（長さ n）
        group_map: group_map[g] = そのグループに所属するプレイヤー index リスト (1-based g)
        last_assigned_arms: 最終エポックの Oracle 割当（HorizonReached 保険）
    """

    def __init__(
        self,
        T: List[List[int]],
        R: List[List[float]],
        samples: List[List[List[float]]],
        K: int,
        M: int,
        n: int,
        good_arms: List[int],
        rank_list: List[int],
    ) -> None:
        self.T = T
        self.R = R
        self.samples = samples
        self.K = K
        self.M = M
        self.n = n
        self.good_arms = good_arms  # G = [k_tilde_1, ..., k_tilde_n] (0-based)
        self.rank_list = rank_list

        # グループマッピング: group_map[g] = list of player indices (1-based g=1..n)
        self.group_map: Dict[int, List[int]] = _build_group_map(rank_list, n)

        # grand leader は rank==1 のプレイヤー
        self.grand_leader_pid: int = next(
            (m for m in range(M) if rank_list[m] == 1), 0
        )
        # sub-leader g は rank==g (2..n) のプレイヤー
        self.subleader_pids: Dict[int, int] = {
            g: next((m for m in range(M) if rank_list[m] == g), -1)
            for g in range(2, n + 1)
        }

        self.last_assigned_arms: List[int] = list(range(M))


class ParallelBeaconEpochMixin:
    """
    ParallelBEACON の初期サンプリングとエポックループを実行する mixin。
    """

    def parallel_beacon_initial_sampling(
        self,
        runner: HeterogeneousRunner,
        good_arms: List[int],
        rank_list: List[int],
    ) -> ParallelBeaconExploreState:
        """
        ParallelBEACON の初期サンプリングフェーズ（K ステップ）。

        BEACON の初期サンプリングと同様、各プレイヤーが全 arm を一通り探索する。
        Grand leader は arm k=0..K-1 を順に、follower m (rank r, 1-based) は
        arm (r-1+k) mod K を引く。

        Args:
            runner: HeterogeneousRunner
            good_arms: FindMultipleGoodArms の出力 G（0-based, 長さ n）
            rank_list: 各プレイヤーの rank（1-based）

        Returns:
            ParallelBeaconExploreState
        """
        runner.set_phase("parallel_beacon_initial_sampling")
        K = self.K
        M = self.M
        n = len(good_arms)

        T = [[0] * M for _ in range(K)]
        R = [[0.0] * M for _ in range(K)]
        samples: List[List[List[float]]] = [[[] for _ in range(M)] for _ in range(K)]

        for k in range(K):
            actions = []
            for m in range(M):
                # rank 1 が grand leader、その他は (rank-1+k) mod K
                actions.append((rank_list[m] - 1 + k) % K)
            result = runner.step(actions)
            for m in range(M):
                a = actions[m]
                T[a][m] += 1
                R[a][m] += result.rewards[m]
                samples[a][m].append(result.rewards[m])

        return ParallelBeaconExploreState(
            T=T, R=R, samples=samples,
            K=K, M=M, n=n,
            good_arms=good_arms,
            rank_list=rank_list,
        )

    def parallel_beacon_run_epochs(
        self,
        runner: HeterogeneousRunner,
        state: ParallelBeaconExploreState,
    ) -> List[int]:
        """
        ParallelBEACON のメインエポックループを実行する。

        各エポック:
            1. p[k][m] = floor(log2(T[k][m])) を計算
            2. アップリンク通信:
                follower → sub-leader （チャネル G[g-1], 簡略集約）
                sub-leader → grand leader（チャネル G[0], 簡略集約）
            3. grand leader が UCB を計算し Oracle で割当決定
            4. ダウンリンク通信:
                grand leader → sub-leader → follower（簡略通知）
            5. state.last_assigned_arms を更新
            6. 探索フェーズ: 2^{p_r} ステップ

        Args:
            runner: HeterogeneousRunner
            state: ParallelBeaconExploreState

        Returns:
            最終割当 arm（0-based, 長さ M）
            ※ HorizonReachedHetero が送出される場合は state.last_assigned_arms を参照

        Raises:
            HorizonReachedHetero: horizon に達した場合
        """
        K = state.K
        M = state.M
        n = state.n
        delta = self.delta
        good_arms = state.good_arms  # 通信チャネル G (0-based)
        rank_list = state.rank_list
        grand_leader = state.grand_leader_pid

        # good_set: 通信チャネルに使う arm 集合（探索では sequential hopping に含む）
        # ダミー腕: good_arms 以外の arm（通信コスト消費用）
        good_set = set(good_arms)
        non_good = [a for a in range(K) if a not in good_set]
        dummy = non_good[0] if non_good else good_arms[0]

        prev_p = [[-1] * M for _ in range(K)]
        # mu_tilde[k][m]: grand leader が保持する量子化済み平均報酬
        mu_tilde = [[0.0] * M for _ in range(K)]

        arm_bits_needed = max(1, math.ceil(math.log2(max(2, K))))

        r = 0
        while True:
            r += 1
            runner.set_phase(f"parallel_beacon_epoch_{r}_comm")

            # 1. p[k][m] を計算
            curr_p = [
                [_floor_log2(state.T[k][m]) for m in range(M)] for k in range(K)
            ]

            # 2. アップリンク: 各グループの統計を集約
            # group_stats[g][k] = (weighted_sum, total_N) のグループ集約値
            group_stats: Dict[int, Dict[int, Tuple[float, int]]] = {}

            for g in range(1, n + 1):
                group_pids = state.group_map.get(g, [])
                group_stats[g] = {}
                for k in range(K):
                    wsum = 0.0
                    tn = 0
                    for m in group_pids:
                        n_use = max(1, 1 << curr_p[k][m]) if curr_p[k][m] >= 0 else 1
                        samps = state.samples[k][m]
                        mu_hat_km = (
                            sum(samps[: min(n_use, len(samps))]) / min(n_use, len(samps))
                            if samps else 0.0
                        )
                        wsum += mu_hat_km * state.T[k][m]
                        tn += state.T[k][m]
                    group_stats[g][k] = (wsum, tn)

                # グループ内通信コスト: 最大 ceil(p/2+3) ビット x tau x Ka (Huang 2022 と同じ式)
                # 簡略化: p の平均から代表 Q を計算し、グループ人数分の通信コストを消費
                Ka = K  # active arms は初期フェーズでは K 全体
                _p_vals_g = [
                    curr_p[kk][mm]
                    for mm in group_pids for kk in range(K)
                    if curr_p[kk][mm] >= 0
                ]
                Q = _ceil(max(0, max(_p_vals_g) if _p_vals_g else 0) / 2.0 + 3)

                # follower → sub-leader の通信コスト（グループ g の follower 数 × Q × Ka）
                n_followers_g = max(0, len(group_pids) - 1)  # sub-leader 除く
                comm_steps_up = n_followers_g * Q * Ka
                _consume_dummy_steps(runner, comm_steps_up, dummy, M)

            # sub-leader → grand leader の通信コスト（sub-leader 数 × Q × Ka）
            n_subleaders = max(0, n - 1)
            _p_vals_gl = [curr_p[kk][grand_leader] for kk in range(K) if curr_p[kk][grand_leader] >= 0]
            Q_gl = _ceil(max(0, max(_p_vals_gl) if _p_vals_gl else 0) / 2.0 + 3)
            comm_steps_gl = n_subleaders * Q_gl * K
            _consume_dummy_steps(runner, comm_steps_gl, dummy, M)

            # 3. grand leader: グループ統計を統合し UCB を計算
            # mu_tilde[k][m]: grand leader から見た全 m の推定値
            for k in range(K):
                for m in range(M):
                    g = _player_group(rank_list[m], n)
                    gs_k = group_stats.get(g, {}).get(k, (0.0, 0))
                    gs_wsum, gs_n = gs_k
                    if gs_n > 0 and g == _player_group(rank_list[grand_leader], n):
                        # grand leader 自身のグループは個別値を使う
                        n_use = max(1, 1 << curr_p[k][m]) if curr_p[k][m] >= 0 else 1
                        samps = state.samples[k][m]
                        mu_tilde[k][m] = (
                            sum(samps[: min(n_use, len(samps))]) / min(n_use, len(samps))
                            if samps else 0.0
                        )
                    elif gs_n > 0:
                        # 他グループは集約値を使う（全メンバーの平均として扱う）
                        mu_tilde[k][m] = gs_wsum / gs_n
                    else:
                        mu_tilde[k][m] = 0.0

            t_r = runner.elapsed
            ln_tr = math.log(max(2, t_r))
            mu_bar = [[0.0] * M for _ in range(K)]
            for k in range(K):
                for m in range(M):
                    p_km = curr_p[k][m] if curr_p[k][m] >= 0 else 0
                    n_eff = max(1, 1 << (p_km + 1))
                    mu_bar[k][m] = mu_tilde[k][m] + math.sqrt(3.0 * ln_tr / n_eff)

            # Oracle で割当決定
            assignment = matching_oracle(mu_bar, M)

            # 4. ダウンリンク通信コスト（grand leader → sub-leader → follower）
            comm_steps_down = (n_subleaders + n_followers_g) * arm_bits_needed
            _consume_dummy_steps(runner, comm_steps_down, dummy, M)

            # 5. last_assigned_arms を更新（探索前に保存）
            state.last_assigned_arms = list(assignment)

            # 6. 探索フェーズ: p_r = min_m p[r][s_r_m, m]; 2^{p_r} ステップ
            runner.set_phase(f"parallel_beacon_epoch_{r}_explore")
            p_r = min(
                (curr_p[assignment[m]][m] if curr_p[assignment[m]][m] >= 0 else 0)
                for m in range(M)
            )
            n_explore = max(1, 1 << p_r)

            for _ in range(n_explore):
                actions = [assignment[m] for m in range(M)]
                result = runner.step(actions)  # HorizonReachedHetero が伝播する
                for m in range(M):
                    k_a = assignment[m]
                    state.T[k_a][m] += 1
                    state.R[k_a][m] += result.rewards[m]
                    state.samples[k_a][m].append(result.rewards[m])

            prev_p = [row[:] for row in curr_p]

        # HorizonReachedHetero が来てここには到達しない
        return state.last_assigned_arms  # type: ignore[return-value]


# ---- ヘルパー関数 ----

def _build_group_map(rank_list: List[int], n: int) -> Dict[int, List[int]]:
    """
    rank_list から group_map[g] = list of player indices を構築する。

    グループ割当:
        j=1: grand leader → group 1
        j=2..n: sub-leader → group j
        j>n: follower → group ((j-1) mod n) + 1

    Args:
        rank_list: 各プレイヤーの rank（1-based）
        n: good arm の数（グループ数と一致）

    Returns:
        group_map: {g: [player_index, ...]} (g は 1-based)
    """
    group_map: Dict[int, List[int]] = {g: [] for g in range(1, n + 1)}
    for m, j in enumerate(rank_list):
        g = _player_group(j, n)
        group_map[g].append(m)
    return group_map


def _player_group(rank: int, n: int) -> int:
    """
    player の rank から所属グループ (1-based) を返す。

    j=1..n → group j (grand leader と sub-leader は自身のグループ番号)
    j>n   → group ((j-1) mod n) + 1

    Args:
        rank: 1-based rank
        n: good arm の数

    Returns:
        1-based group id
    """
    if rank <= n:
        return rank
    return ((rank - 1) % n) + 1


def _consume_dummy_steps(
    runner: HeterogeneousRunner,
    n_steps: int,
    dummy_arm: int,
    M: int,
) -> None:
    """
    通信コストとして n_steps ステップを消費する。

    全プレイヤーが dummy_arm を選ぶダミーアクションで runner.step() を呼ぶ。
    n_steps が 0 以下の場合は何もしない。

    Args:
        runner: HeterogeneousRunner
        n_steps: 消費するステップ数
        dummy_arm: 全プレイヤーが選ぶダミー腕（0-based）
        M: プレイヤー数
    """
    if n_steps <= 0:
        return
    actions = [dummy_arm] * M
    for _ in range(n_steps):
        runner.step(actions)


def _floor_log2(n: int) -> int:
    """floor(log2(n)) を返す。n<=0 なら -1。"""
    if n <= 0:
        return -1
    return int(math.floor(math.log2(n)))
