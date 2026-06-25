"""heterogeneous ParallelBEACON の collision-sensing 初期化。

元の Izumi 2026 初期化は no-sensing プレイヤーを対象にしている。
ゼロ報酬が Bernoulli ノイズではなく衝突に由来することを高確率にするため、
正報酬を持つ good arm を繰り返し引く。heterogeneous collision-sensing
モデルではプレイヤーが衝突フラグを直接観測できるため、これらの反復判定は
1 回の衝突観測で置き換えられる。
"""

from __future__ import annotations

from typing import List, Tuple

from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.algorithms.homogeneous.math_helpers import checked_log as _ln
from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner


class Izumi2026CollisionSensingInitializationMixin:
    """ParallelBEACON 初期化フェーズの collision-sensing 版。"""

    def select_collision_sensing_channels(self, runner: HeterogeneousRunner) -> Tuple[List[int], dict[int, float]]:
        """
        n 本の通信チャネルを選択する。

        no-sensing 版の Izumi 2026 では、通信時に衝突と確率的なゼロ報酬を
        区別する必要があるため、まず正の報酬下界を持つ腕を探す。
        collision-sensing 設定では任意の腕で衝突ビットを伝送できるため、
        このフェーズにサンプリングコストはかからない。
        """
        runner.set_phase("select_collision_sensing_channels")
        channels = list(range(self.n))
        return channels, {k: 1.0 for k in channels}

    def parallel_virtual_musical_chairs_collision_sensing(
        self,
        runner: HeterogeneousRunner,
        good_arms: List[int],
    ) -> List[int]:
        """
        直接観測した衝突フラグを使う ParallelVirtualMusicalChairs。

        no-sensing 版では、サンプルした good arm が衝突していないことを
        正報酬の観測から推定する。ここでは、プレイヤーがチャネル腕を引いて
        衝突なしを観測した時点で external rank を確定する。
        """
        runner.set_phase("parallel_virtual_musical_chairs_collision_sensing")
        K = self.K
        M = self.M
        n = len(good_arms)
        delta = self.delta

        s_list = [-1] * M
        slots = [[-1] * n for _ in range(M)]

        good_set = set(good_arms)
        non_good_arms = [a for a in range(K) if a not in good_set]
        dummy = non_good_arms[0] if non_good_arms else good_arms[0]

        # no-sensing 版: total_steps = ceil(K * tau_rank / n)、tau_rank = ceil(K * ln(1/delta) / mu_min)
        # collision-sensing 版: mu_min=1 として tau_rank_cs = ceil(ln(1/delta)) と置き換える。
        # total_steps = ceil(K * tau_rank_cs / n) → ブロック数 ≈ tau_rank_cs / n
        # 1 ブロックで n チャネル並列 × 1 試行 = n 回のスロット試行が行われるため、
        # 合計スロット試行 = (tau_rank_cs / n ブロック) × n = tau_rank_cs 回となる。
        # 複数プレイヤーが同じ virtual slot を選んで衝突した場合、次のブロックで別スロットを
        # 再試行する機会が必要。1 ブロックだけでは未確定プレイヤーが残り、0 へのフォールバックで
        # rank 重複・M_hat 推定誤りが生じる。
        tau_rank_cs = max(1, _ceil(_ln(1.0 / delta)))
        total_steps = max(1, _ceil(K * tau_rank_cs / max(1, n)))

        for t0 in range(total_steps):
            if t0 % K == 0:
                for m in range(M):
                    if s_list[m] == -1:
                        forbidden_set = set()
                        for i0 in range(n):
                            available = [slot for slot in range(K) if slot not in forbidden_set]
                            if not available:
                                available = list(range(K))
                            slots[m][i0] = self._player_rngs[m].choice(available)
                            forbidden_set = {
                                (slots[m][j0] + (i0 + 1 - j0)) % K
                                for j0 in range(i0 + 1)
                            }
                    else:
                        for i0 in range(n):
                            slots[m][i0] = s_list[m]

            actions: List[int] = []
            pulled_i0_per_player = [-1] * M
            for m in range(M):
                pulled = False
                for i0 in range(n):
                    if slots[m][i0] >= 0 and (t0 + i0) % K == slots[m][i0]:
                        actions.append(good_arms[i0])
                        pulled_i0_per_player[m] = i0
                        pulled = True
                        break
                if not pulled:
                    actions.append(dummy)

            result = runner.step(actions)
            for m in range(M):
                i0 = pulled_i0_per_player[m]
                if i0 >= 0 and s_list[m] == -1 and not result.collisions[m]:
                    s_list[m] = slots[m][i0]
                    for j0 in range(n):
                        slots[m][j0] = s_list[m]

        return s_list

    def parallel_virtual_number_players_collision_sensing(
        self,
        runner: HeterogeneousRunner,
        good_arms: List[int],
        s_list: List[int],
    ) -> Tuple[List[int], List[int]]:
        """
        直接観測した衝突フラグを使う ParallelVirtualNumberPlayers。

        no-sensing 版では、各仮想時刻のテストを tau 回繰り返し、
        報酬が全て 0 のブロックを衝突イベントとして扱う。
        ここでは各テストを 1 回だけ実行し、観測した衝突フラグを直接使う。
        """
        runner.set_phase("parallel_virtual_number_players_collision_sensing")
        K = self.K
        M = self.M
        n = len(good_arms)

        M_hat = [1] * M
        j = [1] * M

        good_set = set(good_arms)
        non_good_arms = [a for a in range(K) if a not in good_set]
        dummy = non_good_arms[0] if non_good_arms else good_arms[0]

        H = _ceil(2 * K / n)
        for h_0 in range(H):
            ell = [[-1] * n for _ in range(M)]
            for m in range(M):
                s_1based = s_list[m] + 1
                for i0 in range(n):
                    v = h_0 * n + i0 + 1
                    if v > 2 * K:
                        continue
                    if v > 2 * s_1based:
                        ell[m][i0] = (v - s_1based - 1) % K
                    else:
                        ell[m][i0] = s_list[m]

            for k_0 in range(K):
                actions: List[int] = []
                matched_i0_per_player = [-1] * M
                for m in range(M):
                    found = False
                    for i0 in range(n):
                        if ell[m][i0] != -1 and (ell[m][i0] + i0) % K == k_0:
                            matched_i0_per_player[m] = i0
                            found = True
                            break
                    actions.append(good_arms[matched_i0_per_player[m]] if found else dummy)

                result = runner.step(actions)
                for m in range(M):
                    i0 = matched_i0_per_player[m]
                    if i0 >= 0 and result.collisions[m]:
                        M_hat[m] += 1
                        v_collision = h_0 * n + i0 + 1
                        s_1based = s_list[m] + 1
                        if v_collision <= 2 * s_1based:
                            j[m] += 1

        return M_hat, j
