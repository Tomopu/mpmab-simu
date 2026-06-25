"""
Wang 2020 Orthogonalization と Rank Assignment。

BEACON の初期化フェーズとして使う。
collision-sensing 設定を利用してプレイヤーごとに一意な state を割り当て、
そこから rank と M_hat を推定する。

論文との対応:
- Wang et al. (2020) の orthogonalization procedure を Shi et al. (2021) BEACON に適用。
- state (0-based): {0, ..., K-2} ← 論文の 1-based {1, ..., K-1} に相当
- broadcast arm (0-based): K-1  ← 論文の 1-based arm K に相当
- 各ブロックは K+1 ラウンド（選択ラウンド 1 + broadcast ラウンド K）
"""

from __future__ import annotations

import math
from typing import List, Tuple

from simulator.algorithms.heterogeneous.shi2021.runner_hetero import (
    HeterogeneousRunner,
    HorizonReachedHetero,
)


class Wang2020OrthogonalizationMixin:
    """
    Wang et al. (2020) の orthogonalization procedure と rank assignment。

    BEACON の初期化で使うため HeterogeneousShiBeacon2021 に継承させる。
    """

    def orthogonalization(
        self,
        runner: HeterogeneousRunner,
        max_blocks: int = 0,
    ) -> List[int]:
        """
        Orthogonalization フェーズを M 人同期実行する。

        各プレイヤーが衝突なしで arm を確保できるまで繰り返すことで、
        一意な外部 state を割り当てる。

        論文の変数対応:
            state (0-based) {0,...,K-2} ← 論文 1-based {1,...,K-1}
            broadcast arm (0-based) K-1 ← 論文 1-based arm K
            ブロックは「選択ラウンド 1 + broadcast ラウンド K」= K+1 ラウンド

        各ブロックの構造（K+1 ラウンド）:
            ラウンド 0 (選択ラウンド):
                未確定プレイヤー → ランダムに state 候補を選んで引く
                確定済みプレイヤー → 自分の state arm を引く
                衝突がなければ候補を state として確定
            ラウンド 1..K (broadcast ラウンド, q=0..K-1, 0-based):
                未確定プレイヤー → broadcast arm (K-1) を引く
                確定済みプレイヤー → q == state なら broadcast arm を引き、それ以外は自分の state arm
                全 K ラウンドで衝突がなければ全員確定 → 終了

        Args:
            runner: HeterogeneousRunner
            max_blocks: 最大ブロック数。0 なら制限なし（horizon に委ねる）。

        Returns:
            state_list: 各プレイヤーの state（0-based, 未確定は -1）

        Raises:
            HorizonReachedHetero: horizon に達した場合
        """
        runner.set_phase("orthogonalization")
        K = self.K
        M = self.M

        # 0-based state: {0,...,K-2}、broadcast arm (0-based) = K-1
        broadcast_arm = K - 1
        num_states = K - 1  # 使える state の数 = K-1

        state_list = [-1] * M  # -1 = 未確定

        block = 0
        while True:
            block += 1
            if max_blocks > 0 and block > max_blocks:
                break

            # ---- 選択ラウンド ----
            actions = []
            candidates = []
            for m in range(M):
                if state_list[m] == -1:
                    # 未確定: {0,...,K-2} からランダムに候補を選ぶ
                    c = self._player_rngs[m].randrange(num_states)
                    candidates.append(c)
                    actions.append(c)
                else:
                    candidates.append(state_list[m])
                    actions.append(state_list[m])

            result = runner.step(actions)

            # 衝突がなければ state を確定
            for m in range(M):
                if state_list[m] == -1 and not result.collisions[m]:
                    state_list[m] = candidates[m]

            # ---- broadcast ラウンド (q=0..K-1, 0-based) ----
            any_collision_in_broadcast = False
            for q in range(K):
                actions = []
                for m in range(M):
                    if state_list[m] == -1:
                        # 未確定 → broadcast arm を引く
                        actions.append(broadcast_arm)
                    elif q == state_list[m]:
                        # 自分のスロット → broadcast arm を引く
                        actions.append(broadcast_arm)
                    else:
                        # 他のスロット → 自分の state arm を引く
                        actions.append(state_list[m])

                result = runner.step(actions)

                if any(result.collisions):
                    any_collision_in_broadcast = True

            # broadcast ラウンドで衝突がなければ全員確定 → 終了
            if not any_collision_in_broadcast:
                break

        return state_list

    def rank_assignment(
        self,
        runner: HeterogeneousRunner,
        state_list: List[int],
    ) -> Tuple[List[int], List[int]]:
        """
        Rank Assignment フェーズを M 人同期実行する。

        orthogonalization で得た state から内部 rank と M_hat を推定する。

        論文の変数対応:
            state (0-based) {0,...,K-2}
            ブロック k (0-based) {0,...,K-2} ← 論文 1-based {1,...,K-1}
            round q (0-based) {0,...,K-2}

        各ブロック k の構造（K-1 ラウンド）:
            for q in 0..K-2 (0-based):
                state==k のプレイヤー → arm q を引く（「自分の存在を各 q に通知」）
                他のプレイヤー → 自分の state arm を引く

            collision_seen_in_block == True なら state k は存在する
                → M_hat += 1
                → k < state なら rank += 1

        Args:
            runner: HeterogeneousRunner
            state_list: 各プレイヤーの state（0-based, 未確定は -1）

        Returns:
            (M_hat_list, rank_list):
                M_hat_list: 各プレイヤーの推定プレイヤー数（1 以上）
                rank_list: 各プレイヤーの内部 rank（1-based, leader = 1）

        Raises:
            HorizonReachedHetero: horizon に達した場合
        """
        runner.set_phase("rank_assignment")
        K = self.K
        M = self.M

        M_hat = [1] * M  # 自分自身の分から始める
        rank = [1] * M   # 1-based（論文と同じ）

        # ブロック k (0-based): 0..K-2
        for k in range(K - 1):
            # ラウンド q (0-based): 0..K-2
            collision_seen_by_others = [False] * M

            for q in range(K - 1):
                actions = []
                for m in range(M):
                    if state_list[m] == k:
                        # state=k のプレイヤーは arm q を引いてアナウンス
                        actions.append(q)
                    else:
                        # 他プレイヤーは自分の state arm を引く
                        # state==-1 (未確定) の場合は broadcast arm (K-1) を引く
                        s = state_list[m] if state_list[m] != -1 else K - 1
                        actions.append(s)

                result = runner.step(actions)

                # state==k のプレイヤーが arm q を引いて衝突があった
                # → arm q を持つ別プレイヤー（state==q）は衝突を観測
                for m in range(M):
                    if state_list[m] != k and result.collisions[m]:
                        collision_seen_by_others[m] = True

            # ブロック k で衝突を観測した → state k が存在する
            # state==k のプレイヤーは他との衝突を観測
            # （ブロック k で state k が自分のアナウンスで衝突を見たかどうか）
            collision_from_k = [False] * M
            # ブロック k で state==k のプレイヤーが衝突を観測した場合も M_hat に含める
            # 簡略化: state k のプレイヤーが arm q を引いて衝突 → state q も存在
            # ここではブロックごとに collision_seen_by_others を使って M_hat を更新
            # state==k のプレイヤー自身には「ブロック k での衝突 = 他 state k との衝突」
            # orthogonalization 後は一意なので state==k は最大 1 人

            # M_hat と rank の更新
            # ブロック k で衝突を観測した全プレイヤーに「state k は存在する」と通知
            for m in range(M):
                if collision_seen_by_others[m]:
                    M_hat[m] += 1
                    # state k < state_list[m] なら rank を +1
                    if state_list[m] != -1 and k < state_list[m]:
                        rank[m] += 1

        return M_hat, rank
