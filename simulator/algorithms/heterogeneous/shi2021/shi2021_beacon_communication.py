"""
BEACON の通信サブルーチン: Send と Receive。

forced collision によるビット伝送を 1 ステップ=1 ビットで実際にシミュレートする。

通信アーム割当:
    player m の通信 arm c_m = state_list[m] (0-based)
    state は orthogonalization で一意に割り当てられた {0,...,K-2}

ビット伝送の規則:
    bit=1 を player i から player j へ送る:
        player i は arm c_j (= j の state arm) を引く → player j が c_j を引くと collision
    bit=0 を player i から player j へ送る:
        player i は arm c_i (= i の state arm) を引く → player j が c_j を引いても衝突なし
    受信側 player j は常に c_j を引き、collision なら 1、なければ 0 と解釈する。
    通信に参加しない他プレイヤーは自分の通信 arm c_m を引く（受動的に待機）。

論文アルゴリズムとの対応:
    BEACON: Leader   Line 8-9:   Receive(delta_tilde, m)
    BEACON: Follower Line 11-13: Send(delta_tilde, 1)
    BEACON: Follower Line 16:    Receive(s_r_m, 1)
    BEACON: Leader   Line 13:    Send(s_r_m, m)
    Algorithm 3 (Send), Algorithm 4 (Receive)
"""

from __future__ import annotations

import math
from typing import List

from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner


class BeaconCommunicationMixin:
    """
    BEACON の Send/Receive を同期シミュレーションとして実行する補助クラス。

    state_list[m] が通信 arm c_m として使われる。
    """

    def _send(
        self,
        runner: HeterogeneousRunner,
        bits: List[int],
        sender: int,
        receiver: int,
        state_list: List[int],
    ) -> None:
        """
        Algorithm 3 Send(): sender から receiver へビット列を送信する。

        各ビットにつき 1 ステップ消費する。
        - bit=1: sender が c_receiver を引く → collision
        - bit=0: sender が c_sender を引く  → 衝突なし
        他プレイヤーは c_m を引く（受動的に待機）。

        Args:
            runner: HeterogeneousRunner
            bits: 送信するビット列（0/1 のリスト）
            sender: 送信側プレイヤー index（0-based）
            receiver: 受信側プレイヤー index（0-based）
            state_list: 各プレイヤーの state（通信 arm として使う, 0-based）
        """
        M = self.M
        c = state_list  # c[m] = player m の通信 arm

        for bit in bits:
            actions = []
            for m in range(M):
                if m == sender:
                    # bit=1 なら受信側の arm を引き、bit=0 なら自分の arm を引く
                    actions.append(c[receiver] if bit == 1 else c[sender])
                elif m == receiver:
                    # 受信側は常に自分の arm を引いて衝突を観測
                    actions.append(c[receiver])
                else:
                    # 無関係プレイヤーは自分の通信 arm を引く（受動的に待機）
                    actions.append(c[m])

            runner.step(actions)

    def _receive(
        self,
        runner: HeterogeneousRunner,
        n_bits: int,
        receiver: int,
        sender: int,
        state_list: List[int],
    ) -> List[int]:
        """
        Algorithm 4 Receive(): receiver が sender からビット列を受信する。

        _send() と同期して呼ばれることを前提とする。
        receiver は c_receiver を引き、衝突あり → 1、衝突なし → 0 として読み取る。

        Args:
            runner: HeterogeneousRunner
            n_bits: 受信するビット数
            receiver: 受信側プレイヤー index（0-based）
            sender: 送信側プレイヤー index（0-based）
            state_list: 各プレイヤーの state（通信 arm として使う, 0-based）

        Returns:
            受信したビット列（0/1 のリスト）

        Note:
            同期シミュレーションでは _send と _receive を同時に行う。
            実際には _send が _receive を兼ねているため、このメソッドは単独では使わない。
            send_receive() を使うこと。
        """
        raise NotImplementedError("同期シミュレーションでは send_receive() を使うこと。")

    def _send_receive(
        self,
        runner: HeterogeneousRunner,
        bits: List[int],
        sender: int,
        receiver: int,
        state_list: List[int],
    ) -> List[int]:
        """
        Send と Receive を同期実行し、受信側が観測したビット列を返す。

        1 ビットにつき 1 ステップ消費する。
        他プレイヤーは c_m を引く（受動的に待機）。

        Args:
            runner: HeterogeneousRunner
            bits: 送信するビット列（0/1 のリスト）
            sender: 送信側プレイヤー index（0-based）
            receiver: 受信側プレイヤー index（0-based）
            state_list: 各プレイヤーの state（通信 arm として使う, 0-based）

        Returns:
            received_bits: receiver が観測したビット列（0/1 のリスト）
        """
        M = self.M
        c = state_list
        received_bits = []

        for bit in bits:
            actions = []
            for m in range(M):
                if m == sender:
                    actions.append(c[receiver] if bit == 1 else c[sender])
                elif m == receiver:
                    actions.append(c[receiver])
                else:
                    actions.append(c[m])

            result = runner.step(actions)
            # 受信側が衝突を観測した場合は bit=1
            received_bits.append(1 if result.collisions[receiver] else 0)

        return received_bits

    @staticmethod
    def _int_to_bits(value: int, n_bits: int) -> List[int]:
        """
        整数を n_bits ビット列（MSB first）に変換する。

        Args:
            value: 変換する整数（0 以上）
            n_bits: ビット数

        Returns:
            ビット列（0/1 のリスト）
        """
        bits = []
        for i in range(n_bits - 1, -1, -1):
            bits.append((value >> i) & 1)
        return bits

    @staticmethod
    def _bits_to_int(bits: List[int]) -> int:
        """
        ビット列（MSB first）を整数に変換する。

        Args:
            bits: ビット列（0/1 のリスト）

        Returns:
            整数
        """
        val = 0
        for b in bits:
            val = (val << 1) | b
        return val

    @staticmethod
    def _quantize_mean(mu: float, n_bits: int) -> int:
        """
        平均報酬 mu ∈ [0,1] を n_bits ビット整数に量子化する。

        ceil 量子化: ceil(mu * 2^n_bits) を n_bits ビット整数として返す。
        論文: tilde_mu = ceil(hat_mu) with ceil(1+p/2) bits

        Args:
            mu: 平均報酬 [0, 1]
            n_bits: ビット数

        Returns:
            量子化された整数値（0 以上 2^n_bits 以下）
        """
        return min(int(math.ceil(mu * (2 ** n_bits))), (1 << n_bits))

    @staticmethod
    def _dequantize_mean(quantized: int, n_bits: int) -> float:
        """
        量子化された整数を平均報酬 [0, 1] に逆変換する。

        Args:
            quantized: 量子化された整数値
            n_bits: ビット数

        Returns:
            平均報酬の近似値 [0, 1]
        """
        return quantized / (2 ** n_bits)
