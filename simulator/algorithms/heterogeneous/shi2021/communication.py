"""BEACON の通信サブルーチン。"""

from __future__ import annotations

import math
from typing import List

from simulator.algorithms.heterogeneous.shi2021.runner_hetero import HeterogeneousRunner


class BeaconCommunicationMixin:
    """BEACON の Send/Receive を同期シミュレーションとして実行する補助クラス。"""

    def _send(
        self,
        runner: HeterogeneousRunner,
        bits: List[int],
        sender: int,
        receiver: int,
        state_list: List[int],
    ) -> None:
        c = state_list

        for bit in bits:
            actions = []
            for m in range(self.M):
                if m == sender:
                    actions.append(c[receiver] if bit == 1 else c[sender])
                elif m == receiver:
                    actions.append(c[receiver])
                else:
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
        raise NotImplementedError("同期シミュレーションでは send_receive() を使うこと。")

    def _send_receive(
        self,
        runner: HeterogeneousRunner,
        bits: List[int],
        sender: int,
        receiver: int,
        state_list: List[int],
    ) -> List[int]:
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
            received_bits.append(1 if result.collisions[receiver] else 0)

        return received_bits

    @staticmethod
    def _int_to_bits(value: int, n_bits: int) -> List[int]:
        bits = []
        for i in range(n_bits - 1, -1, -1):
            bits.append((value >> i) & 1)
        return bits

    @staticmethod
    def _bits_to_int(bits: List[int]) -> int:
        val = 0
        for b in bits:
            val = (val << 1) | b
        return val

    @staticmethod
    def _quantize_mean(mu: float, n_bits: int) -> int:
        return min(int(math.ceil(mu * (2 ** n_bits))), (1 << n_bits))

    @staticmethod
    def _dequantize_mean(quantized: int, n_bits: int) -> float:
        return quantized / (2 ** n_bits)
