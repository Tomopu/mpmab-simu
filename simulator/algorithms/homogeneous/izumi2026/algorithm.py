from __future__ import annotations

import random
from typing import Dict, List, Optional

from simulator.algorithms.homogeneous.izumi2026.algorithm1_find_multiple_good_arms import (
    Izumi2026FindMultipleGoodArmsMixin,
)
from simulator.algorithms.homogeneous.izumi2026.algorithm2_parallel_virtual_musical_chairs import (
    Izumi2026ParallelVirtualMusicalChairsMixin,
)
from simulator.algorithms.homogeneous.izumi2026.algorithm3_parallel_virtual_number_players import (
    Izumi2026ParallelVirtualNumberPlayersMixin,
)
from simulator.algorithms.homogeneous.izumi2026.algorithm4_hierarchical_distributed_exploration import (
    Izumi2026HierarchicalDistributedExplorationMixin,
)
from simulator.algorithms.homogeneous.izumi2026.results import build_result
from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.algorithms.homogeneous.math_helpers import checked_log as _ln
from simulator.core.runner import HorizonReached, Runner


class HomogeneousMultiChannelIzumi2026(
    Izumi2026FindMultipleGoodArmsMixin,
    Izumi2026ParallelVirtualMusicalChairsMixin,
    Izumi2026ParallelVirtualNumberPlayersMixin,
    Izumi2026HierarchicalDistributedExplorationMixin,
):
    """
    Izumi et al. (2026) Homogeneous Multi-Channel MPMAB without Collision Sensing の実装。

    使い方:
        algo = HomogeneousMultiChannelIzumi2026(K=5, M=2, n=2, delta=1e-3, seed=42)
        result = algo.run(runner)

    Args:
        K: 腕の数
        M: プレイヤー数（真値）
        n: FindMultipleGoodArms で探す good arm の数。n < K-M が必要。
        delta: 信頼度パラメータ δ
        seed: 乱数シード（再現性のため）
    """

    def __init__(
        self,
        K: int,
        M: int,
        n: int,
        delta: float,
        seed: Optional[int] = None,
        allow_out_of_range_n: bool = False,
        assignment_rule: str = "channel_owner",
    ) -> None:
        if K < 2:
            raise ValueError("K は 2 以上でなければならない。")
        if M < 1:
            raise ValueError("M は 1 以上でなければならない。")
        if n < 1:
            raise ValueError("n は 1 以上でなければならない。")
        if n >= K - M:
            raise ValueError(f"n < K-M が必要。n={n}, K-M={K - M}")
        # 論文の仮定 (A1): n <= M, 2n <= K。範囲外は allow_out_of_range_n=True のときだけ許す。
        if not allow_out_of_range_n:
            if n > M:
                raise ValueError(f"n <= M が必要。n={n}, M={M}（範囲外を許すなら allow_out_of_range_n=True）")
            if 2 * n > K:
                raise ValueError(f"2n <= K が必要。n={n}, K={K}（範囲外を許すなら allow_out_of_range_n=True）")
        self.allow_out_of_range_n = allow_out_of_range_n
        # n >= 2 の割当規則。"channel_owner"：受理された通信腕は担当リーダーへ（論文の現行規則）。
        # "handoff"：Codex の案 II（抜ける順番を固定し、通信腕はリーダー間で引継ぐ）。
        # reviews/codex/07_assignment_rule_redesign.md の 4 節。n = 1 はどちらでも Huang 型のまま。
        if assignment_rule not in ("channel_owner", "handoff"):
            raise ValueError(f"unknown assignment_rule: {assignment_rule}")
        self.assignment_rule = assignment_rule
        if not (0.0 < delta < 1.0):
            raise ValueError("delta は (0, 1) の範囲でなければならない。")

        self.K = K
        self.M = M
        self.n = n
        self.delta = delta

        # プレイヤーごとに独立な乱数生成器を派生させる
        master_rng = random.Random(seed)
        self._player_rngs: List[random.Random] = [random.Random(master_rng.randrange(1 << 30)) for _ in range(M)]

    def run(self, runner: Runner) -> Dict[str, object]:
        """
        Algorithm 5 ProposedParallelAlgorithm を実行する。

        手順:
            1. FindMultipleGoodArms で n 本の good arm G と下界 mu_tilde を得る。
            2. tau = ceil(ln(1/delta) / min(mu_tilde))
            3. ParallelVirtualMusicalChairs で各プレイヤーに external rank s を割り当てる。
            4. ParallelVirtualNumberPlayers で内部 rank j と推定プレイヤー数 M_hat を得る。
            5. HierarchicalDistributedExploration で各プレイヤーに top-M arm を割り当てる。

        HorizonReached について:
            各フェーズを個別に try/except で囲み、取得済み状態を保持する。
            horizon に達した時点でそれ以前に完了したフェーズの結果を返す。

        Args:
            runner: Runner インスタンス

        Returns:
            {
                "player_states": List[PlayerStateIzumi],
                "phase_durations": Dict[str, int],
            }
        """
        M = self.M
        delta = self.delta

        # フェーズ間で保持する状態（HorizonReached 時の部分結果）
        good_arms: List[int] = []
        mu_tilde_map: Dict[int, float] = {}
        s_list = [-1] * M
        j_list = [1] * M
        M_hat_list = [1] * M
        f_list = [-1] * M

        # 初期化の確率的な失敗（内部ランクの誤推定で j=1 がいない等）は例外にせず、
        # 理由を残して失敗結果として返す。以後の損失は評価側（compute_metrics）が
        # 「残り時間 × 固定割当の期待損失」として数える。
        init_failure_reason: Optional[str] = None

        # 1. FindMultipleGoodArms
        try:
            good_arms, mu_tilde_map = self.find_multiple_good_arms(runner)
        except HorizonReached:
            pass

        if not good_arms:
            # FindMultipleGoodArms すら完了しなかった場合はそのまま返す
            return self._build(M, good_arms, mu_tilde_map, s_list, j_list, M_hat_list, f_list, runner, None)

        # 評価用の保守的な規約: 通信路の順序つきリストと各腕の下界が全プレイヤーで一致しなければ、
        # この試行をここで失敗として止める（以後は未割当のまま。残り時間は評価側が最大損失で数える）。
        # これはプレイヤーが使える通信操作ではない。不一致後にプレイヤー 0 の G で続けた軌跡は
        # 元の分散アルゴリズムを表さないので、評価しない。
        disagreement = self._initialization_disagreement()
        if disagreement is not None:
            return self._build(M, good_arms, mu_tilde_map, s_list, j_list, M_hat_list, f_list, runner, disagreement)

        # tau の計算
        # 論文: tilde_mu_min <- min_{k in G} tilde_mu[k]
        mu_min = min(mu_tilde_map.get(k, 1e-12) for k in good_arms)
        mu_safe = max(mu_min, 1e-12)
        # tau_rank = ceil(K * ln(1/delta) / mu_safe): ParallelVMC に渡す τ
        # Huang2022 と同じ式。ParallelVMC 総ステップ = ceil(K * tau_rank / n) で n 倍高速化。
        tau_rank = _ceil(self.K * _ln(1.0 / delta) / mu_safe)
        # tau_comm = ceil(ln(1/delta) / mu_safe): VNP・HDE に渡す τ
        tau_comm = _ceil(_ln(1.0 / delta) / mu_safe)

        # 2. ParallelVirtualMusicalChairs
        try:
            s_list = self.parallel_virtual_musical_chairs(runner, good_arms, tau_rank)
        except HorizonReached:
            pass
        except ValueError as e:
            init_failure_reason = f"ParallelVirtualMusicalChairs: {e}"

        # 3. ParallelVirtualNumberPlayers
        if init_failure_reason is None:
            try:
                M_hat_list, j_list = self.parallel_virtual_number_players(runner, good_arms, s_list, tau_comm)
            except HorizonReached:
                pass
            except ValueError as e:
                init_failure_reason = f"ParallelVirtualNumberPlayers: {e}"

        # 4. HierarchicalDistributedExploration
        if init_failure_reason is None:
            try:
                f_list = self.hierarchical_distributed_exploration(runner, good_arms, j_list, M_hat_list, tau_comm)
            except HorizonReached:
                pass
            except ValueError as e:
                init_failure_reason = f"HierarchicalDistributedExploration: {e}"

        return self._build(M, good_arms, mu_tilde_map, s_list, j_list, M_hat_list, f_list, runner, init_failure_reason)

    def _initialization_disagreement(self) -> Optional[str]:
        """FindMultipleGoodArms の出力（順序つき G と各腕の下界）が全プレイヤーで一致しなければ理由を返す。"""
        lists = getattr(self, "fmga_player_good_arms", None)
        bounds = getattr(self, "fmga_player_mu_tilde", None)
        if not lists or not bounds:
            return None
        if any(g != lists[0] for g in lists):
            return f"FindMultipleGoodArms: players disagree on the ordered channel list {lists}"
        if any(b != bounds[0] for b in bounds):
            return f"FindMultipleGoodArms: players disagree on the lower bounds {bounds}"
        return None

    def _build(self, M, good_arms, mu_tilde_map, s_list, j_list, M_hat_list, f_list, runner, init_failure_reason):
        """プレイヤーごとの Good Arm（合意の記録用）と失敗理由を含めて結果を組み立てる。"""
        return build_result(
            M, good_arms, mu_tilde_map, s_list, j_list, M_hat_list, f_list, runner,
            player_good_arms=getattr(self, "fmga_player_good_arms", None),
            player_mu_tilde=getattr(self, "fmga_player_mu_tilde", None),
            init_failure_reason=init_failure_reason,
        )
