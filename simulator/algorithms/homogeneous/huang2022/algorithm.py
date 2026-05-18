from __future__ import annotations

import random
from typing import Dict, List, Optional

from simulator.algorithms.homogeneous.huang2022.exploration import Huang2022ExplorationMixin
from simulator.algorithms.homogeneous.huang2022.phases import Huang2022PhaseMixin
from simulator.algorithms.homogeneous.huang2022.results import build_result
from simulator.algorithms.homogeneous.math_helpers import checked_log as _ln
from simulator.algorithms.homogeneous.math_helpers import ceil_int as _ceil
from simulator.core.runner import HorizonReached, Runner

class HomogeneousHuang2022(Huang2022PhaseMixin, Huang2022ExplorationMixin):
    """
    Huang et al. (2022) Homogeneous MPMAB without Collision Sensing の実装。

    使い方:
        algo = HomogeneousHuang2022(K=5, M=2, delta=1e-3, seed=42)
        result = algo.run(runner)

    Args:
        K: 腕の数
        M: プレイヤー数（真値）
        delta: 信頼度パラメータ δ
        seed: 乱数シード（再現性のため）
    """

    def __init__(self, K: int, M: int, delta: float, seed: Optional[int] = None) -> None:
        if K < 2:
            raise ValueError("K は 2 以上でなければならない。")
        if M < 1:
            raise ValueError("M は 1 以上でなければならない。")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta は (0, 1) の範囲でなければならない。")

        self.K = K
        self.M = M
        self.delta = delta

        # プレイヤーごとに独立な乱数生成器を派生させる
        master_rng = random.Random(seed)
        self._player_rngs: List[random.Random] = [
            random.Random(master_rng.randrange(1 << 30)) for _ in range(M)
        ]

    # ------------------------------------------------------------------
    # Algorithm 1: FindGoodArm
    # ------------------------------------------------------------------

    def run(self, runner: Runner) -> Dict[str, object]:
        """
        Algorithm 5 Proposed algorithm を実行する。

        手順:
          1. FindGoodArm で good arm k_tilde と報酬下界 mu_tilde を得る。
          2. VirtualMusicalChairs で各プレイヤーに external rank s を割り当てる。
             tau_rank = K * ln(1/delta) / mu_tilde
          3. VirtualNumberPlayers で内部 rank j と推定プレイヤー数 M_hat を得る。
             tau_comm = ln(1/delta) / mu_tilde
          4. DistributedExploration で各プレイヤーに top-M arm を割り当てる。

        HorizonReached について:
          各フェーズを個別に try/except で囲み、取得済みの状態を保持する。
          horizon に達した時点でそれ以前に完了したフェーズの結果を返す。
          未完了フェーズの出力は -1 のまま。

        Args:
            runner: Runner インスタンス（horizon 管理を含む）

        Returns:
            {
              "player_states": List[PlayerState],  各プレイヤーの最終状態
              "phase_durations": Dict[str, int],   フェーズごとの所要ステップ数
            }
        """
        K = self.K
        M = self.M
        delta = self.delta

        # フェーズ間で保持する状態（HorizonReached 時の部分結果）
        k_tilde = -1
        mu_tilde = 0.0
        s_list = [-1] * M
        j_list = [1] * M
        M_hat_list = [1] * M
        f_list = [-1] * M

        # 1. FindGoodArm
        try:
            k_tilde, mu_tilde = self.find_good_arm(runner)
        except HorizonReached:
            pass

        if k_tilde == -1:
            # FindGoodArm すら完了しなかった場合はそのまま返す
            return build_result(M, s_list, j_list, M_hat_list, k_tilde, mu_tilde, f_list, runner)

        # tau の計算（論文 Algorithm 5 の式）
        mu_safe = max(mu_tilde, 1e-12)
        tau_rank = _ceil(K * _ln(1.0 / delta) / mu_safe)
        tau_comm = _ceil(_ln(1.0 / delta) / mu_safe)

        # 2. VirtualMusicalChairs
        try:
            s_list = self.virtual_musical_chairs(runner, k_tilde, tau_rank)
        except HorizonReached:
            pass

        # 3. VirtualNumberPlayers
        try:
            M_hat_list, j_list = self.virtual_number_players(runner, k_tilde, s_list, tau_comm)
        except HorizonReached:
            pass

        # 4. DistributedExploration
        try:
            f_list = self.distributed_exploration(runner, k_tilde, j_list, M_hat_list, tau_comm)
        except HorizonReached:
            pass

        return build_result(M, s_list, j_list, M_hat_list, k_tilde, mu_tilde, f_list, runner)

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

