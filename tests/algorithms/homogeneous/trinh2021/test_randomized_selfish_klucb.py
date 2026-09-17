"""
Randomized Selfish KL-UCB（Runner 版とバッチ版）のテスト。

確認内容:
- バッチ版の regret・衝突回数が、記録した行動列から計算し直した値と一致すること
- 各 trial の結果が、同じバッチに入れる trial や乱数ブロックの大きさに依存しないこと
- Runner 版が 2 人・2 arm の環境で互いに別の arm に落ち着くこと（Trinh and Combes 2021 の Figure 1 と同じ設定）
"""

import numpy as np

from simulator.algorithms.homogeneous import HomogeneousRandomizedSelfishKLUCB
from simulator.algorithms.homogeneous.trinh2021 import simulate_rskl_batch
from simulator.core.runner import Runner
from simulator.envs import BernoulliMPMABEnv


def test_batch_metrics_match_recorded_actions():
    # Given: 小さな環境（trial ごとに arm の並びが違う）
    means = [[0.9, 0.2, 0.6, 0.1, 0.4], [0.4, 0.1, 0.9, 0.6, 0.2], [0.2, 0.9, 0.1, 0.4, 0.6]]
    M, T = 3, 400

    # When: 行動を記録しながら実行する
    result = simulate_rskl_batch(means, M, T, seeds=[11, 12, 13], sample_times=[0, 200, T], record_actions=True)

    # Then: 行動列から計算し直した衝突ステップ数と期待値ベース regret が一致する
    actions = result.actions  # (T, B, M)
    means_arr = np.asarray(means)
    optimal = np.sort(means_arr, axis=1)[:, ::-1][:, :M].sum(axis=1)
    for b in range(len(means)):
        collision_steps = 0
        pseudo = 0.0
        for t in range(T):
            row = actions[t, b]
            dup = np.array([np.sum(row == a) >= 2 for a in row])
            collision_steps += int(dup.any())
            pseudo += optimal[b] - sum(means_arr[b, a] for a, d in zip(row, dup) if not d)
        assert result.collision_steps[b] == collision_steps
        assert abs(result.pseudo_regret[b] - pseudo) < 1e-8
        # 観測報酬ベースの regret = T * 最適報酬 - 観測報酬の合計
        assert abs(result.cumulative_regret[b] - (T * optimal[b] - result.total_reward[b])) < 1e-8
        # 曲線の最後の点は累積 regret と一致する
        assert abs(result.curve[b, -1] - result.cumulative_regret[b]) < 1e-8


def test_batch_result_does_not_depend_on_batch_or_block():
    # Given: 同じ trial（seed 22）を、別の trial と一緒に回す場合と単独で回す場合
    means = [0.8, 0.5, 0.3, 0.1]
    M, T = 2, 600

    # When
    together = simulate_rskl_batch([means, means], M, T, seeds=[21, 22], sample_times=[0, T], block=1024)
    alone = simulate_rskl_batch([means], M, T, seeds=[22], sample_times=[0, T], block=7)

    # Then: seed 22 の trial の結果は完全に一致する
    assert together.cumulative_regret[1] == alone.cumulative_regret[0]
    assert together.collision_steps[1] == alone.collision_steps[0]


def test_runner_version_orthogonalizes_two_players():
    # Given: M = 2, K = 2, mu = (0.9, 0.1)。Selfish KL-UCB が衝突し続けやすい設定
    means = [0.9, 0.1]
    T = 5000
    settled = 0
    for seed in range(5):
        env = BernoulliMPMABEnv(means=means, num_players=2, seed=seed)
        runner = Runner(env=env, horizon=T)
        algo = HomogeneousRandomizedSelfishKLUCB(K=2, M=2, seed=seed)

        # When
        result = algo.run(runner)

        # Then: 最後に 2 人が別の arm を選び、regret が小さい
        if len(set(result["last_actions"])) == 2 and runner.trace.cumulative_regret < 1000:
            settled += 1
    assert settled >= 4
