"""
割当規則「案 II」（assignment_rule="handoff"）のテスト。

案 II（reviews/codex/07_assignment_rule_redesign.md の 4 節）:
- 探索から抜ける順番を π = (n, ..., 1, M, ..., n+1)（内部ランク）に固定する。
- フォロワーの番に通信腕が受理されたら、非通信腕を持つ割当済みリーダーが通信腕を取り、旧い腕をフォロワーに渡す。

確認内容:
- 判定のたびに、未割当集合が π の末尾になっている（hde_tail_violations == 0）。
- フォロワーは一度も通信腕を持たない（判定の直前の割当表で確認）。
- 引継ぎ元のリーダーが見つからない事態が起きない（hde_handoff_failures == 0）。
- 最終的に全員が相異なる上位 M 本を持つ。
- 引継ぎが実際に起きる設定がある。
- n = 1 は割当規則によらず同じ軌跡（Huang 型の分岐）。
"""
import inspect

import pytest

from simulator.algorithms.homogeneous import HomogeneousMultiChannelIzumi2026
from simulator.core.runner import Runner
from simulator.envs.bernoulli_mpmab import BernoulliMPMABEnv

K = 9
HORIZON = 5_000_000
DELTA = 0.1


def run_hde(means, M, n, channels, ranks, rule, seed=0):
    """初期化を飛ばして HDE だけを走らせ、判定の直前の割当表を記録する。"""
    env = BernoulliMPMABEnv(means=means, num_players=M, seed=seed)
    runner = Runner(env=env, horizon=HORIZON)
    algo = HomogeneousMultiChannelIzumi2026(K=len(means), M=M, n=n, delta=DELTA, seed=seed, assignment_rule=rule)
    snapshots = []
    original = algo._compute_accept_reject

    def observe(mu_hat, N_mat, active_arms, M0, delta, p):
        local = inspect.currentframe().f_back.f_locals
        snapshots.append(list(local["f"]))
        return original(mu_hat, N_mat, active_arms, M0, delta, p)

    algo._compute_accept_reject = observe
    f = algo.hierarchical_distributed_exploration(runner, channels, ranks, [M] * M, 1)
    return algo, f, snapshots, runner


# (平均, M, n, 通信腕) の組。通信腕に上位外の腕を含む場合と、通信腕が遅れて受理される場合を入れる。
CASES = [
    ([0.9, 0.85, 0.8, 0.75, 0.3, 0.2, 0.1, 0.05, 0.04], 4, 2, [0, 1]),        # 通信腕が上位
    ([0.9, 0.85, 0.8, 0.75, 0.3, 0.2, 0.1, 0.05, 0.04], 4, 2, [2, 3]),        # 通信腕が上位の下の方（遅れて受理されやすい）
    ([0.9, 0.85, 0.8, 0.75, 0.3, 0.2, 0.1, 0.05, 0.04], 4, 2, [0, 4]),        # 通信腕 4 は上位外
    ([0.9, 0.85, 0.8, 0.75, 0.7, 0.2, 0.1, 0.05, 0.04], 5, 3, [3, 4, 5]),     # n = 3、通信腕 5 は上位外
    ([0.9, 0.85, 0.8, 0.75, 0.7, 0.2, 0.1, 0.05, 0.04], 5, 3, [0, 1, 2]),
]


@pytest.mark.parametrize("means,M,n,channels", CASES)
@pytest.mark.parametrize("seed", [0, 1])
def test_handoff_invariants(means, M, n, channels, seed):
    ranks = list(range(1, M + 1))[::-1]  # 物理 ID と内部ランクをずらす
    algo, f, snapshots, _ = run_hde(means, M, n, channels, ranks, "handoff", seed)
    top = sorted(range(len(means)), key=lambda k: -means[k])[:M]
    assert sorted(f) == sorted(top), "全員が相異なる上位 M 本を持つ"
    assert algo.hde_tail_violations == 0
    assert algo.hde_handoff_failures == 0
    chan = set(channels)
    for snap in snapshots + [list(f)]:
        for m, arm in enumerate(snap):
            if ranks[m] > n:
                assert arm not in chan, "フォロワーは通信腕を持たない"


def test_handoff_happens():
    """通信腕が非通信腕より後に受理される設定では、引継ぎが起きる。"""
    means = [0.9, 0.85, 0.8, 0.75, 0.3, 0.2, 0.1, 0.05, 0.04]
    total = 0
    for seed in range(3):
        algo, f, _, _ = run_hde(means, 4, 2, [2, 3], [1, 2, 3, 4], "handoff", seed)
        total += algo.hde_handoff_count
    assert total > 0


def test_n1_is_unchanged_by_rule():
    """n = 1 は Huang 型の分岐なので、割当規則を変えても軌跡は同じ。"""
    means = [0.9, 0.85, 0.8, 0.75, 0.3, 0.2, 0.1, 0.05, 0.04]
    runs = []
    for rule in ("channel_owner", "handoff"):
        algo, f, _, runner = run_hde(means, 4, 1, [0], [1, 2, 3, 4], rule, seed=5)
        runs.append((f, runner.elapsed, runner.trace.to_records()[-1]))
    assert runs[0] == runs[1]


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        HomogeneousMultiChannelIzumi2026(K=9, M=4, n=2, delta=0.1, assignment_rule="nope")
