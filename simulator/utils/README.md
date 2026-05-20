# utils/

シミュレーション結果の評価・可視化ユーティリティモジュール。

## 役割

- `Trace` と `PlayerState` から評価指標を計算する（`metrics.py`）。
- 実験結果のグラフ生成を担う（`plotter.py`）。

## 主要ファイル

| ファイル | 内容 |
|---------|------|
| `__init__.py` | `compute_metrics`, plotter helper をエクスポート |
| `metrics.py` | `compute_metrics` の実装 |
| `plotter.py` | regret curve / metric bar chart の保存 |

### `compute_metrics`

```python
metrics = compute_metrics(
    trace=runner.trace,
    player_states=player_states,
    means=means,
    M=M,
)
```

| キー | 説明 |
|-----|------|
| `total_steps` | シミュレーション総ステップ数 |
| `cumulative_regret` | 累積 regret |
| `collision_count` | collision が発生したステップ数 |
| `find_good_duration` | FindGoodArm フェーズのステップ数 |
| `rank_duration` | VirtualMusicalChairs フェーズのステップ数 |
| `number_players_duration` | VirtualNumberPlayers フェーズのステップ数 |
| `exploration_duration` | DistributedExploration フェーズのステップ数 |
| `rank_assignment_success` | 全プレイヤーが rank を得たか |
| `player_count_success` | 全プレイヤーの M_hat が M に一致するか |
| `final_assignment_success` | 全プレイヤーに arm が割り当てられたか |
| `assignment_duplicate` | 割当 arm に重複があるか |

### `plotter.py`

```python
save_regret_curve(curves_df, "outputs/figures/regret_huang_vs_izumi.png")
save_metric_bar(summary_df, "init_duration", "outputs/figures/init_duration_by_n.png")
```

`simulator.experiments.compare_homogeneous` から呼び出し、PNG を `outputs/runs/` に保存する。
regret curve は trial 間の平均値に加えて 95% CI を薄い帯で表示する。
`summary_df` を渡した場合は、各 phase の平均終了時刻を y 軸に並行な縦線として表示する。
