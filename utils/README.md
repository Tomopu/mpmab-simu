# utils/

シミュレーション結果の評価・可視化ユーティリティモジュール。

## 役割

- `Trace` と `PlayerState` から評価指標を計算する（`metrics.py`）。
- 実験結果のグラフ生成を担う（`plotter.py`、未実装）。

## 主要ファイル

| ファイル | 内容 |
|---------|------|
| `__init__.py` | `compute_metrics` をエクスポート |
| `metrics.py` | `compute_metrics` の実装 |

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

## 今後追加予定

- `plotter.py`: cumulative regret の折れ線グラフ、フェーズ別 Gantt チャートなど。
