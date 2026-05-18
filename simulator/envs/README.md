# envs/

MPMAB シミュレーターの環境（問題設定）モジュール。

各時刻に全プレイヤーの action を同時に受け取り、collision 判定・報酬生成・regret 計算を行う。

## 役割

- Bernoulli 報酬を持つ MPMAB 問題の同期シミュレーション環境を提供する。
- アルゴリズム側に渡す情報を「rewards のみ（no-sensing）」に制限する責務を持つ。
  collision フラグは StepResult に記録するが、アルゴリズムには渡さない。

## 主要ファイル

| ファイル | 内容 |
|---------|------|
| `__init__.py` | `BernoulliMPMABEnv`, `StepResult` をエクスポート |
| `bernoulli_mpmab.py` | `BernoulliMPMABEnv` と `StepResult` の実装 |

### `BernoulliMPMABEnv`

```python
env = BernoulliMPMABEnv(means=[0.9, 0.8, 0.1], num_players=2, seed=42)
env.reset()
result = env.step([0, 1])  # actions: 各プレイヤーの arm index (0-based)
```

- `means`: 各 arm の平均報酬（shape `(K,)`）。homogeneous 設定では全プレイヤーが同じ分布を共有。
- `step(actions)` → `StepResult`: 同一 arm を 2 人以上が選んだ場合は全員 reward = 0。
- `optimal_total_reward`: top-M arm の平均報酬和（初期化時に計算）。

## 今後追加予定

- `HeterogeneousMPMABEnv`: プレイヤーごとに異なる報酬行列 `means[m, k]` を持つ環境（heterogeneous 設定）。
