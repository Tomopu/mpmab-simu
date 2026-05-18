# core/

シミュレーションの実行エンジンと履歴管理モジュール。

アルゴリズムと環境を繋ぐ同期インターフェース（Runner）と、全ステップの記録（Trace）を提供する。

## 役割

- `Runner` を通してアルゴリズムが環境に action を送り、結果を受け取る。
- horizon 到達時に `HorizonReached` 例外を送出し、途中フェーズでも停止できる。
- 各ステップの記録を `Trace` に積み、フェーズ別の所要時間や累積 regret を管理する。

## 主要ファイル

| ファイル | 内容 |
|---------|------|
| `__init__.py` | `Trace`, `Runner` をエクスポート |
| `runner.py` | `Runner` と `HorizonReached` の実装 |
| `trace.py` | `Trace` と `StepRecord` の実装 |

### `Runner`

```python
runner = Runner(env=env, horizon=T)
runner.set_phase("find_good_arm")
result = runner.step(actions)  # HorizonReached を送出することがある
```

- `step(actions)` は内部で `env.step()` を呼び、結果を `Trace` に記録する。
- `elapsed` / `remaining`: 経過・残りステップ数。

### `Trace`

```python
trace = runner.trace
df = trace.to_dataframe()            # pandas DataFrame
durations = trace.phase_durations    # {"find_good_arm": 900, ...}
```

- `phase_durations`: フェーズごとの所要ステップ数の dict。
- `to_records()`: 全記録を dict のリストで返す。
- `to_dataframe()`: pandas DataFrame で返す（pandas が必要）。

## 今後追加予定

現状の実装で Step 1〜3 の要件を満たしている。追加予定はなし。
