# experiments/

比較実験スクリプトを置くディレクトリ。

## 役割

- 複数アルゴリズムを同一条件で実行し、結果を `results/` に CSV で保存する。
- グラフを `figures/` に出力する。
- 実験設定（K, M, T, means, seed, trial 数）をスクリプト内で明示する。

## 今後追加予定

| スクリプト | 内容 |
|-----------|------|
| `compare_homogeneous.py` | Huang 2022 vs Izumi 2026 の cumulative regret 比較 |

### `compare_homogeneous.py` の予定仕様

```bash
python experiments/compare_homogeneous.py
# → results/compare_homogeneous_YYYYMMDD.csv
# → figures/compare_homogeneous_regret.png
```

- 設定: K, M, T, means, delta, trial 数をスクリプト冒頭に定数として記述。
- 各 trial で seed をインクリメントして再現性を確保。
- 出力 CSV には trial, algorithm, step, cumulative_regret, phase, collision_count を含む。

## 出力先

| ディレクトリ | 用途 |
|------------|------|
| `results/` | CSV 形式の実験結果 |
| `figures/` | PNG 形式のグラフ |
