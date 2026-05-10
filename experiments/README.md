# experiments/

比較実験スクリプトを置くディレクトリ。

## 役割

- 複数アルゴリズムを同一条件で実行し、結果を `results/` に CSV で保存する。
- グラフを `figures/` に出力する。
- 実験設定（K, M, T, means, seed, trial 数）をスクリプト内で明示する。

## 主要ファイル

| スクリプト | 内容 |
|-----------|------|
| `compare_homogeneous.py` | Huang 2022 vs Izumi 2026 の cumulative regret 比較 |

### `compare_homogeneous.py`

```bash
python experiments/compare_homogeneous.py --experiment small --trials 20
# → results/homogeneous_small_sanity.csv
# → results/homogeneous_small_sanity_curves.csv
# → figures/regret_huang_vs_izumi.png
# → figures/init_duration_by_n.png
# → figures/collision_count_by_n.png
# → figures/success_rate_by_n.png
```

- `--experiment`: `small`, `speedup`, `tradeoff` から選択。
- `--trials`: trial 数を上書き。
- `--horizon`: horizon `T` を上書き。
- `--no-plots`: CSV のみ出力。
- 各 trial で seed をインクリメントして再現性を確保。
- summary CSV には algorithm, K, M, T, delta, n, trial, seed, regret, phase duration, success rate を含む。
- curves CSV には time ごとの cumulative regret を含む。

## 出力先

| ディレクトリ | 用途 |
|------------|------|
| `results/` | CSV 形式の実験結果 |
| `figures/` | PNG 形式のグラフ |
