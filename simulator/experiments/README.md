# experiments/

比較実験スクリプトを置くディレクトリ。

## 役割

- 複数アルゴリズムを同一条件で実行し、実行ごとの結果を `outputs/runs/` に保存する。
- CSV、グラフ、実行設定 JSON を同じ run ディレクトリにまとめる。
- 実験設定（K, M, T, means, seed, trial 数）をスクリプト内で明示する。

## 主要ファイル

| スクリプト | 内容 |
|-----------|------|
| `compare_homogeneous.py` | Huang 2022 vs Izumi 2026 の cumulative regret 比較 |

### `compare_homogeneous.py`

```bash
python -m simulator.experiments.compare_homogeneous --experiment small --trials 20
# → outputs/runs/YYYYMMDD_HHMMSS_homogeneous_small_sanity_K5_M2_T50000/summary.csv
# → outputs/runs/YYYYMMDD_HHMMSS_homogeneous_small_sanity_K5_M2_T50000/curves.csv
# → outputs/runs/YYYYMMDD_HHMMSS_homogeneous_small_sanity_K5_M2_T50000/*.png
# → outputs/runs/YYYYMMDD_HHMMSS_homogeneous_small_sanity_K5_M2_T50000/run_config.json
```

- `--experiment`: `small`, `speedup`, `tradeoff` から選択。
- `--trials`: trial 数を上書き。
- `--horizon`: horizon `T` を上書き。
- `--K`, `--M`: arm 数・player 数を上書き。
- `--n-values`: Izumi 2026 の `n` 値をカンマ区切りで指定。
- `--means`: arm 平均報酬をカンマ区切りで直接指定。
- `--name-suffix`: 出力ファイル名の suffix。
- `--no-plots`: CSV のみ出力。
- `--retry-on-failure`: `final_assignment_success=0` の trial を seed を変えて再試行。
- `--max-attempts`: `--retry-on-failure` 時の最大 attempt 数。
- `--output-root`: run ディレクトリを作る親ディレクトリ。デフォルトは `outputs/runs`。
- 各 trial で seed をインクリメントして再現性を確保。
- summary CSV には algorithm, K, M, T, delta, n, trial, seed, regret, phase duration, success rate を含む。
- curves CSV には time ごとの cumulative regret を含む。

長時間実験で割当失敗 trial をやり直す例:

```bash
python -m simulator.experiments.compare_homogeneous \
  --experiment speedup \
  --trials 50 \
  --horizon 1000000 \
  --retry-on-failure \
  --max-attempts 5
```

再試行で破棄した attempt は CSV に混ぜない。全 attempt が失敗した場合は最後の
attempt を保存し、summary CSV の `retry_exhausted=1` で判別できる。

arm 数と player 数を変える例:

```bash
python -m simulator.experiments.compare_homogeneous \
  --experiment small \
  --K 7 \
  --M 3 \
  --n-values 1,2 \
  --horizon 80000 \
  --trials 20 \
  --name-suffix K7_M3
```

regret curve には trial 間の 95% CI を薄い帯で表示する。

## 出力先

| ディレクトリ | 用途 |
|------------|------|
| `outputs/runs/YYYYMMDD_HHMMSS_<experiment>/summary.csv` | trial 単位の集計 |
| `outputs/runs/YYYYMMDD_HHMMSS_<experiment>/curves.csv` | regret curve 用の時系列 |
| `outputs/runs/YYYYMMDD_HHMMSS_<experiment>/*.png` | グラフ |
| `outputs/runs/YYYYMMDD_HHMMSS_<experiment>/run_config.json` | 実行設定 |
