# Step 5〜6 Experiment Implementation Notes

作成日: 2026-05-10

このドキュメントは `docs/20260510_implementation_and_experiments.md` の Step 5〜6 として追加した、metrics / plotting / homogeneous 比較実験コードの実装メモです。

## 追加ファイル

| Path | 内容 |
|------|------|
| `experiments/compare_homogeneous.py` | Huang 2022 と Izumi 2026 を同一 homogeneous 設定で比較し、CSV と PNG を出力する CLI |
| `utils/plotter.py` | cumulative regret curve と metric bar chart を保存する helper |

## 更新ファイル

| Path | 内容 |
|------|------|
| `utils/__init__.py` | plotter helper を export |
| `experiments/README.md` | 実験スクリプトの実行方法と出力仕様を更新 |
| `utils/README.md` | `plotter.py` の説明を追加 |
| `README.md` | Step 4〜6 の実装状況と実験コマンドを追加 |
| `.gitignore` | 再生成可能な `results/*.csv`, `figures/*.png` を除外 |

## 実験コマンド

```bash
python experiments/compare_homogeneous.py --experiment small --trials 20
```

利用できる設定:

| `--experiment` | 内容 |
|----------------|------|
| `small` | 小規模 sanity check。デフォルト `K=5, M=2, T=50000` |
| `speedup` | multi-channel speedup 確認。`K=10, M=5` |
| `tradeoff` | good arm 探索コストと後続短縮の tradeoff 確認。`K=20, M=5` |

主なオプション:

| Option | 内容 |
|--------|------|
| `--trials N` | trial 数を上書き |
| `--horizon T` | horizon を上書き |
| `--sample-points N` | regret curve 用のサンプル点数 |
| `--no-plots` | CSV のみ出力 |

## 出力

```text
results/<experiment>.csv
results/<experiment>_curves.csv
figures/regret_huang_vs_izumi.png
figures/init_duration_by_n.png
figures/collision_count_by_n.png
figures/success_rate_by_n.png
```

`results/` と `figures/` の成果物は再生成可能なので Git 管理から除外する。

## 注意

`small` の horizon は、実装計画の初期案 `T=5000` では seed によって初期化中に horizon に達するため、デフォルトでは sanity check が完走しやすい `T=50000` にしている。短い horizon の切り詰め挙動を見たい場合は `--horizon 5000` を指定する。

forced-collision bit 通信はまだ完全再現ではないため、現時点の実験結果は simplified implementation の比較として扱う。
