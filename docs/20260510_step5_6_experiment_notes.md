# Step 5〜6 Experiment Implementation Notes

作成日: 2026-05-10

このドキュメントは `docs/20260510_implementation_and_experiments.md` の Step 5〜6 として追加した、metrics / plotting / homogeneous 比較実験コードの実装メモです。

## 追加ファイル

| Path | 内容 |
|------|------|
| `simulator/experiments/compare_homogeneous.py` | Huang 2022 と Izumi 2026 を同一 homogeneous 設定で比較し、CSV と PNG を出力する CLI |
| `simulator/utils/plotter.py` | cumulative regret curve と metric bar chart を保存する helper |

## 更新ファイル

| Path | 内容 |
|------|------|
| `simulator/utils/__init__.py` | plotter helper を export |
| `experiments/README.md` | 実験スクリプトの実行方法と出力仕様を更新 |
| `simulator/utils/README.md` | `plotter.py` の説明を追加 |
| `README.md` | Step 4〜6 の実装状況と実験コマンドを追加 |
| `.gitignore` | 再生成可能な `runs/` を除外 |

## 実験コマンド

```bash
python simulator/experiments/compare_homogeneous.py --experiment small --trials 20
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
| `--K K` | arm 数を上書き |
| `--M M` | player 数を上書き |
| `--n-values 1,2,3` | Izumi 2026 で比較する `n` の値を上書き |
| `--means 0.9,0.8,...` | arm 平均報酬を直接指定 |
| `--mean-high X` | 自動生成 means の最大値 |
| `--mean-low X` | 自動生成 means の最小値 |
| `--name-suffix NAME` | 出力ファイル名に suffix を付ける |
| `--ci 0.95` | regret curve に描く信頼区間 |
| `--no-plots` | CSV のみ出力 |
| `--output-root runs` | run ディレクトリを作成する親ディレクトリ |

## 出力

```text
runs/YYYYMMDD_HHMMSS_<experiment>_<suffix>/summary.csv
runs/YYYYMMDD_HHMMSS_<experiment>_<suffix>/curves.csv
runs/YYYYMMDD_HHMMSS_<experiment>_<suffix>/regret_huang_vs_izumi.png
runs/YYYYMMDD_HHMMSS_<experiment>_<suffix>/init_duration_by_n.png
runs/YYYYMMDD_HHMMSS_<experiment>_<suffix>/collision_count_by_n.png
runs/YYYYMMDD_HHMMSS_<experiment>_<suffix>/success_rate_by_n.png
runs/YYYYMMDD_HHMMSS_<experiment>_<suffix>/run_config.json
```

`runs/` の成果物は再生成可能なので Git 管理から除外する。実行ごとに日付時刻つきのディレクトリを作り、CSV と PNG を同じ場所に保存する。

## 注意

## 実行例

### 1. 小規模 sanity check

```bash
python simulator/experiments/compare_homogeneous.py \
  --experiment small \
  --trials 20 \
  --sample-points 300
```

Huang 2022 と Izumi 2026 (`n=1,2`) を `K=5, M=2, T=50000` で比較する。

### 2. arm 数と player 数を変える

```bash
python simulator/experiments/compare_homogeneous.py \
  --experiment small \
  --K 7 \
  --M 3 \
  --n-values 1,2 \
  --horizon 80000 \
  --trials 20 \
  --name-suffix K7_M3
```

`K` と `M` を変える場合、Izumi 2026 の現在の実装制約により `1 <= n < K-M` を満たす必要がある。

### 3. 報酬分布を直接指定する

```bash
python simulator/experiments/compare_homogeneous.py \
  --experiment small \
  --K 8 \
  --M 3 \
  --means 0.95,0.9,0.82,0.7,0.5,0.3,0.15,0.05 \
  --n-values 1,2,3 \
  --horizon 100000 \
  --trials 30 \
  --name-suffix custom_means
```

`--means` を指定しない場合は `--mean-high` から `--mean-low` までを線形に並べた降順 means を自動生成する。

### 4. 短い horizon の切り詰め挙動を見る

```bash
python simulator/experiments/compare_homogeneous.py \
  --experiment small \
  --horizon 5000 \
  --trials 20 \
  --name-suffix short_horizon
```

`T=5000` では初期化中に horizon に達する seed が多く、成功率が下がる。これはアルゴリズムの最終性能比較というより、短い horizon における初期化コストの影響を見るための設定。

## 信頼区間

`regret_huang_vs_izumi` では、trial ごとの cumulative regret から平均値を描き、薄い帯として 95% CI を描く。

現在の計算は正規近似:

```text
mean ± 1.96 * std / sqrt(trials)
```

trial 数が少ない場合、信頼区間は参考値として扱う。論文用の図では `--trials 50` 以上を推奨する。

## 前回グラフとの差分

`small` の horizon は、実装計画の初期案 `T=5000` では seed によって初期化中に horizon に達するため、デフォルトでは sanity check が完走しやすい `T=50000` にしている。短い horizon の切り詰め挙動を見たい場合は `--horizon 5000` を指定する。

そのため、以前の `T=5000` の図では:

- 初期化が完走しない
- final assignment success rate が 0 になりやすい
- regret が horizon まで上がり続ける

一方、`T=50000` の図では:

- 初期化と割当が完了する
- success rate が 1 になる seed が増える
- 割当後は regret curve が水平になる

この違いにより、グラフの見た目が大きく変わる。

forced-collision bit 通信はまだ完全再現ではないため、現時点の実験結果は simplified implementation の比較として扱う。
