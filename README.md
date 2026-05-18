# mpmab-simu

Multi-Player Multi-Armed Bandit Simulator

no-sensing 設定（collision を観測できない）の MPMAB アルゴリズムを同期シミュレーションで実装・比較するためのリポジトリ。

## 実装状況

- **Step 1〜3 実装済み**: `BernoulliMPMABEnv`、`Runner`/`Trace`、`HomogeneousHuang2022`（Huang et al., 2022）
- **Step 4 実装済み**: `HomogeneousMultiChannelIzumi2026`（Izumi et al., 2026）
- **Step 5〜6 実装済み**: metrics、比較実験 CSV 出力、グラフ生成、pytest
- 注意: forced-collision bit 通信はまだ完全再現ではなく、通信結果の直接集約 + 通信時間コストの簡略実装。

## ディレクトリ構造

```
mpmab-simu/
├── simulator/          # シミュレーター本体
│   ├── envs/           # 環境（問題設定）
│   ├── core/           # 実行エンジン・履歴管理
│   ├── algorithms/     # 論文単位のアルゴリズム実装
│   │   └── homogeneous/
│   │       ├── huang2022.py
│   │       ├── izumi2026.py
│   │       ├── states.py
│   │       └── math_helpers.py
│   ├── utils/          # 評価指標・可視化
│   ├── experiments/    # 比較実験 CLI
│   └── main.py         # sanity check エントリーポイント
├── tests/              # pytest テスト
│   ├── envs/
│   ├── algorithms/
│   └── invariants/
├── docs/               # 設計資料・実装メモ・擬似コード
├── papers/             # 参照論文・参考実装（中身は gitignore）
├── outputs/            # 再生成可能な実験出力（gitignore）
│   ├── runs/
│   ├── results/
│   └── figures/
└── requirements.txt    # numpy, pandas, matplotlib, pytest
```

## クイックスタート

```bash
pip install -r requirements.txt

# sanity check (K=5, M=2, T=50000)
python -m simulator.main

# テスト
python -m pytest tests/ -v

# 比較実験（CSV と PNG を生成）
python -m simulator.experiments.compare_homogeneous --experiment small --trials 20
```

## 設計方針

- arm index / player index は Python 内部で 0-based に統一。
- no-sensing: collision flag はアルゴリズムの意思決定に渡さない。
- 各ステップの記録は `Trace` に積み、`to_dataframe()` で pandas DataFrame として取得できる。
- 再現性のため、確率的処理はすべて seed を受け取る。
