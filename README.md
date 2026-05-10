# mpmab-simu

Multi-Player Multi-Armed Bandit Simulator

no-sensing 設定（collision を観測できない）の MPMAB アルゴリズムを同期シミュレーションで実装・比較するためのリポジトリ。

## 実装状況

- **Step 1〜3 実装済み**: `BernoulliMPMABEnv`、`Runner`/`Trace`、`HomogeneousHuang2022`（Huang et al., 2022）
- Step 4 以降（Izumi 2026 など）は未実装。

## ディレクトリ構造

```
mpmab-simu/
├── envs/               # 環境（問題設定）
│   ├── __init__.py
│   └── bernoulli_mpmab.py   # BernoulliMPMABEnv, StepResult
├── algorithms/         # 論文単位のアルゴリズム実装
│   ├── __init__.py
│   ├── base.py              # BaseAlgorithm 抽象基底クラス
│   └── homogeneous_huang2022.py  # Huang et al. (2022) homogeneous 設定
├── core/               # シミュレーション実行エンジン・履歴管理
│   ├── __init__.py
│   ├── runner.py            # Runner, HorizonReached
│   └── trace.py             # Trace, StepRecord
├── utils/              # 評価指標・可視化ユーティリティ
│   ├── __init__.py
│   └── metrics.py           # compute_metrics
├── agents/             # 将来: 単体プレイヤー agent / baseline 用（現在未使用）
├── experiments/        # 比較実験スクリプト（現在未実装）
├── tests/              # pytest テスト
│   ├── test_env.py                      # BernoulliMPMABEnv (11 tests)
│   └── test_huang2022_initialization.py # HomogeneousHuang2022 初期化 (7 tests)
├── docs/               # 設計資料・実装メモ・擬似コード
├── papers/             # 参照論文の参考実装
├── results/            # 実験結果 CSV（実験スクリプト出力先）
├── figures/            # グラフ（実験スクリプト出力先）
├── main.py             # sanity check エントリーポイント
└── requirements.txt    # numpy, pandas, matplotlib, pytest
```

## クイックスタート

```bash
pip install -r requirements.txt

# sanity check (K=5, M=2, T=50000)
python main.py

# テスト
python -m pytest tests/ -v
```

## 設計方針

- arm index / player index は Python 内部で 0-based に統一。
- no-sensing: collision flag はアルゴリズムの意思決定に渡さない。
- 各ステップの記録は `Trace` に積み、`to_dataframe()` で pandas DataFrame として取得できる。
- 再現性のため、確率的処理はすべて seed を受け取る。
