# mpmab-simu

Multi-player Multi-armed Bandit Simulator

## ディレクトリ構造

```
mp_mab_simulator/
├── envs/                   # 環境（問題設定）に関するディレクトリ
│   ├── __init__.py
│   ├── base_env.py         # 環境の抽象基底クラス
│   └── mab_env.py          # 4つの設定を統合して扱う環境クラス
├── agents/                 # エージェント（アルゴリズム）に関するディレクトリ
│   ├── __init__.py
│   ├── base_agent.py       # エージェントの抽象基底クラス
│   ├── baselines.py        # 比較対象となる既存手法 (UCB, ε-greedyなど)
│   └── musical_chairs.py   # 提案手法 (外部ランク同定、Leader-Follower通信など)
├── core/                   # シミュレーションの実行エンジン
│   ├── __init__.py
│   └── runner.py           # 並列処理やループを回し、結果を記録するクラス
├── utils/                  # 補助ツール
│   ├── __init__.py
│   ├── metrics.py          # Regretや正解アーム選択率の計算
│   └── plotter.py          # matplotlibを用いたグラフ描画用関数群
├── experiments/            # 実験の実行と設定を記述するスクリプト群
│   ├── exp_homo_sensing.py
│   └── exp_hetero_no_sensing.py
├── requirements.txt        # 必要なライブラリ (numpy, joblib, matplotlib 等)
└── main.py                 # サクッとテスト実行するためのエントリーポイント
```