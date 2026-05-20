# simulator/

MPMAB シミュレーター本体の Python パッケージ。

## 構成

| ディレクトリ | 用途 |
|------------|------|
| `envs/` | Bernoulli MPMAB などの問題設定 |
| `core/` | `Runner` と `Trace` による同期実行・履歴管理 |
| `algorithms/` | 論文アルゴリズムの実装 |
| `utils/` | 評価指標と描画 helper |
| `experiments/` | 比較実験 CLI と実験実行ロジック |
| `agents/` | 将来の player agent / baseline 用予約領域 |

## 実行

```bash
python -m simulator.main
python -m simulator.experiments.compare_homogeneous --experiment small --trials 20
```

