# outputs/

再生成可能な実験出力を置くディレクトリ。

## 構成

| ディレクトリ | 用途 |
|------------|------|
| `runs/` | 実行ごとの `summary.csv`, `curves.csv`, PNG, `run_config.json` |
| `results/` | 旧形式または手動集約した CSV |
| `figures/` | 旧形式または手動生成した PNG |

CSV と PNG は Git 管理対象外。必要な実験はコマンドと `run_config.json` から再生成する。

