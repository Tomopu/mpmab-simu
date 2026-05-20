# outputs/runs/

比較実験 CLI の標準出力先。

```bash
python -m simulator.experiments.compare_homogeneous --experiment small --trials 20
```

実行ごとに `YYYYMMDD_HHMMSS_<experiment>/` を作り、CSV、PNG、`run_config.json` を同じディレクトリに保存する。

