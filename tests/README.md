# tests/

pytest によるユニットテストを置くディレクトリ。

## 役割

- 環境（`envs/`）とアルゴリズム（`algorithms/`）の動作を確認する。
- 確率的テストは seed を固定して再現性を保証する。
- 実験スクリプト（`experiments/`）のテストはここには置かず、スクリプト内の sanity check で代替する。

## テストファイル

| ファイル | 対象 | テスト数 |
|---------|------|---------|
| `test_env.py` | `BernoulliMPMABEnv` | 11 |
| `test_huang2022_initialization.py` | `HomogeneousHuang2022` の初期化フェーズ | 7 |
| `test_izumi2026_initialization.py` | `HomogeneousMultiChannelIzumi2026` の初期化フェーズ | 16 |

### 実行方法

```bash
python -m pytest tests/ -v
```

### `test_env.py` の項目

- Collision: 2 人以上が同じ arm を選んだとき全員 reward = 0
- Collision（3 人）: 3 人 collision でも同様
- Partial collision: 一部だけ collision のとき non-collision プレイヤーは正常報酬
- No collision: 異なる arm を選んだとき通常の Bernoulli 報酬
- Reproducibility: 同一 seed で同一結果
- Reset reproducibility: reset 後に同一結果
- Different seeds: 異なる seed で異なる結果
- Optimal reward: top-M arm の平均報酬和と一致
- Instant regret non-negative: regret が負にならない
- Invalid arm index: 範囲外の arm index で ValueError
- Invalid actions length: プレイヤー数と異なる actions で ValueError

### `test_huang2022_initialization.py` の項目

- FindGoodArm が正の平均を持つ arm を返す
- FindGoodArm の mu_tilde が下限として妥当
- VirtualMusicalChairs が rank を割り当てる
- VirtualMusicalChairs の rank が概ね重複なし（10 試行中 5 回以上）
- full run で assigned arm に重複なし
- full run で全プレイヤーが assignment を得る
- full run で phase_durations が記録される

### `test_izumi2026_initialization.py` の項目

- FindMultipleGoodArms が n 本の arm を返す
- FindMultipleGoodArms が正の平均を持つ arm を返す
- FindMultipleGoodArms の mu_tilde が下限として妥当
- FindMultipleGoodArms が重複のない arm set を返す
- ParallelVirtualMusicalChairs で同一ブロック内の good arm pull が重複しない
- ParallelVirtualMusicalChairs が rank を割り当てる
- ParallelVirtualMusicalChairs の rank が概ね重複なし
- ParallelVirtualNumberPlayers のスロット制約が守られる
- ParallelVirtualNumberPlayers の M_hat が妥当な範囲に入る
- full run で assigned arm に重複なし
- full run で全プレイヤーが assignment を得る
- full run で phase_durations が記録される
- PlayerStateIzumi に good_arms が記録される
- n=1 の Izumi 2026 が assignment に成功する
- n=1 の Izumi 2026 と Huang 2022 がどちらも成功する
- n=1 の Izumi 2026 で assigned arm に重複なし

## 今後追加予定

- 比較実験スクリプトの smoke test。
- Step 5/6 実装後の metrics / plotting / CSV 出力テスト。
